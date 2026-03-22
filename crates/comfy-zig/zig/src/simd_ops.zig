// comfy-zig: SIMD-optimized kernel operations
//
// Exports C ABI functions for SiLU, Softmax, and LayerNorm using
// Zig's @Vector SIMD intrinsics. These are compiled only when the
// `zig-native` Cargo feature is enabled.
//
// All functions operate on f32 arrays in-place or to output buffers.

const std = @import("std");
const math = std.math;

/// SIMD vector width for f32 operations (8 = 256-bit AVX).
const VEC_WIDTH = 8;
const VecF32 = @Vector(VEC_WIDTH, f32);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a VecF32 from a slice at the given offset.
/// If fewer than VEC_WIDTH elements remain, pads with `pad`.
inline fn load_vec(data: [*]const f32, offset: usize, len: usize, pad: f32) VecF32 {
    var v: VecF32 = @splat(pad);
    const remaining = if (offset < len) len - offset else 0;
    const count = @min(remaining, VEC_WIDTH);
    for (0..count) |i| {
        v[i] = data[offset + i];
    }
    return v;
}

/// Store a VecF32 into a slice at the given offset.
/// Only writes min(VEC_WIDTH, remaining) elements.
inline fn store_vec(out: [*]f32, offset: usize, len: usize, v: VecF32) void {
    const remaining = if (offset < len) len - offset else 0;
    const count = @min(remaining, VEC_WIDTH);
    for (0..count) |i| {
        out[offset + i] = v[i];
    }
}

/// Horizontal sum of a VecF32.
inline fn hsum(v: VecF32) f32 {
    return @reduce(.Add, v);
}

/// Horizontal max of a VecF32.
inline fn hmax(v: VecF32) f32 {
    return @reduce(.Max, v);
}

/// Element-wise exp approximation via @exp on each lane.
inline fn vec_exp(v: VecF32) VecF32 {
    var result: VecF32 = undefined;
    for (0..VEC_WIDTH) |i| {
        result[i] = @exp(v[i]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// SiLU: x * sigmoid(x)
// ---------------------------------------------------------------------------

/// Compute SiLU activation in-place: `out[i] = x[i] * sigmoid(x[i])`.
///
/// C ABI: `comfy_zig_silu(input, output, len)`.
/// `input` and `output` may alias (in-place operation is safe).
export fn comfy_zig_silu(
    input: [*]const f32,
    output: [*]f32,
    len: usize,
) callconv(.C) void {
    const ones: VecF32 = @splat(1.0);

    var offset: usize = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const x = load_vec(input, offset, len, 0.0);
        const neg_x: VecF32 = -x;
        const exp_neg = vec_exp(neg_x);
        const sigmoid = ones / (ones + exp_neg);
        const result = x * sigmoid;
        store_vec(output, offset, len, result);
    }

    // Scalar tail
    while (offset < len) : (offset += 1) {
        const x = input[offset];
        const sigmoid = 1.0 / (1.0 + @exp(-x));
        output[offset] = x * sigmoid;
    }
}

// ---------------------------------------------------------------------------
// Softmax (per-row, numerically stable)
// ---------------------------------------------------------------------------

/// Compute numerically stable softmax over a single row of `len` elements.
///
/// C ABI: `comfy_zig_softmax(input, output, len)`.
/// Steps: find max → subtract max & exp → sum → divide.
export fn comfy_zig_softmax(
    input: [*]const f32,
    output: [*]f32,
    len: usize,
) callconv(.C) void {
    if (len == 0) return;

    // 1. Find max (SIMD reduction)
    var max_vec: VecF32 = @splat(-math.inf(f32));
    var offset: usize = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const v = load_vec(input, offset, len, -math.inf(f32));
        max_vec = @max(max_vec, v);
    }
    var max_val = hmax(max_vec);
    // Scalar tail for max
    while (offset < len) : (offset += 1) {
        if (input[offset] > max_val) max_val = input[offset];
    }

    // 2. exp(x - max) and sum
    const max_splat: VecF32 = @splat(max_val);
    var sum_vec: VecF32 = @splat(0.0);
    offset = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const v = load_vec(input, offset, len, max_val); // pad with max so exp=1 doesn't affect sum incorrectly
        const shifted = v - max_splat;
        const e = vec_exp(shifted);
        store_vec(output, offset, len, e);
        sum_vec += e;
    }
    var sum_val = hsum(sum_vec);
    // Scalar tail
    while (offset < len) : (offset += 1) {
        const e = @exp(input[offset] - max_val);
        output[offset] = e;
        sum_val += e;
    }

    // 3. Normalize by sum
    const inv_sum: VecF32 = @splat(1.0 / sum_val);
    offset = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const v = load_vec(output, offset, len, 0.0);
        store_vec(output, offset, len, v * inv_sum);
    }
    const inv = 1.0 / sum_val;
    while (offset < len) : (offset += 1) {
        output[offset] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Layer Normalization (per-row)
// ---------------------------------------------------------------------------

/// Compute layer normalization over a single row of `len` elements.
///
/// C ABI: `comfy_zig_layer_norm(input, output, len, eps)`.
/// Computes: `(x - mean) / sqrt(var + eps)` per row.
export fn comfy_zig_layer_norm(
    input: [*]const f32,
    output: [*]f32,
    len: usize,
    eps: f32,
) callconv(.C) void {
    if (len == 0) return;

    const n_f32: f32 = @floatFromInt(len);

    // 1. Compute mean (SIMD)
    var sum_vec: VecF32 = @splat(0.0);
    var offset: usize = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        sum_vec += load_vec(input, offset, len, 0.0);
    }
    var sum_val = hsum(sum_vec);
    while (offset < len) : (offset += 1) {
        sum_val += input[offset];
    }
    const mean = sum_val / n_f32;

    // 2. Compute variance (SIMD)
    const mean_splat: VecF32 = @splat(mean);
    var var_vec: VecF32 = @splat(0.0);
    offset = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const v = load_vec(input, offset, len, mean); // pad with mean so diff=0
        const diff = v - mean_splat;
        var_vec += diff * diff;
    }
    var var_val = hsum(var_vec);
    while (offset < len) : (offset += 1) {
        const diff = input[offset] - mean;
        var_val += diff * diff;
    }
    const inv_std = 1.0 / @sqrt(var_val / n_f32 + eps);

    // 3. Normalize (SIMD)
    const inv_std_splat: VecF32 = @splat(inv_std);
    offset = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const v = load_vec(input, offset, len, mean);
        const normalized = (v - mean_splat) * inv_std_splat;
        store_vec(output, offset, len, normalized);
    }
    while (offset < len) : (offset += 1) {
        output[offset] = (input[offset] - mean) * inv_std;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "silu basic" {
    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var output: [4]f32 = undefined;
    comfy_zig_silu(&input, &output, 4);

    // SiLU(0) = 0
    try std.testing.expectApproxEqAbs(output[0], 0.0, 1e-6);
    // SiLU(1) ≈ 0.7311
    try std.testing.expectApproxEqAbs(output[1], 0.7311, 1e-3);
}

test "softmax sums to 1" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    comfy_zig_softmax(&input, &output, 4);

    var sum: f32 = 0.0;
    for (output) |v| {
        sum += v;
    }
    try std.testing.expectApproxEqAbs(sum, 1.0, 1e-5);
}

test "layer_norm mean approx zero" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    comfy_zig_layer_norm(&input, &output, 4, 1e-5);

    var sum: f32 = 0.0;
    for (output) |v| {
        sum += v;
    }
    const mean = sum / 4.0;
    try std.testing.expectApproxEqAbs(mean, 0.0, 1e-5);
}
