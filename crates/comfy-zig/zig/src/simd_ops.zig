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
// GELU (Gaussian Error Linear Unit) — tanh approximation
// ---------------------------------------------------------------------------
export fn comfy_zig_gelu(
    input: [*]const f32,
    output: [*]f32,
    len: usize,
) callconv(.C) void {
    const sqrt_2_over_pi: f32 = 0.7978845608; // sqrt(2/pi)
    const coeff: f32 = 0.044715;
    const half: VecF32 = @splat(0.5);
    const ones: VecF32 = @splat(1.0);
    const sqrt_vec: VecF32 = @splat(sqrt_2_over_pi);
    const coeff_vec: VecF32 = @splat(coeff);

    var offset: usize = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const x = load_vec(input, offset, len, 0.0);
        const x3 = x * x * x;
        const inner = sqrt_vec * (x + coeff_vec * x3);
        // tanh approximation via (1 - 2/(exp(2x)+1))
        const exp_2inner = vec_exp(inner + inner);
        const tanh_val = ones - (ones + ones) / (exp_2inner + ones);
        const result = half * x * (ones + tanh_val);
        store_vec(output, offset, len, result);
    }
    // Scalar tail
    while (offset < len) : (offset += 1) {
        const x = input[offset];
        const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        const exp_2 = @exp(2.0 * inner);
        const tanh_val = (exp_2 - 1.0) / (exp_2 + 1.0);
        output[offset] = 0.5 * x * (1.0 + tanh_val);
    }
}

// ---------------------------------------------------------------------------
// Group Normalization [batch, channels, spatial...]
// ---------------------------------------------------------------------------
export fn comfy_zig_group_norm(
    input: [*]const f32,
    output: [*]f32,
    batch: usize,
    channels: usize,
    spatial: usize,
    num_groups: usize,
    eps: f32,
) callconv(.C) void {
    if (channels == 0 or num_groups == 0 or spatial == 0) return;
    const channels_per_group = channels / num_groups;
    const group_size = channels_per_group * spatial;

    for (0..batch) |b| {
        for (0..num_groups) |g| {
            const ch_start = g * channels_per_group;
            // Compute mean (SIMD accumulate)
            var sum_vec: VecF32 = @splat(0.0);
            var scalar_sum: f32 = 0.0;
            for (ch_start..ch_start + channels_per_group) |c| {
                var s: usize = 0;
                while (s + VEC_WIDTH <= spatial) : (s += VEC_WIDTH) {
                    const idx = b * (channels * spatial) + c * spatial + s;
                    sum_vec += load_vec(input, idx, b * channels * spatial + channels * spatial, 0.0);
                }
                while (s < spatial) : (s += 1) {
                    const idx = b * (channels * spatial) + c * spatial + s;
                    scalar_sum += input[idx];
                }
            }
            const mean = (hsum(sum_vec) + scalar_sum) / @as(f32, @floatFromInt(group_size));

            // Compute variance (SIMD)
            const mean_splat: VecF32 = @splat(mean);
            var var_vec: VecF32 = @splat(0.0);
            var scalar_var: f32 = 0.0;
            for (ch_start..ch_start + channels_per_group) |c| {
                var s: usize = 0;
                while (s + VEC_WIDTH <= spatial) : (s += VEC_WIDTH) {
                    const idx = b * (channels * spatial) + c * spatial + s;
                    const v = load_vec(input, idx, b * channels * spatial + channels * spatial, mean);
                    const diff = v - mean_splat;
                    var_vec += diff * diff;
                }
                while (s < spatial) : (s += 1) {
                    const idx = b * (channels * spatial) + c * spatial + s;
                    const diff = input[idx] - mean;
                    scalar_var += diff * diff;
                }
            }
            const variance = (hsum(var_vec) + scalar_var) / @as(f32, @floatFromInt(group_size));
            const inv_std = 1.0 / @sqrt(variance + eps);
            const inv_std_splat: VecF32 = @splat(inv_std);

            // Normalize (SIMD)
            for (ch_start..ch_start + channels_per_group) |c| {
                var s: usize = 0;
                while (s + VEC_WIDTH <= spatial) : (s += VEC_WIDTH) {
                    const idx = b * (channels * spatial) + c * spatial + s;
                    const v = load_vec(input, idx, b * channels * spatial + channels * spatial, mean);
                    store_vec(output, idx, b * channels * spatial + channels * spatial, (v - mean_splat) * inv_std_splat);
                }
                while (s < spatial) : (s += 1) {
                    const idx = b * (channels * spatial) + c * spatial + s;
                    output[idx] = (input[idx] - mean) * inv_std;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Multiply-Add: a * scale + b
// ---------------------------------------------------------------------------
export fn comfy_zig_fused_multiply_add(
    a: [*]const f32,
    b: [*]const f32,
    output: [*]f32,
    len: usize,
    scale: f32,
) callconv(.C) void {
    const scale_vec: VecF32 = @splat(scale);

    var offset: usize = 0;
    while (offset + VEC_WIDTH <= len) : (offset += VEC_WIDTH) {
        const va = load_vec(a, offset, len, 0.0);
        const vb = load_vec(b, offset, len, 0.0);
        store_vec(output, offset, len, va * scale_vec + vb);
    }
    while (offset < len) : (offset += 1) {
        output[offset] = a[offset] * scale + b[offset];
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

test "gelu basic" {
    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var output: [4]f32 = undefined;
    comfy_zig_gelu(&input, &output, 4);

    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(output[0], 0.0, 1e-6);
    // GELU(1) ≈ 0.8412
    try std.testing.expectApproxEqAbs(output[1], 0.8412, 1e-3);
    // GELU(-1) ≈ -0.1588
    try std.testing.expectApproxEqAbs(output[2], -0.1588, 1e-3);
}

test "fused_multiply_add basic" {
    var a_arr = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b_arr = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var output: [4]f32 = undefined;
    comfy_zig_fused_multiply_add(&a_arr, &b_arr, &output, 4, 2.0);

    // a * scale + b = [1*2+10, 2*2+20, 3*2+30, 4*2+40] = [12, 24, 36, 48]
    try std.testing.expectApproxEqAbs(output[0], 12.0, 1e-5);
    try std.testing.expectApproxEqAbs(output[1], 24.0, 1e-5);
    try std.testing.expectApproxEqAbs(output[2], 36.0, 1e-5);
    try std.testing.expectApproxEqAbs(output[3], 48.0, 1e-5);
}
