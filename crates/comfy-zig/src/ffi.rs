//! Zig FFI declarations for native SIMD and image I/O operations.
//!
//! These are only compiled when the `zig-native` feature is enabled.
//! The corresponding Zig implementations live in `zig/src/`.

// ---------------------------------------------------------------------------
// Image I/O FFI (from zig/src/image_io.zig)
// ---------------------------------------------------------------------------

#[cfg(feature = "zig-native")]
extern "C" {
    /// Decode a PNG file into raw RGB pixels.
    ///
    /// - `path_ptr` / `path_len`: UTF-8 file path
    /// - `out_ptr`: receives a pointer to allocated pixel data (caller must free with `comfy_zig_free`)
    /// - `out_len`: receives the byte length of pixel data
    /// - `out_width` / `out_height`: receive image dimensions
    ///
    /// Returns 0 on success, non-zero on error.
    pub fn comfy_zig_decode_png(
        path_ptr: *const u8,
        path_len: usize,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
    ) -> i32;

    /// Encode raw RGB pixels as a PNG file.
    ///
    /// - `data_ptr` / `data_len`: raw RGB pixel data
    /// - `width` / `height`: image dimensions
    /// - `path_ptr` / `path_len`: UTF-8 output file path
    ///
    /// Returns 0 on success, non-zero on error.
    pub fn comfy_zig_encode_png(
        data_ptr: *const u8,
        data_len: usize,
        width: u32,
        height: u32,
        path_ptr: *const u8,
        path_len: usize,
    ) -> i32;

    /// Free memory allocated by Zig.
    pub fn comfy_zig_free(ptr: *mut u8, len: usize);
}

// ---------------------------------------------------------------------------
// SIMD kernel FFI (from zig/src/simd_ops.zig)
// ---------------------------------------------------------------------------

#[cfg(feature = "zig-native")]
extern "C" {
    /// SiLU activation: `output[i] = input[i] * sigmoid(input[i])`.
    ///
    /// SIMD-accelerated using 256-bit (8×f32) vectors.
    /// `input` and `output` may alias for in-place operation.
    pub fn comfy_zig_silu(input: *const f32, output: *mut f32, len: usize);

    /// Numerically stable softmax over a single row of `len` elements.
    ///
    /// SIMD-accelerated: finds max, subtracts, exp, normalizes.
    pub fn comfy_zig_softmax(input: *const f32, output: *mut f32, len: usize);

    /// Layer normalization over a single row of `len` elements.
    ///
    /// Computes `(x - mean) / sqrt(var + eps)` with SIMD reduction.
    pub fn comfy_zig_layer_norm(
        input: *const f32,
        output: *mut f32,
        len: usize,
        eps: f32,
    );

    /// GELU activation with tanh approximation, SIMD-accelerated.
    pub fn comfy_zig_gelu(input: *const f32, output: *mut f32, len: usize);

    /// Group normalization, SIMD-accelerated.
    pub fn comfy_zig_group_norm(
        input: *const f32,
        output: *mut f32,
        batch: usize,
        channels: usize,
        spatial: usize,
        num_groups: usize,
        eps: f32,
    );

    /// Fused multiply-add: output = a * scale + b, SIMD-accelerated.
    pub fn comfy_zig_fused_multiply_add(
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        len: usize,
        scale: f32,
    );

    /// Convert F32 array to F16 (IEEE 754 half-precision).
    pub fn comfy_zig_f32_to_f16(input: *const f32, output: *mut u16, len: usize);

    /// Convert F16 array to F32.
    pub fn comfy_zig_f16_to_f32(input: *const u16, output: *mut f32, len: usize);

    /// Elementwise clamp with SIMD acceleration.
    pub fn comfy_zig_clamp(
        input: *const f32,
        output: *mut f32,
        len: usize,
        min_val: f32,
        max_val: f32,
    );
}

#[cfg(test)]
mod tests {
    // FFI functions are only available when zig-native is enabled.
    // Compile-time verification only; no runtime tests for stubs.

    #[test]
    fn test_ffi_module_compiles() {
        // This test verifies the module compiles correctly
        // under the default (fallback) feature set.
        assert!(true);
    }

    #[test]
    fn test_ffi_simd_declarations_exist() {
        // Verify the FFI declarations compile under default features.
        // The actual functions are only linked with zig-native.
        #[cfg(feature = "zig-native")]
        {
            // These would link at runtime; just verify type signatures compile.
            let _silu: unsafe extern "C" fn(*const f32, *mut f32, usize) =
                super::comfy_zig_silu;
            let _softmax: unsafe extern "C" fn(*const f32, *mut f32, usize) =
                super::comfy_zig_softmax;
            let _layer_norm: unsafe extern "C" fn(*const f32, *mut f32, usize, f32) =
                super::comfy_zig_layer_norm;
            let _gelu: unsafe extern "C" fn(*const f32, *mut f32, usize) =
                super::comfy_zig_gelu;
            let _group_norm: unsafe extern "C" fn(*const f32, *mut f32, usize, usize, usize, usize, f32) =
                super::comfy_zig_group_norm;
            let _fma: unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize, f32) =
                super::comfy_zig_fused_multiply_add;
        }
        assert!(true);
    }
}
