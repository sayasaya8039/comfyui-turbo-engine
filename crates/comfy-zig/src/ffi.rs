//! Zig FFI declarations for native SIMD and image I/O operations.
//!
//! These are only compiled when the `zig-native` feature is enabled.
//! The corresponding Zig implementations live in `zig/src/`.

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
}
