// comfy-zig: Zig native image I/O stubs
//
// These are placeholder exports that will be implemented with
// SIMD-optimized image decoding/encoding in a future phase.
// They match the extern "C" declarations in src/ffi.rs.

const std = @import("std");

export fn comfy_zig_decode_png(
    _path_ptr: [*]const u8,
    _path_len: usize,
    _out_ptr: *?[*]u8,
    _out_len: *usize,
    _out_width: *u32,
    _out_height: *u32,
) callconv(.C) i32 {
    // TODO: implement PNG decoding with SIMD acceleration
    return -1; // not implemented
}

export fn comfy_zig_encode_png(
    _data_ptr: [*]const u8,
    _data_len: usize,
    _width: u32,
    _height: u32,
    _path_ptr: [*]const u8,
    _path_len: usize,
) callconv(.C) i32 {
    // TODO: implement PNG encoding with SIMD acceleration
    return -1; // not implemented
}

export fn comfy_zig_free(_ptr: ?[*]u8, _len: usize) callconv(.C) void {
    // TODO: free Zig-allocated memory
}

test "stubs compile" {
    // Verify the C ABI exports compile successfully
    try std.testing.expect(comfy_zig_decode_png(undefined, 0, undefined, undefined, undefined, undefined) == -1);
}
