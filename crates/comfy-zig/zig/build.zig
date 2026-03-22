const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Image I/O library ---
    const image_lib = b.addStaticLibrary(.{
        .name = "comfy_zig_image",
        .root_source_file = b.path("src/image_io.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(image_lib);

    // --- SIMD ops library ---
    const simd_lib = b.addStaticLibrary(.{
        .name = "comfy_zig_simd",
        .root_source_file = b.path("src/simd_ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(simd_lib);

    // --- Combined library (links both object files) ---
    const combined_lib = b.addStaticLibrary(.{
        .name = "comfy_zig",
        .root_source_file = b.path("src/image_io.zig"),
        .target = target,
        .optimize = optimize,
    });
    combined_lib.addObjectFile(simd_lib.getEmittedBin());
    b.installArtifact(combined_lib);

    // --- Tests ---
    const test_step = b.step("test", "Run unit tests");

    const image_tests = b.addTest(.{
        .root_source_file = b.path("src/image_io.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_step.dependOn(&b.addRunArtifact(image_tests).step);

    const simd_tests = b.addTest(.{
        .root_source_file = b.path("src/simd_ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_step.dependOn(&b.addRunArtifact(simd_tests).step);
}
