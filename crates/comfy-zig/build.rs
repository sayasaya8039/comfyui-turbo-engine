//! Build script for comfy-zig.
//!
//! When the `zig-native` feature is enabled, this script compiles the Zig
//! SIMD sources (image_io.zig and simd_ops.zig) into a static library and
//! links it. Under the default `fallback` feature, this is a no-op.

fn main() {
    #[cfg(feature = "zig-native")]
    build_zig();
}

#[cfg(feature = "zig-native")]
fn build_zig() {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    let zig_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("zig");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Determine Zig target triple for MSVC on Windows
    let zig_target = if cfg!(target_os = "windows") && cfg!(target_env = "msvc") {
        "x86_64-windows-msvc"
    } else if cfg!(target_os = "windows") {
        "x86_64-windows-gnu"
    } else if cfg!(target_os = "linux") {
        "x86_64-linux-gnu"
    } else if cfg!(target_os = "macos") {
        "aarch64-macos-none"
    } else {
        "native"
    };

    let prefix = out_dir.join("zig-out").join("lib");

    let status = Command::new("zig")
        .arg("build")
        .arg(&format!("-Dtarget={}", zig_target))
        .arg("-Doptimize=ReleaseFast")
        .arg(&format!("--prefix={}", out_dir.join("zig-out").display()))
        .current_dir(&zig_dir)
        .status()
        .expect("Failed to run `zig build`. Is Zig installed?");

    if !status.success() {
        panic!("Zig build failed with status: {}", status);
    }

    // Link the combined static library
    println!("cargo:rustc-link-search=native={}", prefix.display());
    println!("cargo:rustc-link-lib=static=comfy_zig");

    // Re-run if Zig sources change
    println!("cargo:rerun-if-changed=zig/src/image_io.zig");
    println!("cargo:rerun-if-changed=zig/src/simd_ops.zig");
    println!("cargo:rerun-if-changed=zig/build.zig");
}
