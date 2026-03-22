pub mod ffi;
pub mod fusion;
pub mod image_io;
pub mod kernels;
pub mod simd;

pub use fusion::{fused_gemm_gelu, fused_gemm_silu, fused_layer_norm_silu, fused_softmax_scale};
pub use image_io::{decode_image, encode_png, normalize_u8_to_f32};
pub use kernels::{gelu, gemm, group_norm, layer_norm, silu, softmax};
pub use simd::{fused_multiply_add, generate_noise_f32};
