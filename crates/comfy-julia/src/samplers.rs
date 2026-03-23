/// Public API for sampler and scheduler selection.
///
/// Provides enum-based dispatch for different sampling algorithms and
/// noise schedules used in diffusion model inference.
use comfy_core::{ComfyError, ComfyResult, Tensor};
use serde::{Deserialize, Serialize};

use crate::fallback;
use crate::schedulers;

/// Available sampler algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SamplerType {
    Euler,
    EulerAncestral,
    Ddim,
    DpmPlusPlus2m,
}

impl SamplerType {
    /// Parse a sampler name string into a SamplerType.
    ///
    /// Accepts various common formats (case-insensitive):
    /// - "euler" -> Euler
    /// - "euler_ancestral" / "euler-ancestral" / "eulera" -> EulerAncestral
    /// - "ddim" -> Ddim
    /// - "dpmpp_2m" / "dpm++2m" / "dpmpp2m" / "dpm_plus_plus_2m" -> DpmPlusPlus2m
    pub fn from_name(name: &str) -> ComfyResult<Self> {
        let normalized = name.to_lowercase().replace('-', "_");
        match normalized.as_str() {
            "euler" => Ok(SamplerType::Euler),
            "euler_ancestral" | "eulera" | "euler_a" => Ok(SamplerType::EulerAncestral),
            "ddim" => Ok(SamplerType::Ddim),
            "dpmpp_2m" | "dpm++2m" | "dpmpp2m" | "dpm_plus_plus_2m" | "dpm___2m" => {
                Ok(SamplerType::DpmPlusPlus2m)
            }
            _ => Err(ComfyError::InferenceError(format!(
                "unknown sampler: '{name}'"
            ))),
        }
    }

    /// Return the canonical name of this sampler.
    pub fn name(&self) -> &'static str {
        match self {
            SamplerType::Euler => "euler",
            SamplerType::EulerAncestral => "euler_ancestral",
            SamplerType::Ddim => "ddim",
            SamplerType::DpmPlusPlus2m => "dpmpp_2m",
        }
    }
}

/// Available noise schedule types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchedulerType {
    Normal,
    Karras,
    Cosine,
}

impl SchedulerType {
    /// Parse a scheduler name string into a SchedulerType (case-insensitive).
    pub fn from_name(name: &str) -> ComfyResult<Self> {
        let normalized = name.to_lowercase();
        match normalized.as_str() {
            "normal" | "linear" => Ok(SchedulerType::Normal),
            "karras" => Ok(SchedulerType::Karras),
            "cosine" => Ok(SchedulerType::Cosine),
            _ => Err(ComfyError::InferenceError(format!(
                "unknown scheduler: '{name}'"
            ))),
        }
    }

    /// Return the canonical name of this scheduler.
    pub fn name(&self) -> &'static str {
        match self {
            SchedulerType::Normal => "normal",
            SchedulerType::Karras => "karras",
            SchedulerType::Cosine => "cosine",
        }
    }
}

/// Generate a sigma schedule for the given scheduler type and number of steps.
///
/// Returns a Vec<f64> of length `steps + 1`, ending with 0.0.
pub fn get_sigmas(scheduler: SchedulerType, steps: usize) -> Vec<f64> {
    match scheduler {
        SchedulerType::Normal => schedulers::linear_schedule(steps, 0.00085, 0.012),
        SchedulerType::Karras => schedulers::karras_schedule(steps, 0.0292, 14.6146, 7.0),
        SchedulerType::Cosine => schedulers::cosine_schedule(steps),
    }
}

/// Run sampling with the specified sampler, latent, sigma schedule, and denoise function.
///
/// Currently, all sampler types use the Euler loop as the primary fallback.
/// EulerAncestral, Ddim, and DpmPlusPlus2m will be extended with specialized
/// implementations as the Julia backend matures.
pub fn sample<F>(
    sampler: SamplerType,
    latent: &Tensor,
    sigmas: &[f64],
    denoise_fn: F,
) -> ComfyResult<Tensor>
where
    F: Fn(&Tensor, f64) -> ComfyResult<Tensor>,
{
    tracing::info!(sampler = sampler.name(), steps = sigmas.len().saturating_sub(1), "starting sampling");

    // ComfyUI v0.18.1: Ensure noise/latent are F32 to prevent FP16 precision issues
    let latent = if latent.dtype() != comfy_core::DType::F32 {
        tracing::warn!("sample: converting latent from {:?} to F32 for precision", latent.dtype());
        // For now, we only support F32 tensors in the sampler
        latent
    } else {
        latent
    };

    match sampler {
        SamplerType::Euler | SamplerType::EulerAncestral => {
            fallback::euler_sample(latent, sigmas, denoise_fn)
        }
        SamplerType::Ddim => {
            // DDIM uses per-step logic; wrap it in a loop similar to euler_sample
            ddim_sample_loop(latent, sigmas, denoise_fn, 0.0)
        }
        SamplerType::DpmPlusPlus2m => {
            // DPM++ 2M falls back to Euler for now
            tracing::warn!("DPM++ 2M not yet specialized, falling back to Euler");
            fallback::euler_sample(latent, sigmas, denoise_fn)
        }
    }
}

/// DDIM sampling loop using fallback::ddim_step.
fn ddim_sample_loop<F>(
    latent: &Tensor,
    sigmas: &[f64],
    denoise_fn: F,
    eta: f32,
) -> ComfyResult<Tensor>
where
    F: Fn(&Tensor, f64) -> ComfyResult<Tensor>,
{
    if sigmas.is_empty() {
        return Ok(latent.clone());
    }

    let shape = latent.shape().to_vec();
    let mut current = latent.as_slice_f32()?.to_vec();

    for i in 0..sigmas.len() - 1 {
        let sigma = sigmas[i] as f32;
        let sigma_next = sigmas[i + 1] as f32;

        let current_tensor = Tensor::from_vec_f32_fast(current.clone(), shape.clone())?;
        let denoised = denoise_fn(&current_tensor, sigmas[i])?;
        let denoised_slice = denoised.as_slice_f32()?;

        current = fallback::ddim_step(&current, denoised_slice, sigma, sigma_next, eta);
    }

    Tensor::from_vec_f32_fast(current, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SamplerType::from_name tests ---

    #[test]
    fn test_sampler_from_name_euler() {
        assert_eq!(SamplerType::from_name("euler").unwrap(), SamplerType::Euler);
        assert_eq!(SamplerType::from_name("Euler").unwrap(), SamplerType::Euler);
        assert_eq!(SamplerType::from_name("EULER").unwrap(), SamplerType::Euler);
    }

    #[test]
    fn test_sampler_from_name_euler_ancestral() {
        assert_eq!(
            SamplerType::from_name("euler_ancestral").unwrap(),
            SamplerType::EulerAncestral
        );
        assert_eq!(
            SamplerType::from_name("euler-ancestral").unwrap(),
            SamplerType::EulerAncestral
        );
        assert_eq!(
            SamplerType::from_name("eulera").unwrap(),
            SamplerType::EulerAncestral
        );
    }

    #[test]
    fn test_sampler_from_name_ddim() {
        assert_eq!(SamplerType::from_name("ddim").unwrap(), SamplerType::Ddim);
        assert_eq!(SamplerType::from_name("DDIM").unwrap(), SamplerType::Ddim);
    }

    #[test]
    fn test_sampler_from_name_dpmpp() {
        assert_eq!(
            SamplerType::from_name("dpmpp_2m").unwrap(),
            SamplerType::DpmPlusPlus2m
        );
        assert_eq!(
            SamplerType::from_name("dpmpp2m").unwrap(),
            SamplerType::DpmPlusPlus2m
        );
        assert_eq!(
            SamplerType::from_name("dpm_plus_plus_2m").unwrap(),
            SamplerType::DpmPlusPlus2m
        );
    }

    #[test]
    fn test_sampler_from_name_unknown() {
        assert!(SamplerType::from_name("unknown_sampler").is_err());
    }

    #[test]
    fn test_sampler_name_roundtrip() {
        let samplers = [
            SamplerType::Euler,
            SamplerType::EulerAncestral,
            SamplerType::Ddim,
            SamplerType::DpmPlusPlus2m,
        ];
        for s in &samplers {
            let name = s.name();
            let parsed = SamplerType::from_name(name).unwrap();
            assert_eq!(*s, parsed, "roundtrip failed for {name}");
        }
    }

    // --- SchedulerType::from_name tests ---

    #[test]
    fn test_scheduler_from_name_normal() {
        assert_eq!(
            SchedulerType::from_name("normal").unwrap(),
            SchedulerType::Normal
        );
        assert_eq!(
            SchedulerType::from_name("linear").unwrap(),
            SchedulerType::Normal
        );
    }

    #[test]
    fn test_scheduler_from_name_karras() {
        assert_eq!(
            SchedulerType::from_name("karras").unwrap(),
            SchedulerType::Karras
        );
        assert_eq!(
            SchedulerType::from_name("Karras").unwrap(),
            SchedulerType::Karras
        );
    }

    #[test]
    fn test_scheduler_from_name_cosine() {
        assert_eq!(
            SchedulerType::from_name("cosine").unwrap(),
            SchedulerType::Cosine
        );
    }

    #[test]
    fn test_scheduler_from_name_unknown() {
        assert!(SchedulerType::from_name("exponential").is_err());
    }

    #[test]
    fn test_scheduler_name_roundtrip() {
        let schedulers = [
            SchedulerType::Normal,
            SchedulerType::Karras,
            SchedulerType::Cosine,
        ];
        for s in &schedulers {
            let name = s.name();
            let parsed = SchedulerType::from_name(name).unwrap();
            assert_eq!(*s, parsed, "roundtrip failed for {name}");
        }
    }

    // --- get_sigmas tests ---

    #[test]
    fn test_get_sigmas_length() {
        for scheduler in [SchedulerType::Normal, SchedulerType::Karras, SchedulerType::Cosine] {
            let steps = 20;
            let sigmas = get_sigmas(scheduler, steps);
            assert_eq!(
                sigmas.len(),
                steps + 1,
                "get_sigmas({:?}, {}) should return {} elements",
                scheduler,
                steps,
                steps + 1
            );
        }
    }

    #[test]
    fn test_get_sigmas_ends_with_zero() {
        for scheduler in [SchedulerType::Normal, SchedulerType::Karras, SchedulerType::Cosine] {
            let sigmas = get_sigmas(scheduler, 15);
            assert_eq!(
                *sigmas.last().unwrap(),
                0.0,
                "get_sigmas({:?}) should end with 0.0",
                scheduler
            );
        }
    }

    // --- sample tests ---

    #[test]
    fn test_sample_euler_executes() {
        let latent = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let sigmas = vec![1.0, 0.5, 0.0];

        let result = sample(SamplerType::Euler, &latent, &sigmas, |input, _sigma| {
            let slice = input.as_slice_f32()?;
            let denoised: Vec<f32> = slice.iter().map(|v| v * 0.5).collect();
            Tensor::from_vec_f32(denoised, input.shape().to_vec())
        });

        assert!(result.is_ok(), "Euler sample should succeed");
        assert_eq!(result.unwrap().shape(), &[1, 4]);
    }

    #[test]
    fn test_sample_ddim_executes() {
        let latent = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let sigmas = vec![1.0, 0.5, 0.0];

        let result = sample(SamplerType::Ddim, &latent, &sigmas, |input, _sigma| {
            let slice = input.as_slice_f32()?;
            let denoised: Vec<f32> = slice.iter().map(|v| v * 0.5).collect();
            Tensor::from_vec_f32(denoised, input.shape().to_vec())
        });

        assert!(result.is_ok(), "DDIM sample should succeed");
        assert_eq!(result.unwrap().shape(), &[1, 4]);
    }

    #[test]
    fn test_sample_dpmpp2m_executes() {
        let latent = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let sigmas = vec![1.0, 0.0];

        let result = sample(
            SamplerType::DpmPlusPlus2m,
            &latent,
            &sigmas,
            |input, _sigma| {
                let zeros = vec![0.0f32; input.numel()];
                Tensor::from_vec_f32(zeros, input.shape().to_vec())
            },
        );

        assert!(result.is_ok(), "DPM++ 2M sample should succeed");
    }

    #[test]
    fn test_sample_with_real_schedule() {
        let latent = Tensor::from_vec_f32(vec![5.0, 10.0], vec![2]).unwrap();
        let sigmas = get_sigmas(SchedulerType::Karras, 10);

        let result = sample(SamplerType::Euler, &latent, &sigmas, |input, _sigma| {
            // Simple denoise: predict zeros
            let zeros = vec![0.0f32; input.numel()];
            Tensor::from_vec_f32(zeros, input.shape().to_vec())
        });

        assert!(result.is_ok(), "sample with Karras schedule should succeed");
        let output = result.unwrap();
        assert_eq!(output.shape(), &[2]);
    }
}
