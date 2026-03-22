/// Fallback sampler implementations in pure Rust.
///
/// These are used when Julia is not available, providing CPU-based
/// sampling for diffusion models.
use comfy_core::{ComfyResult, Tensor};

/// Perform a single Euler method step.
///
/// Given current latent `x`, model prediction `denoised`, and sigma values,
/// computes: `x_next = x + (x - denoised) / sigma * (sigma_next - sigma)`
pub fn euler_step(x: &[f32], denoised: &[f32], sigma: f32, sigma_next: f32) -> Vec<f32> {
    let len = x.len();
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        // d = (x - denoised) / sigma
        let d = if sigma.abs() > f32::EPSILON {
            (x[i] - denoised[i]) / sigma
        } else {
            0.0
        };
        // x_next = x + d * dt, where dt = sigma_next - sigma
        let dt = sigma_next - sigma;
        result.push(x[i] + d * dt);
    }

    result
}

/// Perform a single DDIM (Denoising Diffusion Implicit Models) step.
///
/// DDIM interpolates between DDPM and ODE-based sampling via the `eta` parameter.
/// When `eta = 0`, this is deterministic (DDIM). When `eta = 1`, it's equivalent to DDPM.
pub fn ddim_step(x: &[f32], denoised: &[f32], sigma: f32, sigma_next: f32, eta: f32) -> Vec<f32> {
    let len = x.len();
    let mut result = Vec::with_capacity(len);

    // Compute alpha values from sigma: alpha = 1 / (1 + sigma^2)
    let alpha = 1.0 / (1.0 + sigma * sigma);
    let alpha_next = 1.0 / (1.0 + sigma_next * sigma_next);

    // Noise coefficient
    let sigma_ddim = eta * ((1.0 - alpha_next) / (1.0 - alpha) * (1.0 - alpha / alpha_next)).sqrt();
    let sqrt_alpha_next = alpha_next.sqrt();
    let sqrt_one_minus_alpha = (1.0 - alpha).sqrt();
    let sqrt_one_minus_alpha_next_minus_sigma2 =
        ((1.0 - alpha_next) - sigma_ddim * sigma_ddim).max(0.0).sqrt();

    for i in 0..len {
        // Predicted x0 component
        let pred_x0 = denoised[i];

        // Direction pointing to x_t
        let dir_xt = if sqrt_one_minus_alpha.abs() > f32::EPSILON {
            (x[i] - alpha.sqrt() * pred_x0) / sqrt_one_minus_alpha
        } else {
            0.0
        };

        // DDIM update (deterministic part — no noise term since we don't have a random source here)
        let x_next =
            sqrt_alpha_next * pred_x0 + sqrt_one_minus_alpha_next_minus_sigma2 * dir_xt;

        result.push(x_next);
    }

    result
}

/// Full Euler sampling loop over a sigma schedule.
///
/// `latent`: initial noisy latent tensor (F32)
/// `sigmas`: noise schedule (length = num_steps + 1, ending with 0.0)
/// `denoise_fn`: model function that takes (noisy_input, sigma) -> denoised prediction
pub fn euler_sample<F>(latent: &Tensor, sigmas: &[f64], denoise_fn: F) -> ComfyResult<Tensor>
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

        // Get model prediction at current sigma
        let current_tensor = Tensor::from_vec_f32(current.clone(), shape.clone())?;
        let denoised = denoise_fn(&current_tensor, sigmas[i])?;
        let denoised_slice = denoised.as_slice_f32()?;

        // Euler step
        current = euler_step(&current, denoised_slice, sigma, sigma_next);
    }

    Tensor::from_vec_f32(current, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_step_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let denoised = vec![0.5, 1.0, 1.5, 2.0];
        let sigma = 1.0;
        let sigma_next = 0.5;

        let result = euler_step(&x, &denoised, sigma, sigma_next);
        assert_eq!(result.len(), x.len());

        // d = (x - denoised) / sigma = [0.5, 1.0, 1.5, 2.0]
        // dt = sigma_next - sigma = -0.5
        // result = x + d * dt = [1.0 - 0.25, 2.0 - 0.5, 3.0 - 0.75, 4.0 - 1.0]
        let expected = vec![0.75, 1.5, 2.25, 3.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-6,
                "euler_step mismatch: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_euler_step_zero_sigma() {
        let x = vec![1.0, 2.0];
        let denoised = vec![0.5, 1.0];

        // When sigma is 0, d should be 0, so result = x + 0 = x
        let result = euler_step(&x, &denoised, 0.0, 0.0);
        assert_eq!(result, x);
    }

    #[test]
    fn test_euler_step_to_zero() {
        // When sigma_next = 0 (final step), dt = -sigma
        let x = vec![2.0, 4.0];
        let denoised = vec![1.0, 2.0];
        let sigma = 1.0;

        let result = euler_step(&x, &denoised, sigma, 0.0);
        // d = [1.0, 2.0], dt = -1.0
        // result = [2.0 - 1.0, 4.0 - 2.0] = [1.0, 2.0] = denoised
        let expected = vec![1.0, 2.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-6,
                "euler_step to zero: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_ddim_step_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let denoised = vec![0.5, 1.0, 1.5, 2.0];
        let sigma = 1.0;
        let sigma_next = 0.5;
        let eta = 0.0; // Deterministic DDIM

        let result = ddim_step(&x, &denoised, sigma, sigma_next, eta);
        assert_eq!(result.len(), x.len());

        // Result should be closer to denoised than x is
        for i in 0..x.len() {
            let dist_from_denoised = (result[i] - denoised[i]).abs();
            let orig_dist = (x[i] - denoised[i]).abs();
            assert!(
                dist_from_denoised <= orig_dist + 1e-6,
                "DDIM step should move toward denoised: result[{i}]={}, denoised[{i}]={}, x[{i}]={}",
                result[i],
                denoised[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_ddim_step_zero_sigma_next() {
        let x = vec![1.0, 2.0];
        let denoised = vec![0.5, 1.5];

        // When sigma_next = 0, alpha_next = 1.0, so result ≈ denoised
        let result = ddim_step(&x, &denoised, 1.0, 0.0, 0.0);

        for (i, (&r, &d)) in result.iter().zip(denoised.iter()).enumerate() {
            assert!(
                (r - d).abs() < 1e-5,
                "DDIM final step should return denoised: result[{i}]={r}, denoised[{i}]={d}"
            );
        }
    }

    #[test]
    fn test_euler_sample_identity_denoise() {
        // Simple test: denoise_fn returns the input scaled down
        let latent = Tensor::from_vec_f32(vec![4.0, 8.0, 12.0, 16.0], vec![1, 4]).unwrap();
        let sigmas = vec![1.0, 0.5, 0.0];

        let result = euler_sample(&latent, &sigmas, |input, _sigma| {
            // "Denoised" prediction: just halve the input
            let slice = input.as_slice_f32()?;
            let denoised: Vec<f32> = slice.iter().map(|v| v * 0.5).collect();
            Tensor::from_vec_f32(denoised, input.shape().to_vec())
        })
        .unwrap();

        assert_eq!(result.shape(), &[1, 4]);
        let slice = result.as_slice_f32().unwrap();
        // After 2 steps of Euler with this denoise fn, values should decrease
        for &v in slice {
            assert!(v < 16.0, "values should decrease through sampling");
        }
    }

    #[test]
    fn test_euler_sample_convergence() {
        // With a denoise_fn that returns 0 (target), Euler should converge toward 0
        let latent = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2]).unwrap();
        let sigmas = vec![1.0, 0.75, 0.5, 0.25, 0.0];

        let result = euler_sample(&latent, &sigmas, |input, _sigma| {
            // Predict clean as zeros
            let zeros = vec![0.0f32; input.numel()];
            Tensor::from_vec_f32(zeros, input.shape().to_vec())
        })
        .unwrap();

        let slice = result.as_slice_f32().unwrap();
        // Should converge to 0
        for (i, &v) in slice.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "euler_sample should converge to 0: result[{i}]={v}"
            );
        }
    }

    #[test]
    fn test_euler_sample_empty_sigmas() {
        let latent = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]).unwrap();
        let result = euler_sample(&latent, &[], |_input, _sigma| {
            panic!("should not be called");
        })
        .unwrap();

        // Should return original latent unchanged
        let slice = result.as_slice_f32().unwrap();
        assert_eq!(slice, &[1.0, 2.0]);
    }
}
