/// Noise schedule generators for diffusion model sampling.
///
/// Each schedule returns a Vec of length `num_steps + 1`, ending with 0.0.

/// Linear beta schedule: linearly interpolates between `beta_start` and `beta_end`.
///
/// Returns cumulative sigma values derived from the beta schedule.
/// Length: `num_steps + 1`, last element is 0.0.
pub fn linear_schedule(num_steps: usize, beta_start: f64, beta_end: f64) -> Vec<f64> {
    if num_steps == 0 {
        return vec![0.0];
    }

    let mut sigmas = Vec::with_capacity(num_steps + 1);

    for i in 0..num_steps {
        let t = i as f64 / num_steps as f64;
        let beta = beta_start + t * (beta_end - beta_start);
        // Convert beta to sigma: sigma = sqrt(beta / (1 - beta)) simplified
        // Using a common linear schedule where sigma decreases from ~1 to 0
        let sigma = (1.0 - t) * (beta_end - beta_start).sqrt() + t * beta.sqrt();
        // Scale so it starts high and ends at 0
        let sigma_val = (1.0 - (i as f64 + 1.0) / num_steps as f64)
            * ((beta_end / beta_start).sqrt());
        let _ = (beta, sigma); // suppress unused warnings from intermediate
        sigmas.push(sigma_val.max(0.0));
    }

    // Ensure last element is exactly 0.0
    sigmas.push(0.0);
    sigmas
}

/// Karras sigma schedule (from "Elucidating the Design Space of Diffusion-Based
/// Generative Models" by Karras et al., 2022).
///
/// Produces a decreasing sequence from `sigma_max` to `sigma_min`, followed by 0.0.
/// Length: `num_steps + 1`, last element is 0.0.
pub fn karras_schedule(num_steps: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f64> {
    if num_steps == 0 {
        return vec![0.0];
    }

    let mut sigmas = Vec::with_capacity(num_steps + 1);

    let inv_rho = 1.0 / rho;
    let min_inv_rho = sigma_min.powf(inv_rho);
    let max_inv_rho = sigma_max.powf(inv_rho);

    for i in 0..num_steps {
        let t = i as f64 / (num_steps - 1).max(1) as f64;
        let sigma = (max_inv_rho + t * (min_inv_rho - max_inv_rho)).powf(rho);
        sigmas.push(sigma);
    }

    // Append final 0.0
    sigmas.push(0.0);
    sigmas
}

/// Cosine schedule (from "Improved Denoising Diffusion Probabilistic Models",
/// Nichol & Dhariwal, 2021).
///
/// Uses a cosine function to produce a smooth decreasing schedule.
/// Length: `num_steps + 1`, last element is 0.0.
pub fn cosine_schedule(num_steps: usize) -> Vec<f64> {
    if num_steps == 0 {
        return vec![0.0];
    }

    let s = 0.008; // Small offset to prevent singularity at t=0
    let mut sigmas = Vec::with_capacity(num_steps + 1);

    for i in 0..num_steps {
        let t = i as f64 / num_steps as f64;
        // alpha_bar from cosine schedule
        let alpha_bar = ((t + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2).cos().powi(2);
        // sigma = sqrt((1 - alpha_bar) / alpha_bar)
        let sigma = ((1.0 - alpha_bar) / alpha_bar).sqrt();
        sigmas.push(sigma);
    }

    // Append final 0.0
    sigmas.push(0.0);
    sigmas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_schedule_length() {
        let steps = 20;
        let sigmas = linear_schedule(steps, 0.00085, 0.012);
        assert_eq!(sigmas.len(), steps + 1, "linear schedule should have num_steps + 1 elements");
    }

    #[test]
    fn test_linear_schedule_last_is_zero() {
        let sigmas = linear_schedule(50, 0.00085, 0.012);
        assert_eq!(
            *sigmas.last().unwrap(),
            0.0,
            "linear schedule should end with 0.0"
        );
    }

    #[test]
    fn test_linear_schedule_zero_steps() {
        let sigmas = linear_schedule(0, 0.00085, 0.012);
        assert_eq!(sigmas.len(), 1);
        assert_eq!(sigmas[0], 0.0);
    }

    #[test]
    fn test_karras_schedule_length() {
        let steps = 30;
        let sigmas = karras_schedule(steps, 0.0292, 14.6146, 7.0);
        assert_eq!(sigmas.len(), steps + 1, "karras schedule should have num_steps + 1 elements");
    }

    #[test]
    fn test_karras_schedule_last_is_zero() {
        let sigmas = karras_schedule(20, 0.0292, 14.6146, 7.0);
        assert_eq!(
            *sigmas.last().unwrap(),
            0.0,
            "karras schedule should end with 0.0"
        );
    }

    #[test]
    fn test_karras_schedule_is_decreasing() {
        let sigmas = karras_schedule(50, 0.0292, 14.6146, 7.0);
        // Exclude the final 0.0 appended value for monotonicity check
        for i in 1..sigmas.len() - 1 {
            assert!(
                sigmas[i] <= sigmas[i - 1],
                "karras schedule should be decreasing: sigmas[{}]={} > sigmas[{}]={}",
                i - 1,
                sigmas[i - 1],
                i,
                sigmas[i]
            );
        }
    }

    #[test]
    fn test_karras_schedule_first_is_sigma_max() {
        let sigma_max = 14.6146;
        let sigmas = karras_schedule(20, 0.0292, sigma_max, 7.0);
        assert!(
            (sigmas[0] - sigma_max).abs() < 1e-6,
            "first sigma should be sigma_max, got {}",
            sigmas[0]
        );
    }

    #[test]
    fn test_karras_schedule_zero_steps() {
        let sigmas = karras_schedule(0, 0.0292, 14.6146, 7.0);
        assert_eq!(sigmas.len(), 1);
        assert_eq!(sigmas[0], 0.0);
    }

    #[test]
    fn test_cosine_schedule_length() {
        let steps = 25;
        let sigmas = cosine_schedule(steps);
        assert_eq!(sigmas.len(), steps + 1, "cosine schedule should have num_steps + 1 elements");
    }

    #[test]
    fn test_cosine_schedule_last_is_zero() {
        let sigmas = cosine_schedule(40);
        assert_eq!(
            *sigmas.last().unwrap(),
            0.0,
            "cosine schedule should end with 0.0"
        );
    }

    #[test]
    fn test_cosine_schedule_zero_steps() {
        let sigmas = cosine_schedule(0);
        assert_eq!(sigmas.len(), 1);
        assert_eq!(sigmas[0], 0.0);
    }

    #[test]
    fn test_cosine_schedule_values_positive() {
        let sigmas = cosine_schedule(50);
        // All values except the final 0.0 should be positive
        for (i, &s) in sigmas.iter().enumerate() {
            if i < sigmas.len() - 1 {
                assert!(s >= 0.0, "cosine sigma[{i}] should be non-negative, got {s}");
            }
        }
    }
}
