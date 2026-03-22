#
# ComfyUI-Turbo Julia Schedulers
#
# Noise schedule generators for diffusion model sampling.
# Each schedule returns a Vector of length num_steps + 1, ending with 0.0.
#

"""
    linear_schedule(num_steps, beta_start, beta_end) -> Vector{Float64}

Linear beta schedule: linearly interpolates between beta_start and beta_end.
Returns a decreasing sigma schedule of length num_steps + 1, ending with 0.0.
"""
function linear_schedule(num_steps::Int, beta_start::Float64, beta_end::Float64)::Vector{Float64}
    if num_steps == 0
        return [0.0]
    end

    sigmas = Vector{Float64}(undef, num_steps + 1)

    for i in 0:(num_steps - 1)
        t = Float64(i) / Float64(num_steps)
        sigma_val = (1.0 - (Float64(i) + 1.0) / Float64(num_steps)) * sqrt(beta_end / beta_start)
        sigmas[i + 1] = max(0.0, sigma_val)
    end

    sigmas[end] = 0.0
    return sigmas
end

"""
    karras_schedule(num_steps, sigma_min, sigma_max, rho) -> Vector{Float64}

Karras sigma schedule from "Elucidating the Design Space of Diffusion-Based
Generative Models" (Karras et al., 2022).

Returns a decreasing sequence from sigma_max to sigma_min, followed by 0.0.
Length: num_steps + 1.
"""
function karras_schedule(num_steps::Int, sigma_min::Float64,
                         sigma_max::Float64, rho::Float64)::Vector{Float64}
    if num_steps == 0
        return [0.0]
    end

    sigmas = Vector{Float64}(undef, num_steps + 1)

    inv_rho = 1.0 / rho
    min_inv_rho = sigma_min^inv_rho
    max_inv_rho = sigma_max^inv_rho

    denom = max(1, num_steps - 1)
    for i in 0:(num_steps - 1)
        t = Float64(i) / Float64(denom)
        sigmas[i + 1] = (max_inv_rho + t * (min_inv_rho - max_inv_rho))^rho
    end

    sigmas[end] = 0.0
    return sigmas
end

"""
    cosine_schedule(num_steps) -> Vector{Float64}

Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
(Nichol & Dhariwal, 2021).

Returns a smooth decreasing schedule of length num_steps + 1, ending with 0.0.
"""
function cosine_schedule(num_steps::Int)::Vector{Float64}
    if num_steps == 0
        return [0.0]
    end

    s = 0.008  # Small offset to prevent singularity at t=0
    sigmas = Vector{Float64}(undef, num_steps + 1)

    for i in 0:(num_steps - 1)
        t = Float64(i) / Float64(num_steps)
        alpha_bar = cos((t + s) / (1.0 + s) * pi / 2.0)^2
        sigma = sqrt((1.0 - alpha_bar) / alpha_bar)
        sigmas[i + 1] = sigma
    end

    sigmas[end] = 0.0
    return sigmas
end
