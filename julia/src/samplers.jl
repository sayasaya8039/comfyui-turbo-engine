#
# ComfyUI-Turbo Julia Samplers
#
# Sampler step implementations for diffusion model inference.
#

"""
    euler_step(x, denoised, sigma, sigma_next) -> Vector{Float32}

Perform a single Euler method step.
x_next = x + (x - denoised) / sigma * (sigma_next - sigma)
"""
function euler_step(x::Vector{Float32}, denoised::Vector{Float32},
                    sigma::Float32, sigma_next::Float32)::Vector{Float32}
    if abs(sigma) < eps(Float32)
        return copy(x)
    end

    dt = sigma_next - sigma
    d = (x .- denoised) ./ sigma
    return x .+ d .* dt
end

"""
    ddim_step(x, denoised, sigma, sigma_next, eta) -> Vector{Float32}

Perform a single DDIM (Denoising Diffusion Implicit Models) step.
When eta=0, this is deterministic DDIM. When eta=1, equivalent to DDPM.
"""
function ddim_step(x::Vector{Float32}, denoised::Vector{Float32},
                   sigma::Float32, sigma_next::Float32,
                   eta::Float32)::Vector{Float32}
    # Compute alpha values from sigma: alpha = 1 / (1 + sigma^2)
    alpha = 1.0f0 / (1.0f0 + sigma^2)
    alpha_next = 1.0f0 / (1.0f0 + sigma_next^2)

    # Noise coefficient (unused in deterministic mode)
    sigma_ddim = eta * sqrt(max(0.0f0,
        (1.0f0 - alpha_next) / (1.0f0 - alpha) * (1.0f0 - alpha / alpha_next)))

    sqrt_alpha_next = sqrt(alpha_next)
    sqrt_one_minus_alpha = sqrt(1.0f0 - alpha)
    sqrt_coeff = sqrt(max(0.0f0, (1.0f0 - alpha_next) - sigma_ddim^2))

    result = similar(x)
    for i in eachindex(x)
        pred_x0 = denoised[i]

        dir_xt = if abs(sqrt_one_minus_alpha) > eps(Float32)
            (x[i] - sqrt(alpha) * pred_x0) / sqrt_one_minus_alpha
        else
            0.0f0
        end

        result[i] = sqrt_alpha_next * pred_x0 + sqrt_coeff * dir_xt
    end

    return result
end
