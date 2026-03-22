#!/usr/bin/env julia
#
# ComfyUI-Turbo Julia Server
#
# JSON-over-stdio IPC server for high-performance sampling and scheduling.
# Protocol: one JSON object per line on stdin/stdout.
#
# Request format:  {"id": <int>, "method": <string>, "params": <object>}
# Response format: {"id": <int>, "result": <any>, "error": <string|null>}

using JSON

include("samplers.jl")
include("schedulers.jl")

"""
    handle_request(request::Dict) -> Dict

Dispatch a request to the appropriate handler and return a response.
"""
function handle_request(request::Dict)
    id = get(request, "id", 0)
    method = get(request, "method", "")
    params = get(request, "params", Dict())

    try
        result = if method == "schedule"
            handle_schedule(params)
        elseif method == "sample_step"
            handle_sample_step(params)
        elseif method == "quit"
            return Dict("id" => id, "result" => "bye", "error" => nothing)
        elseif method == "ping"
            Dict("status" => "ok", "version" => "0.1.0")
        else
            error("unknown method: $method")
        end

        return Dict("id" => id, "result" => result, "error" => nothing)
    catch e
        return Dict("id" => id, "result" => nothing, "error" => string(e))
    end
end

"""
    handle_schedule(params::Dict) -> Dict

Generate a noise schedule.
Params: scheduler (string), steps (int), optional sigma_min, sigma_max, rho
"""
function handle_schedule(params::Dict)
    scheduler = get(params, "scheduler", "karras")
    steps = get(params, "steps", 20)

    sigmas = if scheduler == "karras"
        sigma_min = get(params, "sigma_min", 0.0292)
        sigma_max = get(params, "sigma_max", 14.6146)
        rho = get(params, "rho", 7.0)
        karras_schedule(steps, sigma_min, sigma_max, rho)
    elseif scheduler == "cosine"
        cosine_schedule(steps)
    elseif scheduler == "linear" || scheduler == "normal"
        beta_start = get(params, "beta_start", 0.00085)
        beta_end = get(params, "beta_end", 0.012)
        linear_schedule(steps, beta_start, beta_end)
    else
        error("unknown scheduler: $scheduler")
    end

    return Dict("sigmas" => sigmas)
end

"""
    handle_sample_step(params::Dict) -> Dict

Perform a single sampler step.
Params: sampler (string), x (array), denoised (array), sigma (float), sigma_next (float), eta (float, optional)
"""
function handle_sample_step(params::Dict)
    sampler = get(params, "sampler", "euler")
    x = Float32.(get(params, "x", Float32[]))
    denoised = Float32.(get(params, "denoised", Float32[]))
    sigma = Float32(get(params, "sigma", 1.0))
    sigma_next = Float32(get(params, "sigma_next", 0.0))

    result = if sampler == "euler"
        euler_step(x, denoised, sigma, sigma_next)
    elseif sampler == "ddim"
        eta = Float32(get(params, "eta", 0.0))
        ddim_step(x, denoised, sigma, sigma_next, eta)
    else
        error("unknown sampler: $sampler")
    end

    return Dict("x" => result)
end

"""
    main()

Main event loop: read JSON lines from stdin, dispatch, write JSON lines to stdout.
"""
function main()
    # Signal readiness
    println(JSON.json(Dict("id" => 0, "result" => Dict("status" => "ready"), "error" => nothing)))
    flush(stdout)

    for line in eachline(stdin)
        line = strip(line)
        isempty(line) && continue

        request = try
            JSON.parse(line)
        catch e
            println(JSON.json(Dict("id" => 0, "result" => nothing, "error" => "invalid JSON: $(string(e))")))
            flush(stdout)
            continue
        end

        response = handle_request(request)
        println(JSON.json(response))
        flush(stdout)

        # Exit on quit
        if get(request, "method", "") == "quit"
            break
        end
    end
end

# Run the server
main()
