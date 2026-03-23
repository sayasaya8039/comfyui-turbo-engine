# ComfyUI Turbo Engine v2.0.0

High-performance Rust backend for ComfyUI Desktop. 134ms startup, GPU+NPU parallel inference pipeline.

## Features

- **134ms startup** — instant server boot before Python loads
- **GPU+NPU parallel pipeline** — CLIP on NPU, UNet on GPU, VAE on NPU simultaneously
- **Zig SIMD kernels** — AVX 256-bit accelerated SiLU/GELU/Softmax/LayerNorm/GroupNorm
- **Rayon parallel DAG** — independent nodes execute in parallel across CPU cores
- **WASM plugin sandbox** — wasmtime-powered plugin execution with memory limits
- **283 tests** — comprehensive test coverage across 8 crates
- **Zero-copy tensors** — Arc-backed immutable tensors with fast byte reinterpretation
- **LRU tensor cache** — 4GB default with concurrent access support
- **ComfyUI REST API compatible** — drop-in replacement for Python server

## Architecture

```
8 Crates:
  comfy-core       — DAG executor, tensor, scheduler, cache, hardware detection
  comfy-inference   — ONNX Runtime (CUDA/DirectML/OpenVINO/CPU) + multi-device pipeline
  comfy-server      — Axum HTTP/WebSocket API (ComfyUI compatible)
  comfy-nodes       — 7 standard nodes + pipelined SD executor
  comfy-julia       — Diffusion samplers (Euler/DDIM/DPM++2M) + noise schedules
  comfy-zig         — SIMD kernels (Zig native + Rust fallback) + image I/O
  comfy-wasm        — wasmtime plugin sandbox
  comfy-python      — PyO3 custom node bridge
```

## GPU+NPU Pipeline

```
GPU:  [--------KSampler--------][----KSampler batch2----]
NPU:  [CLIP+][CLIP-][VAEDecode ][CLIP b2  ][VAEDecode  ]
CPU:  [Load]  [Lat]  [SaveImage] [Lat]      [SaveImage  ]
```

## Installation

### Windows Installer
Download `ComfyUI-Turbo-v2.0.0-setup.exe` from [Releases](../../releases).

### Portable ZIP
Download `ComfyUI-Turbo-v2.0.0-win-x64.zip`, extract, and run `comfy-server.exe`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFY_PORT` | 8188 | Server port |
| `COMFY_FRONTEND` | — | Path to frontend static files |
| `COMFY_VENV` | — | Python venv path for templates |

## Build from Source

```bash
cargo build --release
# Binary: target/release/comfy-server.exe (7.5MB)

# With Zig SIMD kernels:
cargo build --release --features zig-native

# Run tests:
cargo test --all
```

## License

GPL-3.0-only
