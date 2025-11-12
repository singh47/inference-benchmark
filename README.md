# Inference-benchmark

A comparative benchmark of ONNX Runtime (CUDA FP32) and TensorRT FP16 inference performance, executed via Triton Inference Server (Python Backend). 
This repository can be extended to add more inference benchmarks from different inference backends and harwares.

## 📂 Repository Structure

- results/ # Screenshots and summarized results
- scripts/ # Benchmark and setup scripts
- docs/ # documentation for GitHub Pages

# ResNet50 Inference Performance Benchmark


## Test Environment

**Hardware**
- **GPU:** NVIDIA RTX 2000 Ada Generation  
- **CPU:** AMD EPYC 7352 (24 cores / 48 threads)  
- **RAM:** 256 GB  
- **Platform:** Runpod cloud instance  

**Software**
- **Inference Server:** PyTriton (Triton Inference Server)
- **Backend:** Python Backend  
- **Inference Engine:** ONNX Runtime
  - CUDAExecutionProvider (GPU)
  - CPUExecutionProvider (CPU)
- **Model:** ResNet50-v1-7 (ONNX)
- **Input:** 3×224×224 float32 images

> ⚙️ Note: This setup calls ONNX Runtime through the Python backend — expect ~1–2 ms Python overhead compared to the native C++ backend.

## Architecture Overview
```
┌─────────────────────────────────────────────────┐
│  PyTriton (Triton Inference Server wrapper)     │
│  - Handles HTTP/gRPC requests                   │
│  - Manages batching & queueing                  │
│  - Provides metrics & monitoring                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Python Backend (Triton's Python executor)      │
│  - Runs your Python inference function          │
│  - Executes on "CPU device 0" (Python worker)   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  ONNX Runtime (Actual inference engine)         │
│  - CUDAExecutionProvider (runs on GPU)          │
│  - Executes the neural network                  │
└─────────────────────────────────────────────────┘
```

## Benchmark Results Summary

### GPU (CUDA FP32)

| Config | Throughput (infer/sec) | Latency p99 (ms) | Avg Compute (ms) | Queue (ms) |
|--------|------------------------|------------------|------------------|------------|
| Batch 1 × Conc 1 | 121.68 | 8.99 | 6.12 | 0.07 |
| Batch 1 × Conc 10 | **163.35** | 65.38 | 5.92 | 52.64 |
| Batch 8 × Conc 10 | 73.19 | 142.81 | 67.36 | 63.11 |

### CPU (CPUExecutionProvider)

| Config | Throughput (infer/sec) | Latency p99 (ms) | Avg Compute (ms) | Queue (ms) |
|--------|------------------------|------------------|------------------|------------|
| Batch 1 × Conc 10 | 4.61 | 2,496 | 215 | 1,920 |
| Batch 8 × Conc 10 | 8.24 | 1,600 | 608 | 520 |

### TensorRT FP16 (Optimized)

| Config | Throughput (infer/sec) | Latency p99 (ms) | Compute (ms) |
|--------|------------------------|------------------|--------------|
| Concurrency 1 | 157.86 | 7.03 | 4.03 |
| Concurrency 10 | **239.50** | 45.88 | 3.97 |

---
### Screenshot

More screenshots are under /results directory

<img width="1364" height="446" alt="Screenshot 2025-11-10 at 5 35 12 PM" src="https://github.com/user-attachments/assets/035f882c-b2fb-4c65-a2ae-a2873ae6d7a3" />

---

## Performance Comparison

| Metric | GPU (Best) | CPU (Best) | GPU Advantage |
|--------|-------------|------------|----------------|
| Throughput | 163.35 infer/sec | 8.24 infer/sec | **19.8× faster** |
| Latency (p99) | 8.99 ms | 1600 ms | **178× faster** |
| Compute per image | 6.12 ms | 121 ms | **19.8× faster** |

### TensorRT FP16 vs CUDA FP32

- **+30–47%** higher throughput  
- **–22–33%** lower latency  
- **~50%** lower GPU memory use  

---

## Key Findings

1. **GPU Optimal Setup:** Batch 1, Concurrency 10 → 163 infer/sec @ 65 ms p99  
2. **CPU Bottleneck:** 20× slower, ~12% total utilization across 48 cores  
3. **Batching Impact:**
   - CPU: +79% throughput improvement  
   - GPU: –55% throughput reduction (already saturated)  
4. **TensorRT FP16:** +47% throughput, –33% compute time  

---

## 🏁 Recommendations

| Use Case | Recommended Config |
|-----------|--------------------|
| **Production (Throughput)** | TensorRT FP16, Concurrency 10 |
| **Real-Time (Latency)** | TensorRT FP16, Concurrency 1 |
| **Fallback (No GPU)** | CPU Concurrency 10 (≈ 20× slower) |

---


## Results & Reproduction

See `results/` for perf analyzer output and logs.  
Use `scripts/` to reproduce.

---

## View Online

This report is also hosted as a GitHub Page:  
👉 [**Benchmark Report**](https://singh47.github.io/inference-benchmark)

---

## 📄 License

MIT License © 2025 [Harman Singh]


