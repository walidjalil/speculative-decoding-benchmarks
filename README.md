# Speculative Decoding for Low-Latency LLM Inference

## Overview
This repository implements and benchmarks **speculative decoding** as a latency-optimization technique for large language model (LLM) inference.

The project is **engineering-oriented**, not research-driven.  
The focus is on correctness, performance, and cost-efficiency rather than model training or downstream task quality.

Speculative decoding is treated explicitly as a **rejection sampling procedure**, ensuring that the final output distribution exactly matches that of the target model.

---

## Motivation
Modern LLM inference is often constrained by **latency and cost**, especially in interactive or real-time settings.

Speculative decoding addresses this by:
- using a smaller *draft model* to propose multiple tokens
- verifying proposals with a larger *target model*
- emitting accepted tokens without changing the target distribution

This repository explores when and why this approach is beneficial in practice.

---

## Goals
- Implement speculative decoding with **distributional correctness**
- Compare speculative decoding against standard autoregressive decoding
- Measure:
  - latency
  - throughput
  - acceptance rate
  - GPU memory usage
  - cost per generated token
- Evaluate trade-offs across different GPU architectures

---

## Non-Goals
This project does **not** aim to:
- train foundation models from scratch
- optimize prompt quality or downstream task performance
- build RAG systems, agents, or application-layer demos
- publish academic research

---

## Method (High-Level)
Speculative decoding is implemented as follows:

1. A draft model proposes a sequence of tokens
2. The target model evaluates the proposed tokens
3. Tokens are accepted or rejected according to a deterministic acceptance rule
4. Rejected tokens are resampled to preserve the target distribution

The implementation follows the standard formulation of speculative decoding as rejection sampling.

---

## Evaluation Metrics
The following metrics are reported for all experiments:

- **Latency** (milliseconds per generated token)
- **Throughput** (tokens per second)
- **Acceptance rate**
- **Cost per generated token**
- **GPU memory utilization**

All benchmarks are run with controlled batch sizes and sequence lengths.

---

## Hardware
Experiments are conducted on the following GPUs:

- NVIDIA A100
- NVIDIA H100

Hardware configuration, precision mode, and runtime parameters are documented per experiment.

---

## Reference

This project is based on the speculative decoding / speculative sampling approach introduced in:

- Leviathan, Y., Kalman, M., & Matias, Y.  
  *Accelerating Large Language Model Decoding with Speculative Sampling* (2022)  
  https://arxiv.org/abs/2211.17192


  ## Repository Structure
```text
.
├── src/
│   ├── decoding/
│   │   ├── baseline.py        # Standard autoregressive decoding
│   │   └── speculative.py     # Speculative decoding implementation
│   ├── models/
│   └── utils/
├── benchmarks/
├── scripts/
└── README.md


---
