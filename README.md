# Speculative Decoding for LLM Inference

## Goal
Implement and benchmark speculative decoding for LLM inference.

## Models
- Draft model: ~3B
- Target model: 8B–30B

## Baseline
- Standard autoregressive decoding

## Metrics
- Tokens/sec
- Speedup
- Acceptance rate

## Infrastructure
- RunPod (A100 40GB)
- Single- and multi-GPU setups

## TODO
- Implement baseline decoding
- Implement speculative decoding
- Benchmark single GPU
- Benchmark multi-GPU
