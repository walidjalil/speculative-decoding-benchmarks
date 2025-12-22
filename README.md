# Speculative Decoding for LLM Inference

## Goal
Implement and benchmark speculative decoding for LLM inference.

## Models
- Draft model: ~3B
- Target model: 8B–30B  (Haven't decided, might go for bigger models since I have access to compute)

## Baseline
- Standard autoregressive decoding

## Metrics
- Tokens/sec
- Speedup
- Acceptance rate

## Infrastructure
- RunPod (A100 40GB GPUs)
- Single- and multi-GPU setups

## TODO
- Implement baseline decoding
- Implement speculative decoding
- Benchmark single GPU
- Benchmark multi-GPU



## Background: Speculative Decoding as Rejection Sampling

Speculative decoding can be understood as a form of **rejection sampling** applied to autoregressive language model decoding.

In this framework:

- A **draft model** acts as a *proposal distribution*, generating multiple candidate tokens ahead of time.
- A **target model** verifies these tokens using a deterministic acceptance rule.
- Accepted tokens are guaranteed to follow **exactly the same distribution** as standard autoregressive decoding.

### Why this works

Rejection sampling requires that the proposal and target distributions share the same support.  
In speculative decoding, this condition is satisfied because:

- Both models operate over the **same token vocabulary**.
- The draft model is typically a **smaller or distilled version** of the target model.

As a result:
- **Correctness** is guaranteed by shared support and the acceptance rule.
- **Efficiency** depends on how closely the draft model matches the target distribution.

### Speed without quality loss

If the draft model closely approximates the target model:
- Most proposed tokens are accepted.
- Multiple tokens can be verified in a single forward pass.

If tokens are rejected:
- The algorithm safely falls back to the target model.
- Output correctness is still preserved.

This leads to:
- **Improved decoding latency**
- **No quality or performance degradation**
- At the cost of running an additional (smaller) draft model alongside the target model.

### Key takeaway

Speculative decoding trades additional draft-model computation and memory for lower inference latency, while preserving the exact output distribution via rejection sampling.
