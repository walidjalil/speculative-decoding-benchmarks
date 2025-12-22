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



## Background: Speculative Decoding (Intuition)

Speculative decoding can be thought of as a practical form of **rejection sampling** applied to language model decoding.

The basic idea is simple:

- A smaller **draft model** proposes several future tokens.
- A larger **target model** then checks these proposals using a deterministic acceptance rule.
- Accepted tokens are guaranteed to follow **the exact same distribution** as standard autoregressive decoding.

In other words, output quality stays the same — only **latency** improves.

### Why this works

Rejection sampling requires the proposal and target distributions to share the same support.  
In speculative decoding, this is naturally satisfied because both models use the **same tokenizer and vocabulary**.

In practice, the draft model is usually a **smaller or distilled version** of the target model. This makes the two distributions similar, which increases the acceptance rate and directly improves **latency**.

- Shared support → guarantees correctness
- Distributional similarity → determines **latency gains**

### Faster decoding without quality loss

When the draft model closely matches the target model:
- Most proposed tokens are accepted
- Multiple tokens can be verified in a single forward pass
- End-to-end decoding **latency** is significantly reduced

If proposals are rejected:
- The algorithm safely falls back to the target model
- Correctness is still preserved
- **Latency** benefits degrade gracefully rather than breaking

The result is:
- **Lower decoding latency**
- **No quality or performance penalty**
- At the cost of running an additional (smaller) draft model in parallel

### In short

Speculative decoding trades extra draft-model compute and memory for reduced inference **latency**, while preserving the exact output distribution.

---

## Reference

This project is based on the speculative decoding / speculative sampling approach introduced in:

- Leviathan, Y., Kalman, M., & Matias, Y.  
  *Accelerating Large Language Model Decoding with Speculative Sampling* (2022)  
  https://arxiv.org/abs/2211.17192

