# The engines' own KV-cache methods on the final models

Commit b23b0c62afd88c052b07e98de74321977a54974e; model TinyLlama/TinyLlama-1.1B-Chat-v1.0; seed 0; written 2026-11-23 23:52:18 UTC.

SnapKV and InfiniGen cannot run in vLLM or llama.cpp, so the KV step is replaced by each engine's own KV-cache storage: vLLM kv_cache_dtype fp8 (8 bits) and llama.cpp cache type q4_0 (4 bits, keys and values), against each engine's 16-bit cache. Every model is built without its Stage 3 step; the two rows of a model differ only in the KV cache. vLLM reserves a fixed 1 GB for the cache, so its peak memory does not change; vllm_kv_cache_tokens is how many tokens that space holds. llamacpp_kv_cache_gb is the cache llama-perplexity allocated (all windows it reads at once). config.json holds the last row's settings; each row's KV settings are in its CSV columns.

| model | KV cache | vllm_size_gb | vllm_ppl_val | vllm_ppl_heldout | vllm_decode_ms_per_token | vllm_prefill_ms | vllm_peak_memory_gb | vllm_kv_cache_tokens | llamacpp_size_gb | llamacpp_ppl_val | llamacpp_ppl_heldout | llamacpp_decode_tokens_per_s | llamacpp_prefill_tokens_per_s | llamacpp_kv_cache_gb | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_fp16 | 16-bit KV cache | 2.2 | 10.18 | 10.62 | 2.537 | 8.907 | 3.766 | 4.765e+04 | 2.201 | 8.121 | 8.564 | 437.1 | 2.255e+04 | 0.04614 |  |
| baseline_fp16 | compressed KV cache | 2.2 | 10.21 | 10.64 | 2.743 | 8.914 | 4.49 | 9.531e+04 | 2.201 | 8.223 | 8.69 | 401 | 1.794e+04 | 0.01298 |  |
| rtn_snapkv | 16-bit KV cache | 0.8981 | 11.1 | 11.54 | 1.966 | 8.657 | 2.448 | 4.765e+04 | 0.9183 | 8.401 | 8.842 | 506.5 | 1.216e+04 | 0.04614 |  |
| rtn_snapkv | compressed KV cache | 0.8981 | 11.14 | 11.58 | 2.496 | 12.6 | 3.171 | 9.531e+04 | 0.9183 | 8.532 | 8.996 | 462.6 | 1.026e+04 | 0.01298 |  |
| omniquant_infinigen | 16-bit KV cache | 0.8375 | 12.96 | 13.37 | 2.006 | 9.054 | 2.397 | 4.765e+04 | 0.8963 | 9.206 | 9.653 | 512.5 | 1.175e+04 | 0.04614 |  |
| omniquant_infinigen | compressed KV cache | 0.8375 | 13.02 | 13.44 | 2.484 | 12.69 | 3.12 | 9.531e+04 | 0.8963 | 9.407 | 9.859 | 464.6 | 1.151e+04 | 0.01298 |  |
