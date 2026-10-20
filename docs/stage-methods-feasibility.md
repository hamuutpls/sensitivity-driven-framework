# Stages 1-4: methods, libraries and where they run

Checked 2026-10-04 against PyPI (latest release of each package, which wheels it publishes). Targets: Mohammad's
PC (Windows, RTX 5070 Ti, a Blackwell card, compute capability 12.0, which needs PyTorch built for CUDA 12.8 or
newer) and Google Colab (Linux; T4, L4 or A100).

## In plain words

Most published compression tools were written for Linux servers and ship code that must be compiled for each
graphics card. On Windows, and on a card as new as the 5070 Ti, that compilation often fails. So the plan is:

1. **Write each compression method in plain PyTorch inside this repo** (as Stage 0 already does). That runs the
   same on the PC and on Colab, follows the Stage 0 plan layer by layer (the libraries mostly apply one setting
   to every layer), and gives the real accuracy of the compressed model.
2. **Measure real speed and real file size in Stage 4**, by exporting the compressed model to the programs people
   use to run models (llama.cpp on the PC; vLLM and TensorRT-LLM on Colab or WSL2, since they only run on Linux).

Until a method is written, `main.py` (MODE `"stages"`) lists it as a failed row that says "not implemented yet".

## Stage 1: quantization, weights

| Method | Library on PyPI | Windows + 5070 Ti | Colab | Plan |
|---|---|---|---|---|
| RTN (baseline) | none | yes | yes | **done** (`rtn`) |
| GPTQ | `gptqmodel` 7.5.0: source only, compiles CUDA kernels; `auto-gptq` 0.7.1: abandoned (2024) | build is unreliable, no Blackwell wheel | works | **done** (`gptq`, `sdf/stages/weights.py`) |
| AWQ | `autoawq` 0.2.9: deprecated (last release May 2025), source only; successor is `llmcompressor` 0.14 (pure Python) | `llmcompressor` installs, its output runs in vLLM (Linux) | works | **done** (`awq`: scale search + RTN, no clip search) |

| OmniQuant, SqueezeLLM, SpQR, EfficientQAT (Block-AP) | forks `hamuutpls/<name>`: research repos | – | works | **done**, simplified torch ports in `sdf/stages/weights_a.py` (fork file and commit above each method) |
| AQLM, QuIP, QuIP#, PB-LLM, BiLLM | forks `hamuutpls/<name>`: research repos, custom kernels | – | works | **done**, simplified torch ports in `sdf/stages/weights_b.py`; BiLLM has no per-layer bit width, so only its standard row runs |
| bitsandbytes | `bitsandbytes` 0.50 | CUDA wheel | works | **done** (`sdf/stages/bnb.py`): NF4 and int8 only |
| QTIP, ABQ-LLM | forks: CUDA pipelines with their own packed checkpoints | – | – | **not ported**: recorded as failed rows with the reason |

## Stage 1: quantization, activations

| Method | Library | Windows + 5070 Ti | Colab | Plan |
|---|---|---|---|---|
| RTN (baseline) | none | yes | yes | **done** (`rtn_act`) |
| SmoothQuant | none needed | yes | yes | **done** (`sdf/stages/activation_methods.py`), `smoothquant_alpha` is in the search space |
| QuaRot | `fast-hadamard-transform` 1.1.0: source only (CUDA) | build is unreliable | works | **done** with torch matrices. The residual stream gets a random-sign Hadamard matrix (2048 is a power of two); the down-projection input (width 5632 = 44 x 128, not a power of two) gets a random orthogonal matrix from a QR factorisation instead of the repo's Kronecker-product Hadamard, and the `o_proj` input only a per-head rotation. |
| RPTQ | research repo only | – | – | **done**: channel clustering + per-cluster scales, no physical reorder |
| SpinQuant | research repo only | – | – | **done**: learns R1 on a proxy loss; the slowest Stage 1 activation method |

## Stage 2: pruning

Prunes only, rounds nothing (low-rank stores its factors at the plan's bits). Runs on the FP16 model, or after a
Stage 1 weight method (`<pruning>_after_<quantization>`).

| Method | Library | Windows + 5070 Ti | Colab | Plan |
|---|---|---|---|---|
| Structured pruning | `torch-pruning` 1.6.1 (pure Python) | yes | yes | **done** (`structured_prune`: feed-forward channels) |
| Unstructured pruning (Wanda) | none needed | yes | yes | **done** (`unstructured_prune`) |
| Low-rank (SVD) | none needed | yes | yes | **done** (`low_rank`: activation-aware SVD) |

## Stage 3: KV cache

| Method | Library | Windows + 5070 Ti | Colab | Plan |
|---|---|---|---|---|
| RTN (baseline) | none | yes | yes | **done** (`rtn_kv`; its Our method row forgets with H2O's rule, so it matches H2O's) |
| QuaRot KV | as QuaRot | | | in repo (`stages/kv_methods.py`): keys after RoPE and values rotated by a Hadamard matrix per head, rounded per token and head; standard row keys at `quarot_k_bits` |
| KVQuant | research repo only | – | – | in repo, simplified port: pre-RoPE per-channel keys with calibrated ranges, per-token values, non-uniform levels from plain k-means, 1% outliers and the first word unrounded |
| H2O | research repo only | – | – | in repo, attention function: a fixed cache of ceil(share × window) words, filled first, then half recent words and half heavy hitters, one evicted per new word |
| SnapKV | research repo only | – | – | in repo, attention function: every word treated as a prompt end, window 32, max-pool kernel 7 |
| InfiniGen | research repo with custom offloading | no | partly | in repo, token selection only (guess from 30% of q/k channels, no offline skewing); the CPU offloading is not run, so latency is not comparable |

All Stage 3 rows are simulated on HF (accuracy real, memory predicted). Standard rows: rounding techniques at
`kv_uniform_bits` keeping every word, forgetting techniques at 16 bits keeping `kv_standard_keep_ratio` (20%). Our
method rows follow the whole KV plan (bits and kept share); rtn_kv, QuaRot KV and KVQuant forget with H2O's rule, so
rtn_kv's Our method row is the same computation as H2O's (Mohammad's choice, 2026-10-18). With SnapKV and InfiniGen a
layer keeps its share of the words each word can see so far, as Stage 0 sized the plan; H2O's rule keeps a fixed
cache of ceil(share × window) words, filled before anything is forgotten (Mohammad, 2026-10-20), so at the same share
it keeps more words on average than SnapKV and InfiniGen (at 20% of 512: about 93 against 52). No backend
can forget words or set per-layer KV bits, so llama.cpp, vLLM and TensorRT-LLM run only the uncompressed row and
every Stage 3 row says why (Mohammad, 2026-10-18).

## Stage 4: backends and downstream tasks

| What | Package | Windows + 5070 Ti | Colab |
|---|---|---|---|
| HF Transformers | `transformers` | yes (today's measurements) | yes |
| llama.cpp | `llama-cpp-python` 0.3.36 is source only (needs Visual Studio + CUDA toolkit); the llama.cpp GitHub releases ship ready-made Windows CUDA programs (`llama-perplexity`, `llama-bench`); `gguf` 0.19 (pure Python) converts the model | yes, with the release programs | yes (build with cmake, a few minutes) |
| vLLM | `vllm` 0.30.0: Linux wheels only | WSL2 only | yes |
| TensorRT-LLM | `tensorrt-llm` 1.2.1: source on PyPI, Linux wheels from NVIDIA | WSL2 only | heavy install; L4 or A100 |
| Downstream tasks | `lm-eval` 0.4.13 (pure Python) | yes | yes |
| Search | `optuna` 5.0.0 (pure Python), `pymoo` 0.6.2 (Windows wheels) | yes | yes |

Open points for Stage 4 (decide when we get there):

- llama.cpp's own perplexity tool cuts the text into windows its own way; to keep conditions identical, feed it the
  same WikiText-2 windows (same tokenizer) or report it as a separate, labelled measurement.
- `bitsandbytes` 0.50.2 has Windows wheels and stores 4/8-bit weights for real in HF Transformers, which would give
  a real file size and memory on the PC before the other backends are set up.
