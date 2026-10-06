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

## Stage 1: weights

| Method | Library on PyPI | Windows + 5070 Ti | Colab | Plan |
|---|---|---|---|---|
| RTN (baseline) | none | yes | yes | **done** (`rtn`) |
| GPTQ | `gptqmodel` 7.5.0: source only, compiles CUDA kernels; `auto-gptq` 0.7.1: abandoned (2024) | build is unreliable, no Blackwell wheel | works | **done** (`gptq`, `sdf/stages/weights.py`) |
| AWQ | `autoawq` 0.2.9: deprecated (last release May 2025), source only; successor is `llmcompressor` 0.14 (pure Python) | `llmcompressor` installs, its output runs in vLLM (Linux) | works | **done** (`awq`: scale search + RTN, no clip search) |
| Structured pruning | `torch-pruning` 1.6.1 (pure Python) | yes | yes | **done** (`structured_prune`: feed-forward channels) |
| Unstructured pruning (Wanda) | none needed | yes | yes | **done** (`unstructured_prune`) |
| Low-rank (SVD) | none needed | yes | yes | **done** (`low_rank`: activation-aware SVD) |

## Stage 2: activations

| Method | Library | Windows + 5070 Ti | Colab | Plan |
|---|---|---|---|---|
| RTN (baseline) | none | yes | yes | **done** (`rtn_act`) |
| SmoothQuant | none needed | yes | yes | in repo, `smoothquant_alpha` is in the search space |
| QuaRot | `fast-hadamard-transform` 1.1.0: source only (CUDA) | build is unreliable | works | in repo with a torch Hadamard matrix. TinyLlama's feed-forward width 5632 = 44 x 128 is not a power of two, so the down-projection needs the Kronecker product of a 44 x 44 and a 128 x 128 Hadamard matrix, |
| RPTQ | research repo only | – | – | in repo (channel clustering + per-cluster scales) |
| SpinQuant | research repo only | – | – | in repo; learns its rotations, so it is the slowest Stage 2 method |

## Stage 3: KV cache

| Method | Library | Windows + 5070 Ti | Colab | Plan |
|---|---|---|---|---|
| RTN (baseline) | none | yes | yes | **done** (`rtn_kv`, bits only) |
| QuaRot KV | as QuaRot | | | in repo, shares the Hadamard code with QuaRot; `quarot_k_bits` is in the search space |
| KVQuant | research repo only | – | – | in repo (pre-RoPE per-channel keys, dense-and-sparse outliers) |
| H2O | research repo only | – | – | in repo, attention hook; uses the Stage 0 per-layer token budget |
| SnapKV | research repo only | – | – | in repo, attention hook |
| InfiniGen | research repo with custom offloading | no | partly | in repo, token selection only; its speed claim depends on CPU offloading, so latency will not be comparable |

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
