# Running the framework on another model

The framework was built and tested on TinyLlama-1.1B. This page says how to point it at another Hugging Face
model, and which parts only work on Llama-style models. Everything here is what the code does today.

## 1. Install

```bash
pip install -e ".[dev,search,downstream,figures]"  # search = Optuna + pymoo, downstream = lm-eval, figures = plots
pip install pytest-xdist && pytest -n 4                # or plain `pytest`
```

The model is loaded with `AutoModelForCausalLM.from_pretrained(name)`, with no `trust_remote_code`, so it must be
an architecture built into `transformers`. A gated model (Llama 3, Gemma) needs `HF_TOKEN` set.

Memory: Stage 0 first loads the model at full precision (4 bytes per weight) and measures in float32 by default;
Stages 1-3 and the search use float16 (2 bytes per weight). So a 7B model needs about 28 GB of RAM to load and a
large GPU. With no GPU, `device: auto` quietly uses the CPU, which is very slow for anything above about 1B.

## 2. Check the model first

Most of the framework finds layers by type, not by name (it looks for `model.model.layers`, else the longest list
of layers). The main parts that use names are below. This script (CPU, a few
seconds, random weights) tells you if a model has them:

```python
from transformers import AutoConfig, AutoModelForCausalLM
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stage0.kv_cache import kv_projections
from sdf.stages.kv_methods import attention_module

name = "Qwen/Qwen2.5-0.5B"  # the model to check
cfg = AutoConfig.from_pretrained(name, num_hidden_layers=2)  # 2 layers, random weights: only the names matter
layer = find_decoder_layers(AutoModelForCausalLM.from_config(cfg))[0]
kv_projections(layer, ("k_proj", "v_proj"))  # Stage 0 KV plan, Stage 3
attention_module(layer)  # Stage 3
llama = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
         "input_layernorm", "post_attention_layernorm"]  # SmoothQuant, QuaRot, SpinQuant
names = dict(layer.named_modules())
print("missing Llama names:", [n for n in llama if n not in names] or "none")
```

If `kv_projections` fails, the KV plan and Stage 3 cannot run. If `attention_module` fails, Stage 3 cannot run.
Missing Llama names rule out SmoothQuant, QuaRot and SpinQuant. Checked: Qwen2.5-0.5B, Qwen2.5-1.5B and
Mistral-7B pass; `facebook/opt-125m` fails at `attention_module`; `gpt2` fails at `kv_projections`; `microsoft/phi-2`
passes both but misses 5 Llama names. Stage 0 also ran end to end on Qwen2.5-0.5B on a CPU with small settings
(16 passages of 128 tokens, KV and joint plans off).

## 3. Settings to change

Most settings are at the top of `main.py`. A config key with no `main.py` setting (shown as —) is set by adding
one line to `build_config()` in `main.py`, e.g. `"stage0.profile_dtype": "bfloat16",`. (`sdf-stage0 --set key=value`
also works, but runs Stage 0 only.) For another model, look at these:

| Setting in `main.py` | Config key | What to do |
|---|---|---|
| `MODEL` | `model.name` | The Hugging Face name, e.g. `"Qwen/Qwen2.5-0.5B"`. |
| `CALIB_DATASET`, `CALIB_SAMPLES` | `hyperparams.*` | Calibration text: `wikitext2`, `c4` or `pile10k`, 16 to 128 passages. Another dataset needs a code change: the allowed names are in `src/sdf/search_space.py`, and the source goes in `data.sources` with a `train` split and a `text` column. |
| `CALIB_SEQ_LEN`, `EVAL_SEQ_LEN` | `calibration.seq_len`, `eval.seq_len` | 512 tokens by default. Keep them under the model's context length. |
| `TARGET_PPL` | `requirement.target_ppl` | The perplexity limit for the search. The value in `main.py` was picked for TinyLlama. Perplexity differs a lot between models, so run `MODE = "single"` first, read the FP16 "Prediction error on test text (validation half)" (`ppl_val`) in `stage_0/report.md`, and set the limit from that. |
| `TARGET_MEMORY_GB` | `requirement.target_memory_gb` | Size limit in GB (`None` = no limit). Also model-specific. |
| `KV_MODULE_NAMES` | `stage0.kv_module_names` | The Linear layers that make keys and values. `["k_proj", "v_proj"]` fits Llama, Mistral, Qwen. A model with one fused QKV layer cannot use the KV plan. |
| `GUARD_TOP_K` | `stage0.guard_top_k` | How many layers are never pruned (5). The guarded layers are picked by measurement, not by number, so this works for any model; with many more layers you may want more. |
| — | `stage0.profile_dtype` | Stage 0 measures in float32 (4 bytes per weight). For a 7B model on a GPU, set `bfloat16`. |
| — | `requirement.hardware_profile` | A label saved with the run's config and results (`configs/tinyllama.yaml` sets `RTX-5070-Ti`). |

Use a new `CACHE_DIR` or `RUN_ID` per model if you like; cache keys include the model name, so results of two
models never mix.

## 4. Stage by stage

| Stage | `MODE` | Works on any decoder model? |
|---|---|---|
| 0 Sensitivity plan | `single` (also `sweep`, `compare_scores`, `prune_sweep`, `threshold_sweep`) | Yes. Each layer is skipped (or rounded) and the perplexity rise measured. The KV plan needs the `k_proj`/`v_proj` names (above), and its token-keeping part needs the model to return attention weights (`output_attentions=True`); it stops with an error if not. |
| 1 Quantization | `stages` | Weight methods (RTN, GPTQ, AWQ, OmniQuant, SqueezeLLM, SpQR, EfficientQAT, QuIP, QuIP#, PB-LLM, bitsandbytes) and `rtn_act`, RPTQ work on every Linear in each layer (AWQ also scales `v_proj`/`o_proj` when they exist). **SmoothQuant, QuaRot and SpinQuant use Llama names** (`input_layernorm`, `post_attention_layernorm`, `q/k/v_proj`, `gate/up/down_proj`); QuaRot and SpinQuant also use `model.model.norm`, `model.model.embed_tokens` and `lm_head`. |
| 2 Pruning | `stages` | Yes. Unstructured and low-rank work on every Linear; structured finds the feed-forward block by its shape (using `config.hidden_size`). |
| 3 KV cache | `stages` | Needs `.self_attn` with `num_key_value_groups`. Stage 3 swaps in its own attention code, which is plain full attention: it ignores sliding windows and score caps, so on Gemma 2/3 or models with sliding-window layers the results change with no error. |
| 4 Backends | rows of `stages`, and the final model of `search` | See section 6. |

`STAGE0_DIR` reuses a finished Stage 0 run; it must be a run of the same model (nothing checks this).

## 5. The search

`MODE = "search"` builds chained models 0 → 1 → 2 → 3, picking settings with two search methods (MOBO and MFBO;
NSGA-III also works), and keeps the
smallest one whose validation perplexity is at most `TARGET_PPL`. Only that final model goes to Stage 4 and the
accuracy tests (`SEARCH_ACCURACY_TASKS`), next to the uncompressed model.

For another model, check `SEARCH_STAGE1`, `SEARCH_STAGE2` and `SEARCH_STAGE3`: drop the techniques the model
cannot run (section 4). In particular, the default `SEARCH_STAGE1` pairs every weight method with `smoothquant`,
which needs Llama names. The modes `stage_search` and `chain` run the same stages in other orders and have the same model
needs. Every combination tried is written to `<CACHE_DIR>/search_tried.csv` and not built
again (failed ones are); the key includes the model, so another model's tries are not reused.

## 6. Backends (Stage 4)

| Backend | Setting | Model needs |
|---|---|---|
| Hugging Face | always on | — |
| llama.cpp | `eval.llamacpp_dir`, `eval.llamacpp_convert` | An architecture llama.cpp's `convert_hf_to_gguf.py` supports. Per-layer bits use llama.cpp's layer names `blk.<i>.`, which every converted model has. A SentencePiece tokenizer must have its `tokenizer.model` file on the Hub. |
| vLLM | `VLLM_PYTHON` (`eval.vllm_python`), Linux only | An architecture vLLM supports, with module names containing `layers.<i>.` and an output head named `lm_head` (true for Llama, Mistral, Qwen; not GPT-2). |
| TensorRT-LLM | `eval.trtllm_python`, Linux only | An architecture TensorRT-LLM converts. Mixed-bit plans name layers `transformer.layers.<i>.attention` / `.mlp`, TensorRT-LLM's Llama-family names. |

## 7. Where results go

```
<OUTPUT_ROOT>/<run_id>/
  stage_0/                  report.md, stage_0_comparison.xlsx, results.json, the plans (*.json), sensitivity_profile.json
  stage_0_pruning_levels/   the pruning levels test of every Stage 0 run
  stage_0_<study>/          the other Stage 0 modes (sweep, scores, prune_sweep, threshold_sweep)
  stage_1/ stage_2/ stage_3/  the same three report files per stage (MODE "stages")
  all_stages_comparison.xlsx, master_report.md   (MODE "stages")
  search/                   search_report.md, search_results.xlsx, trials.json, plots, points/<hash>/
<CACHE_DIR>/                FP16 baseline, sensitivity profiles, method results, search_tried.csv
```

`OUTPUT_ROOT` is `thesis_compression/results` and `CACHE_DIR` `thesis_compression/cache`, relative to where you run.

## 8. What assumes Llama or TinyLlama

| Where | What | What to change for another model |
|---|---|---|
| `stages/activation_methods.py` | SmoothQuant, QuaRot and SpinQuant use Llama names and assume Llama's norm layer (RMSNorm, no bias). QuaRot and SpinQuant also assume a head size of `hidden_size / num_attention_heads`, use `config.intermediate_size`, fold the norm as weight × w (wrong for Gemma's 1 + w norm), and do not rotate biases on `v_proj`, `o_proj` or `down_proj`. | On **Qwen2** (it has a `v_proj` bias) the QuaRot / SpinQuant rotation changes the model's output before any rounding, so their results are wrong (checked on a small random Qwen2; on Llama the output stays the same). Leave QuaRot and SpinQuant out for such models, or fix the code to rotate the bias too. |
| `stages/kv_methods.py` | Stage 3 needs `.self_attn.num_key_value_groups`. `FORGET_PPL_BATCH = 16` was sized for TinyLlama (~5 GB). | Lower `FORGET_PPL_BATCH` for a 7B model or 2048-token windows. |
| `stages/methods.py` | AQLM and QuIP# size counts assume 7 Linears per layer (`_LINEARS = 7`). | Set it to the model's count (Llama, Mistral, Qwen have 7). |
| `eval/vllm_worker.py` | KV cache fixed at 1 GB (`kv_cache_memory_bytes`). | No setting; change it in the code if a bigger model needs more. |
| `eval/trtllm_worker.py` | KV cache of 46,000 tokens, about 1 GB for TinyLlama. | A bigger model needs more memory for the same tokens; lower the 46000 in `eval/trtllm_worker.py` line 79 if the GPU runs out (no setting). |
| `eval/trtllm_backend.py` | Mixed-bit plans use TensorRT-LLM's Llama-family layer names. | Check the names TensorRT-LLM gives your model. |
| `reporting/search.py` | The accuracy table text says "TinyLlama". | Text only; the numbers come from the model you ran. |
| `configs/tinyllama.yaml`, `main.py` | Default model and limits. | Section 3. |

The tests (`tests/`) build only small Llama models, so no other architecture is covered by them.
