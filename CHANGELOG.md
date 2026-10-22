# Changelog

## 2026-10-22: Pruned models reach the backends (search and chain fix)

- A pruned model (Stage 2 after Stage 1, as every search try and the chain's Stage 2 rows are) was refused by the
  compressed-tensors (vLLM) and GPTQ-format (vLLM, TensorRT-LLM) exports ("is a quantization method and removes
  nothing"), so its vLLM and TensorRT-LLM numbers were missing and every search try failed.
  The backends now get the bits-only plan: the pruned weights are already zero in the model. vLLM VERSION 6 and
  TensorRT-LLM VERSION 5, so their numbers are measured again.

## 2026-10-20: the other ponytail audit fixes (audits 2026-10-19 and 2026-10-20)

How results are judged (Mohammad's 09:17 choice, 2026-10-20 findings 21 and 30):
- Backend perplexity (llama.cpp, vLLM, TensorRT-LLM) is no longer won or lost in the verdict. Each backend loads only
  its own file format, so every method's weights are rounded a second time onto a plain grid, and the number measures
  that re-rounded file, not the method. The columns stay in the tables, now named "... perplexity of the re-rounded
  file". Perplexity now counts once in every verdict (Hugging Face validation and held-out halves agreeing).
- Excel colours follow the verdict (2026-10-19 #16); the held-out perplexity half is coloured with the validation
  half when it was itself judged the same way (not when it is missing or a tie), and one-off costs are red. A row
  compared only on backend perplexity says no measure could be compared.

Runs and measurements:
- Stage 3 (22): rtn_kv and KVQuant standard rows run the same attention code as their Our rows (versions 2 and 3).
- SpinQuant (25): keeps a random sample of each calibration batch, not every input (no out-of-memory on 16 GB);
  version 2, and the `*_with_spinquant` joint rows get a new version.
- AQLM (29): no longer computes input statistics it never used, and calibration settings leave its cache key;
  same weights.
- Runner (2026-10-19 #1, #2, #4, #5, #9, #13, #14, #15, #18): a backend failure for a passing reason (busy card, out
  of memory, a failed or timed-out program) is not saved to the cache, while one that repeats on every run (a plan
  the backend's format cannot hold) is saved with the row; the FP16 row keeps its Hugging Face numbers when a backend fails; Stage 2
  levels come from the plan's own share; standard rows' cache keys hold the plan's bits and pruning only, not the
  sensitivity scores; exposure is measured against Stage 0's reference; rows of BiLLM, whose bits are fixed, drop the
  predicted size of a plan it does not follow; duplicate capped levels are skipped with a note; the export group size
  enters a key only when something is rounded; settings a method never reads (`weight_zero_point` for SqueezeLLM, SpQR,
  AQLM, QuIP#, PB-LLM, BiLLM and bitsandbytes; `act_group_size` for RPTQ's own rows) stay out of its key, and a
  Stage 2 method after a WxAy method keeps `act_group_size` in its key (it still rounds activations).
- Backends (#3): vLLM and TensorRT-LLM programs time out after 4 h, each llama.cpp program after 2 h, with the
  backend's normal error, and the program's child processes are killed with it (also when the run is interrupted). Timings record the latency
  mode that ran and rows timed in different modes are not compared (#7); a row cached before the mode was recorded is
  compared with a CUDA graph row but not with an eager one. Timing settings are checked when the config is built
  (#17).
- Stage 0 (#6, #10, #12): pruning tests leave timing settings out of their keys; sweeps that round nothing predict
  sizes at baseline bits and their text names those bits; report text follows the scaling, guard and bit settings,
  and the sensitivity column says "rank" only under rank scaling.
- Low-rank (15, lean): the unreachable factor rounding and the group size in its key are gone.

Colab (26, 27, #8):
- The notebook no longer uses Google Drive (no sign-in box); results and caches go to `/content/thesis_compression/`,
  and its last cell packs the results into `/content/thesis_results.tgz`, which the host PC session copies to
  `D:\Thesis` (on a local runtime on the PC they go to `D:\Thesis\thesis_compression\` directly; `sdf.utils.colab`).
  Step 1 says the runtime type matters only on a free plan.
- `scripts/colab_stage1.py` runs the methods given after it (`%run colab_stage1.py rtn gptq`), saves where the notebook
  saves, writes its result lines to `stage1_summary.txt` in the run folder (closed even on an error), and packs the
  results the same way; both remove an earlier archive before the run starts, so a killed run never leaves the last
  run's archive to be copied as its own.
- The notebook drops loaded `sdf` modules before reloading `main`, so a fresh pull takes effect.

Figures (28) are redrawn for Stages 3 and 4. Lean: unused `CALIB`, the duplicate `share_removed` and the bits-only KV
plan in Stages 1-3 are gone.

Numbers that change: standard weight and joint WxAy rows of Stages 1 and 2, and 16-bit pruning rows, get new cache
keys (measured again once; same values);
rtn_kv and KVQuant standard rows' speed and memory, and their perplexity and its difference from the 16-bit model may
shift slightly (eager attention instead of SDPA); SpinQuant rows; exposure columns; verdict lines and colours
wherever a backend ran; BiLLM rows lose their predicted size; low-rank and AQLM rows are measured again (same
values).

## 2026-10-20: audit fixes the final test waits for (ponytail audit 2026-10-20, findings 19, 23, 24)

- Joint plan budget (19): the joint weight-and-activation plan gets the separate activation plan's real average
  activation bits in every mode, not only when that plan comes from the weight plan. A measured plan that spends less
  than `act_avg_bits` no longer leaves the joint plan more bits than the plan it is compared with.
- Size per method (23): the predicted size counts what SqueezeLLM, SpQR, AQLM, QuIP# and bitsandbytes store besides
  their bits (codebooks, outliers, their own scales) instead of one scale and zero per `gptq_groupsize` weights.
  Assumed outlier shares: SqueezeLLM 0.45% (the paper's) at 48 bits each (a 16-bit value and a 32-bit column index,
  which matches the paper's 4.05 -> 4.27 bits per weight), SpQR about 1% at 32 bits (its own 16-bit index). Per-row and per-Linear parts (codebooks, row scales) do not shrink when weights are pruned. Our
  method's SqueezeLLM rows grow most (a 256-entry codebook per row costs about 1.6 bits per weight at 8 bits), so
  their size relative to the standard row rises. Stage 0 sizes stay method-agnostic, so a Stage 1 row of these
  methods shows a different size from the same plan in Stage 0. AQLM's codebooks are counted for a Llama layer's 7
  Linears; per-row parts are counted for every row, so after structured pruning or low-rank the size is slightly
  high. The settings search checks the size limit with the same count (search version 2, so saved points are
  redone). Only size columns change; nothing is re-measured.
- AWQ under grouped-query attention (24): o_proj is rounded plainly, as the reference implementation does when its
  scale cannot be folded into v_proj (TinyLlama: 4 key/value heads for 32 query heads). AWQ is version 3, so its
  cached standard rows are redone.

## 2026-10-20: Settings search: 1.1 GB, 100 full tries, joint plan

Mohammad's answers (2026-10-20): "over 1 gb +10% is fine. do 100 tries. and only the built models count as tries";
"the goal is joint planning"; "its ok to build 438 models".

- main.py: TARGET_MEMORY_GB 1.1, SEARCH_BUDGET 100.
- The budget is 100 full tries per searcher (a model built and measured on all the windows). MFBO's rough tries count
  by their share of the windows, so it builds more models (about 438 at budget 100) for the same compute; the
  searcher comparison shows the models built and how many were on all the windows.
- Only a model built for the searcher costs: a point the plan puts over the limit, a failure before building and a
  repeat of the searcher's own try from the cache cost nothing; a failure after building still costs (it is not
  cached, so its repeat rebuilds and costs again), so a broken setting cannot rebuild for free. `search.max_tries_per_budget` (20) stops a searcher after budget x 20 tries.
- Stage 1 always follows the joint weight x activation plan; the `stage1_plan` search setting is removed.
- MFBO gives more windows only to points it measured. MOBO completes a point over the limit with the worst values
  built so far and its shortfall as the constraint, so Optuna's GPSampler (which learns only from completed trials)
  learns where the limit is; a point that crashed or was not measured is completed the same way with constraint 1,
  so MOBO does not propose it again.
- The limit is checked on `search.memory_metric` before and after the build (`check_requirement`); it must be
  `predicted_weight_memory_gb`, the only size known before building; `ppl_heldout` is refused as an objective.
- The report says unstructured pruning gives no vLLM speed-up, counts the tries over the limit, and no longer counts
  them as failures.

## 2026-10-19: Settings search under a memory limit (MODE = "search")

Mohammad's asks: "make the framework dynamic and find the best settings that satisfy perplexity, memory, and
latency"; then memory = model size below 1 GB, latency from vLLM on Colab, 50 tries per searcher, the project's
search-space list plus a technique for each stage.

- `src/sdf/search.py`: a point is the SEARCH_SPACE hyperparameters plus a Stage 1 WxAy technique (separate or joint
  Stage 0 plan), a Stage 2 pruning technique (at the plan's prune_ratio_aggressive) and a Stage 3 KV technique.
  `evaluate` builds the chained Our method model 0 -> 1 -> 2 -> 3 and measures it (cached per point).
- Searchers, same budget and seed: MOBO (Optuna GPSampler, the requirement as a constraint), MFBO (successive halving
  over the evaluation windows) and NSGA-III (pymoo). `pip install .[search]`.
- Stage 3 rows built on a Stage 2 row now send that Stage 2 model to the backends (they cannot run the KV techniques),
  so a search point gets vLLM latency; the row says so.
- Outputs in `<run>/search/`: search_report.md, search_results.xlsx (Trials, Pareto, Searcher comparison, Best
  configs), trade-off plots and hypervolume per searcher. Settings in main.py section 8.

## 2026-10-19: Sequential run of Stages 0 -> 1 -> 2 -> 3 -> 4 (MODE = "chain")

Mohammad's ask: "start a thread that will run stages 0 1 2 3 4 sequentially", then "stage 1 should be the best wxay
pick", "test everything as long as we can reach the highest memory reduction and lowest perplexity" and "yes" to
following each side's memory vs perplexity trade-off line.

- `run_chain` (`src/sdf/stages/runner.py`, `MODE = "chain"`): Stage 1 runs `STAGE1_METHODS`; every WxAy row of a side
  (Standard, Our method) on its memory vs perplexity trade-off line is pruned by every `STAGE2_METHODS` method at
  every share (`stage_2_<side>_after_<row>/`); every Stage 2 row of the side on its line gets every
  `STAGE3_METHODS` method (`stage_3_<side>_after_<Stage 1 row>_<Stage 2 row>/`). Held-out perplexity never chooses.
- `run_stage(after=...)`: Stage 2 after a Stage 1 weight or WxAy row (`series()` now takes WxAy methods: the
  activations stay rounded while pruning calibrates and the model is measured). `run_stage(before=...)`: Stage 3 on a
  Stage 2 row. `only=` runs one side. All join the cache key.
- Stage 4 (evaluation) runs inside every row: Hugging Face always; backends for Stage 1 and 2 rows, and for a
  chained Stage 3 row on the Stage 2 model it was built on (they cannot run the KV techniques).
- Low-rank still cannot follow Stage 1, so its chained rows fail and say so.

## 2026-10-20: SnapKV and InfiniGen use H2O's cache size (fairness)

After the H2O fix, H2O kept a fixed cache of ceil(share x window) words while SnapKV and InfiniGen kept a share of the
words each word can see, so at 20% of 512 words H2O kept about 93 words on average against 52. Mohammad approved giving
all three the same cache ("do what you recommend", 2026-10-20 09:17): word i keeps min(i + 1, ceil(share x window))
words (`cache_budgets` in `stages/kv_methods.py`). SnapKV and InfiniGen are version 3, so their cached standard rows are
redone. Stage 0's coverage still sizes the plan with the visible-words share and is unchanged.

## 2026-10-20: H2O keeps a fixed cache (fix)

The 0-3-4 run of 2026-10-19 gave perplexity about 398 for every row that forgets with H2O's rule (rtn_kv, QuaRot KV
and KVQuant Our method rows, both H2O rows). Our simulation gave word i a cache of ceil(share x (i + 1)) words, so at
word 1 the cache held one word, the word itself, and the first word (the attention sink the model leans on) was
forgotten for good. Mohammad chose "A", the paper's rule: a fixed cache of ceil(share x window) words, filled before
anything is forgotten. Only H2O changed; SnapKV and InfiniGen keep the visible-words share, so at the same share H2O now
keeps more words on average (at 20% of 512 words: about 93 against 52), and more than Stage 0's coverage assumed.
H2O is version 3, so its cached standard rows are redone.

## 2026-10-18: Stage 3 KV-cache techniques

Mohammad's asks in the KV cache (Stage 3) thread: all five techniques, rounding and forgetting, HF and backends,
on Colab; then "do what you recommend" for the four build details.

- QuaRot KV, KVQuant, H2O, SnapKV and InfiniGen are implemented (`src/sdf/stages/kv_methods.py`). Forgetting is
  simulated per word in the one-pass perplexity: each word sees only what the technique would still hold.
- Standard rows follow the papers: rounding techniques at 4 bits keeping every word; H2O, SnapKV and InfiniGen at 16
  bits keeping 20% of the words (H2O's setting; SnapKV and InfiniGen set no share in their papers;
  `stage0.kv_standard_keep_ratio`, `KV_STANDARD_KEEP_RATIO`).
- Our method rows use the whole Stage 0 KV plan (`kv_cache_plan.json`: bits and kept share); rtn_kv, QuaRot KV and
  KVQuant forget with H2O's rule, so rtn_kv's Our method row is the same computation as H2O's (Mohammad's choice).
  rtn_kv's framework row used the bits-only plan before.
- Each word keeps ceil(share x the words it can see so far), as Stage 0 sized the shares (Mohammad, 2026-10-18).
- QuaRot KV's standard row stores keys at `quarot_k_bits` (QUAROT_K_BITS), and its predicted memory says so.
- `transformers>=4.56` (Stage 3's attention function needs `AttentionInterface` and `masking_utils`).
- Backends run only the uncompressed row in Stage 3; every Stage 3 row records why it is not run there.

## 2026-10-18: Stage 2 records the share of the decoder layers' weights removed

Mohammad's answer in the Stage 2 pruning thread ("yes", then "both your picks": a new column, the old one kept).

- Every Stage 2 row has a new metric, `decoder_weights_removed`: the share of the decoder layers' Linear weights the
  row's plan prunes (0.3 for a row that removes 30%). It matches the share in the row label and is listed in the
  Stage 2 results, report and spreadsheet next to `sparsity`. The uncompressed row has 0.
- `sparsity` is unchanged: it still divides by all of the model's parameters, including the embeddings, output head
  and norms that are never pruned, so it reads lower (about 0.264 for 30% in TinyLlama). Stage 0 and Stages 1 and 3
  are unchanged.
- New `planner.decoder_share_removed(plan, layer_numel)`, also used for the row label.

## 2026-10-13: Stage 2 share sweep

Mohammad's asks in the Stage 2 pruning thread ("do a sweep i dont want only 30%", "4", and the answers after).

- Stage 0 saves one Stage 2 plan pair per level in `stage0.pruning_levels` (30% to 90%):
  `prune_plan_<level>.json` and `prune_plan_same_size_<level>.json`. Stage 2 runs Standard, Our method and Our method
  same size at every level, as rows named `<method>_<level>` (e.g. `low_rank_030`). Each label gives the level and
  the share of the decoder layers' weights really removed. An older Stage 0 folder without level plans runs once, as
  before, with a warning.
- Every Stage 2 row also runs on llama.cpp, vLLM and TensorRT-LLM (before: Stage 1 only). Pruned models keep their
  zeros, so their memory and speed there match the uncompressed model.
- Structured pruning removes at most 75% of a layer, or the layer's feed-forward share when lower, in Stage 2 and
  in the Stage 0 pruning levels test (before: the feed-forward share, 78.6% in TinyLlama). A capped same-size plan
  moves what a capped layer cannot take onto the other unguarded layers, so it removes as much as Standard while
  they can hold it.
- A llama.cpp failure keeps the row and records why, as vLLM and TensorRT-LLM already did.
- Pruning levels test: the CSV column `share_of_model_removed` is now `share_of_decoder_weights_removed` (it counts
  only the decoder layers' weights), and the report column says so. A report with no comparison and no failures no
  longer says "see the failures below".

## 2026-10-13: audit follow-ups

Two audit fixes were only partly done in the entry below (found by a review after it merged); tests in
`tests/test_audit_fixes_20261013.py` fail on 697b3f1.

- SmoothQuant, QuaRot, SpinQuant and RPTQ, alone and in joint methods, now also leave layers at `baseline_bits` or
  above unrounded, as `rtn_act` and Stage 0 do (they used 16) (15). Their rows change only for a non-default
  `baseline_bits`. Method versions are not bumped, so default rows stay cached; a cache made with a non-default
  `baseline_bits` for these methods (or `rtn_act` joint rows) should be cleared by hand.
- vLLM and TensorRT-LLM report `<backend>_prefill_ms_std` and `<backend>_decode_ms_per_token_std` (spread across the
  timed repeats), so a timing difference within it is a tie in the verdicts, as for Hugging Face Transformers (4).
  Backend versions bumped (vLLM 5, TensorRT-LLM 4).

## 2026-10-13: ponytail audit fixes

From the ponytail audit of main 9a7ef8f (`/mnt/project-files/audits/ponytail-audit-20261013.md`); numbers are the
audit's. Tests in `tests/test_audit_fixes_20261013.py` (every test there fails on 9a7ef8f) cover 1-7, 9, 10, 13-15, 18,
19, the evaluation-data label, the sweep grid and the three review fixes; 11, 12, the other report-text changes and the figures are
untested small changes. Finding 8 (the
Stage 3 standard row for token-dropping methods) is left for the Stage 3 work.

- Pruning: the same-size plan at prune ratio 0 prunes nothing (it gave every layer ~1e-19, so low-rank cut every
  layer) (1). Structured pruning raises when a layer is planned above its feed-forward share instead of silently
  removing less than the size it reports (2). Pruning after a Stage 1 method keeps that method's limits (QTIP and
  ABQ-LLM fail with their reason, BiLLM's framework row is refused) (5); low-rank after a Stage 1 method is
  refused, since it re-rounds its factors with round-to-nearest and replaces the Stage 1 rounding (6).
- Serving: vLLM runs with prefix caching off and TensorRT-LLM with KV block reuse off, so prefill time is a real
  prefill, not a cache hit (backend versions bumped: those rows are measured again) (3). Both size the model for
  the latency prompt plus new tokens and refuse a prompt longer than an eval window (13). llama.cpp refuses a
  SentencePiece tokenizer without its `tokenizer.model` (perplexity was silently wrong) (14).
- Reports: one verdict (`StageReporter.verdict`) behind both "Key findings" and the plain summary, on the main
  metrics only; a timing difference within the rows' std is a tie (4). The master workbook shows NaN/inf as text
  (19). Evaluation-data text follows `eval.dataset`, the threshold-sweep rounding line follows
  `prune_sweep_quantize`, the score comparison counts the real layers, `handoff.md` lists the joint plan files (20).
- Stage 0: a run clears its old plan files first, so a resumed run never mixes plans of two runs (7). The FP16 cache
  key keeps vLLM/TensorRT-LLM settings at their defaults (changing a path re-measured everything). Entries made
  with the default settings stay valid; a cache made with `vllm_python` or `trtllm_python` set is recomputed once:
  the FP16 baseline and every row keyed on it (9). `kv_uniform_bits` must be one of `kv_bits_options` (a typo crashed Stage 0 after
  profiling) (10). With `act_plan="from_weights"` the joint plan gets that plan's real average activation bits
  (11). The score comparison uses `stage0.normalization` (12).
- Stage 1: one input-rounding function, so `rtn_act` alone and inside a joint method round by the same rule: layers below
  `baseline_bits`, as Stage 0 measured them (15; joint rows change only for a non-default `baseline_bits`).
  SqueezeLLM, SpQR, AQLM, QuIP#, PB-LLM and BiLLM no longer key their cache on `gptq_groupsize`, which they ignore
  (every row of those methods, and joint and series rows built on them, is recomputed once) (18).
- Figures: the stage roles and run ID match the current pipeline (17).
- Also fixed (found by the review of the audit): low-rank at prune ratio 1.0 removes the layer's weights, as the
  size prediction counts it (it kept rank 1); validation and held-out perplexity count as one measure in the
  verdicts, a win or loss only when both halves agree; the Stage 1-3 reports' evaluation-data text follows
  `eval.dataset`.
- Lean: deleted `scripts/colab_aqlm_debug.py`, `StageReporter.findings`, `BILLM_FIXED_BITS`, an unused import and a
  redundant check; the GPTQ and compressed-tensors exports share `save_tensors`; the vLLM and TensorRT-LLM backends
  share `run_worker`; the Stage 0 sweep grid refuses settings Stage 0 ignores (`smoothquant_alpha`).

## 2026-10-11: test sweep, 28 edge-case bugs fixed

Most are covered by `tests/test_edge_cases.py` (each of those tests failed before its fix); the serving workers'
one-decode-token case, the export group-size cache key, the threshold-name check and `baseline_bits` in the joint
profile are small untested changes. Cache keys of the default settings
are unchanged, so existing cached measurements stay valid.

- Reports: a framework row equal to the original now reads "ties with" (it said there was no original row); the %
  change keeps its sign when the reference is negative (Excel coloured such losses green); NaN no longer counts
  as a loss in the plain verdicts; NaN/inf cells show as text in Excel instead of blank; a NaN/inf measurement
  fails its target instead of reading "not measured".
- Config: `--set` values are cast to the field's type (`run.run_id=20261005` crashed `start_run`;
  `eval.max_windows=1e3` stayed a string). `act_bits_options`, `kv_bits_options`, `joint_w_options` and
  `joint_a_options` must be ascending without repeats (the planners treat the first as cheapest).
- Logging: a second run in the same process no longer also writes into the first run's `run.log`.
- Cache keys include `data.sources` for a dataset whose source differs from the default (changing it reused old
  profiles and FP16 numbers).
- Stage 0: the joint plan keeps the exact weight budget when layers differ in size; a bit budget below the
  cheapest option is refused (it was silently exceeded); rank agreement with a flat profile is NaN, not 1; a huge
  loss gives infinite perplexity instead of OverflowError; `predict_cost` refuses mismatched layer counts (layers
  were dropped from the size); prune and threshold levels that would share a row name are refused; the joint
  profile records `baseline_bits` for the curves.
- Rounding: the integer-zero grid now spans min(0, group min)..max(0, group max), as GPTQ does. Groups whose
  weights all have one sign were clipped at their largest value. Mixed-sign groups (nearly all real 128-weight
  groups) are unchanged; cached results were made with the old grid
  (start a fresh cache to be exact).
- Serving: `gptq_groupsize=-1` (one scale per row) exports instead of crashing; a 16-bit layer is marked
  unquantized for vLLM (`-:` dynamic rule) and left out of TensorRT-LLM's quantized layers; export errors (e.g. a
  6-bit plan) become VLLMError/TRTLLMError so the runner keeps the row; one decode token gives NaN decode time,
  not ZeroDivisionError; the cache key of weight methods without a
  group-size parameter (e.g. bitsandbytes) includes the export group size when vLLM/TensorRT-LLM is on; a diverged perplexity raises
  FloatingPointError instead of OverflowError.
- From review: long integer strings in `--set` stay exact; a scalar given for a bit-option list gets the clear
  error; a plan with no layer below 16 bits is refused by the GPTQ export; a framework row with nothing comparable
  says so instead of "ties with"; the sweep names a flat profile in its summary and shows NaN in its workbook.

## 2026-10-10: perplexity against WxAy per layer

- Stage 0 writes `joint_curves.csv` (each layer's calibration perplexity with only that layer rounded, at every
  weight x activation pair; `measured` is False for the copied unrounded baseline; `chosen` marks the joint plan's
  pair) and, with the new `figures` extra (matplotlib), `joint_curves/joint_curves_all_layers.png` (4 columns) and
  one PNG per layer. Points only, coloured by activation bits and shaped by weight bits. A failure here is logged
  and never stops Stage 0. `scripts/joint_curves.py` redraws them from a saved `stage_0` folder.

## 2026-10-10: joint plan refined with every layer rounded; SpinQuant on GPU

- The per-layer joint plan scored worse than the separate plans on Colab A100. Stage 0 now refines it in context
  (`stage0.joint_refine_rounds`, default 2): starting from the better fully rounded plan of the per-layer joint
  plan and the separate plans, each layer is re-measured at every pair with every other layer rounded, the plan is
  re-made, and a new plan is kept only if the fully rounded model's calibration perplexity drops. The row info
  records each start's and each round's perplexity.
- SpinQuant's Hadamard rotation is now built on the model's device (it stayed on the CPU, so GPTQ + SpinQuant
  failed on GPU; PR #58).

## 2026-10-10: joint weight x activation plan (Stage 0e)

- Stage 0 measures every decoder layer with its weights and its Linear inputs rounded together, at every pair in
  `joint_w_options` x `joint_a_options` (4, 8, 16 each), and picks one pair per layer with the smallest summed rise
  under both budgets at once (exact dynamic programming; average weight bits = the Stage 1 weight plan's,
  average activation bits = `act_avg_bits`, unless set). 16 bits is an option, so no layer is protected by a fixed
  rule. Files: `joint_weight_plan.json`, `joint_activation_plan.json`, `joint_profile.json`; Stage 0 row
  `joint_weights_activations` reports the pairs, the predicted rise of the joint and of the separate plans, and
  how far rounding both differs from the sum of rounding each alone.
- Stage 1 joint methods get a second framework row, `<name>_joint`, that follows the joint plan.

## 2026-10-10: weights and activations quantized together, per layer

- Stage 1 joint methods `<weight method>_with_<activation method>` quantize the weights at the Stage 0 weight plan's
  bits and the Linear inputs at the Stage 0 activation plan's bits, layer by layer, on one model (`JointPlan`), so
  layers get W8A8, W8A4, W4A8 or W4A4. Standard row: W4A8 on every layer; `..._w8a8` adds a W8A8-everywhere
  standard row. The activation method rewrites the model first (smoothing, rotation; nothing for `rtn_act`), then
  the weight method quantizes the rewritten weights. Row descriptions count the layers per combination.
- `scripts/colab_stage1_joint.py` runs the working weight techniques with `rtn_act`, plus SmoothQuant + RTN, QuaRot +
  GPTQ and SpinQuant + GPTQ, on Colab.

## 2026-10-10: diagrams redrawn from the code

- `docs/diagrams/`: 4 class and 4 sequence diagrams of the current code (Mermaid `.mmd` + `.svg` + `.png`),
  indexed in its README; joint `<weights>_with_<activations>` methods (JointPlan) drawn in. Replaces the stale per-stage pages (stage0-4.md); `search.md` (planned) stays.

## 2026-10-09: every Stage 1 technique from the forks

- New Stage 1 weight methods (simplified plain-torch ports of the forks, one commit and file cited above each):
  `omniquant`, `squeezellm`, `spqr`, `efficientqat`, `aqlm`, `quip`, `quipsharp`, `pbllm`, `billm`, `bitsandbytes`;
  activation methods `smoothquant`, `quarot`, `spinquant`, `rptq`. `qtip` and `abqllm` are registered as failed rows
  with the reason. All follow the plan's per-layer bits and refuse a plan that removes anything.
- BiLLM has no bit-width knob: its standard row runs, its plan-following row fails with that reason
  (`Method.fixed_bits`). Rows carry a `note` (simplified port, nominal size) and the metric `zero_weight_share`.
- `scripts/colab_stage1.py` runs Stage 0 + Stage 1 on Colab with no Drive and prints every number.

## 2026-10-08: the budget plan is removed; standard quantization against our method

- Removed: the Stage 0 budget plan and its nothing-removed benchmark (rows `allocation_same_size`,
  `allocation_same_size_no_prune`, `planner.budget_matched_plan`, `stage0.no_prune_compressed_bits`, the files
  `compression_plan_budget_matched.json` and `quant_plan_budget_matched.json`, the budget columns and sections of
  the Stage 0 report and handoff, the budget parts of the sweeps) and the Stage 1 plan key `quant_same_size`.
- The comparison is two-way: standard quantization (every weight at one bit length) against our method (bit lengths
  mixed per layer, the Stage 0 plan). Reports label the two rows "Standard method" and "Our method". The uniform
  6-bit and 8-bit activation rows stay as the standard side for activations. Stage 2 keeps its same-size pruning plan.
- Cached results on disk are untouched; budget rows in older run folders are just no longer read.

## 2026-10-08: TensorRT-LLM backend; vLLM backend fixes from the first real run (Colab A100)

- `eval.trtllm_python` (the Python with TensorRT-LLM; Linux only) adds TensorRT-LLM rows: the same GPTQ export,
  plus TensorRT-LLM's own `hf_quant_config.json` (W4A16_GPTQ / W8A16_GPTQ, MIXED_PRECISION for mixed plans; it does
  not read a GPTQ `quantization_config`), built into a TensorRT engine by `eval/trtllm_worker.py`: engine size,
  GPU memory loaded/after the run (read from the device; TensorRT does not allocate through torch), build time,
  prefill and decode time, perplexity of both halves from the context logits.
- vLLM: the export is symmetric (vLLM 0.31 rejects sym=False GPTQ), rounds against the FP16 scale it stores, the
  worker gets its venv's bin/ on PATH (ninja), and reserves a fixed 1 GB KV cache so peak memory measures the model.
- A plan a serving backend cannot load (vLLM has no 3-bit GPTQ kernel) keeps its HF/llama.cpp row and records the
  backend's own reason in the row info (`vllm_error`, `trtllm_error`).

## 2026-10-07: vLLM backend for Stage 1 weight rows

- `eval.vllm_python` (the Python with vLLM; Linux only, so run inside WSL2 or Colab) turns it on. Each weight row is
  saved as a GPTQ-format checkpoint at the plan's per-layer bits (`eval/gptq_export.py`: 2/3/4/8-bit packing, vLLM
  `dynamic` per-layer overrides) and measured by `eval/vllm_worker.py`: file size, weight memory, peak memory,
  prefill and decode time (warm-up and repeats as in HF), perplexity of both halves.
- Weights are re-rounded onto grids of their own when saved (as in llama.cpp), so `vllm_ppl_*` is comparable between
  rows, not with HF. Cache keys are unchanged when the backend is off.

## 2026-10-07: Stage 1 is quantization only, Stage 2 is pruning only

Mohammad's split: Stage 1 = quantization of weights and activations (path 0 > 1 > 4), Stage 2 = pruning
(0 > 2 > 4 on the FP16 model, 0 > 1 > 2 > 4 after a Stage 1 method), Stage 3 = KV cache (0 > 3 > 4).

- Stage 1 methods: `rtn`, `gptq`, `awq` (weights) and `rtn_act`, `smoothquant`, `quarot`, `rptq`, `spinquant`
  (activations, formerly Stage 2). They refuse a plan that removes anything (`require_bits_only`). GPTQ's fixed-mask
  (SparseGPT) path and RTN's magnitude pruning are gone; `rtn` no longer prunes.
- Stage 2 methods: `unstructured_prune` (Wanda), `structured_prune`, `low_rank`, in `stages/pruning.py`. They round
  nothing, except low-rank, whose factors are stored at the plan's bits. Run alone the weights stay FP16; run as
  `<pruning>_after_<quantization>` (config `stages.stage2_after`, e.g. `["gptq"]`) the Stage 1 method runs first.
- Stage 0 now also writes `quant_plan.json` (main plan, bits only), `quant_plan_budget_matched.json` (the
  nothing-removed budget plan, robust layers at 3 bits), `prune_plan.json` (the main plan's ratios, bits at 16) and
  `prune_plan_same_size.json` (as many numbers removed as the standard method, placed by sensitivity). The runner
  reads these instead of `compression_plan*.json`, so Stage 0 must be rerun.
- Standard versions: Stage 1 = uniform 4-bit, nothing removed; Stage 2 = every layer pruned at
  `prune_ratio_aggressive`, nothing rounded (4-bit uniform first when after Stage 1).
- llama.cpp rows are made for Stage 1 weight methods only. Cache versions bumped for every changed method.
- All earlier combined prune-plus-4-bit Stage 1 numbers (Wanda, structured, low-rank rows) belong to the series path
  and are stale; Stage 1 numbers for RTN, GPTQ and AWQ on the main and budget plans change (no pruning).

## 2026-10-07: Stage 1/4 audit fixes

- Stage 1-3 cache keys now include the settings a plan does not carry (`weight_zero_point`, `baseline_bits`,
  `act_group_size`, `kv_group_size`, `kv_module_names`) when they differ from the defaults. Before, changing one
  reused stale rows. Default-settings entries stay valid.
- Structured pruning and low-rank rows predict their size without the 1-bit-per-weight mask (whole channels or
  factors are removed). Only `predicted_weight_memory_gb` and `avg_bits_per_weight` of those rows change; no
  perplexity does. Cached rows are not stale (the prediction is recomputed every run).
- Wanda ranks weights with float32 scores (FP16 scores overflowed to inf when a channel's summed squares passed
  65504 squared). `unstructured_prune` is version 2, so its cached rows are measured again.
- llama-bench JSON is parsed up to its end, so text after it on stderr cannot break it.

## 2026-10-06: CUDA-graph latency works with transformers 5.18

- Timing passes an explicit all-ones attention mask. Without one, transformers 5.18 reads a GPU value to decide
  whether to skip the causal mask, which breaks CUDA graph capture; latency then fell back to eager (~40 ms/token
  on Colab) and each failed capture left 32 MiB allocated, so peak memory grew row by row.
- Every capture reuses one warm-up stream. A new stream per capture kept a new 32 MiB cuBLAS workspace alive, so peak
  memory grew 64 MiB per measured row.

## 2026-10-06: Latency timed with deterministic algorithms off

- The run's deterministic mode (`run.deterministic`) made HF decode 21% slower (4.12 -> 4.99 ms/token on the host
  PC) through slower kernels. `measure_latency` now turns it off while timing and restores it after; perplexity and
  everything else stay deterministic. The FP16 cache key has `latency: 2`, so cached rows are re-measured.

## 2026-10-06: Stage 4 llama.cpp backend (Stage 1 rows)

- With `eval.llamacpp_dir` and `eval.llamacpp_convert` set, each Stage 1 row is also saved as a GGUF file with every
  decoder layer at its planned bits (8 -> Q8_0, 4 -> Q4_K, ...; embeddings and output head F16, `--pure`), then
  measured: file size, llama-bench prefill/decode tok/s, llama-perplexity on both halves. llama.cpp re-rounds on its
  own grid and scores only the second half of each window, so its perplexity is compared between rows only.
- The saved model keeps its SentencePiece `tokenizer.model` (transformers 5 omits it; without it the GGUF had flat
  token scores and FP16 perplexity 17.8 instead of 8.1).
- Pruned weights are stored as zeros (GGUF has no sparse format), so pruning saves no size here.

## 2026-10-06: Latency measures the GPU, not Python

- Eager HF decode of a 1B model is mostly Python and kernel-launch overhead (32.8 ms/token on the host PC), so it
  said nothing about compression. `eval.latency_mode = "cuda_graph"` (default) runs prefill and each decode step
  from a fixed-size KV cache and replays them as CUDA graphs. If capture fails (e.g. a method's hook syncs with
  the CPU) it times eagerly; each raw latency record has `mode`. `"eager"` times the same loop without
  graphs. The setting is in the FP16 cache key, so cached baselines are re-measured.

## 2026-10-06: Stage 1 pruning and low-rank methods

- `unstructured_prune` (Wanda: |w| x input norm per output row), `structured_prune` (whole feed-forward channels,
  found by shape, lowest input norm x output-column norm first) and `low_rank` (activation-aware SVD, rank keeping
  1 - planned share of each layer's numbers). Each then rounds to the planned bits.
- Their standard version removes `prune_ratio_aggressive` from every layer (Method.prunes), since removing nothing
  would just be RTN. All six Stage 1 methods are the default.
- The plan's predicted size assumes a bitmask for removed numbers; structured and low-rank need none, so their
  real size is slightly smaller than predicted (Stage 4 measures it).

## 2026-10-06: Stage 1 GPTQ and AWQ

- `gptq` and `awq` (src/sdf/stages/weights.py), plain torch, layer by layer with inputs from the already-compressed
  layers before. Both use the round-to-nearest grid, follow each layer's planned bits, and keep the plan's pruned
  weights at 0 (GPTQ feeds their error back, as SparseGPT does). AWQ has no weight-clipping search yet.
- Default Stage 1 methods: rtn, gptq, awq.

## 2026-10-05: "Nothing removed" is a benchmark, not a plan for later stages

- The budget-size plan with nothing removed stays in the Stage 0 rows, sweeps and report, labelled
  "Benchmark: budget size, nothing removed". It is no longer saved as a plan file, returned in `Stage0Result`,
  listed in handoff.md or run as a Stage 1 framework row. Handoff and Stage 1 call the 0.768 GB plan the budget plan.

## 2026-10-05: Realistic size for pruned weights; comparable fragile-parts score; code version in reports

- Pruning is unstructured, so deleted weights only save space in a sparse file. `stage0.sparse_storage`:
  "bitmask" (default, +1 bit per weight of a partly pruned layer), "dense" (no saving), "free" (old ideal).
  The fair pruning test now pairs versions by share removed (a bitmask makes its plan slightly smaller).
- The fragile-parts score (exposure) of a non-removal plan is scored against the layer-removal ranks: against
  its own ranks every measure protecting k layers scores the same (Hessian 0.58 = removal 0.58 by construction).
- `environment` in every results.json records `git_commit` and `git_dirty`.

## 2026-10-05: Weight rounding keeps 0 on the grid (integer zero point)

- `round_to_nearest` offset its grid by the group minimum (a float), so 0 was usually not one of the 16 values.
  Magnitude pruning keeps pruned weights at exactly 0, so pruned layers got a 17th value for free: the standard
  method scored 10.87 with 10% removed against 11.32 with nothing removed. Weights now use an integer zero point
  (`stage0.weight_zero_point = "int"`, GPTQ/AWQ format); `"float"` reproduces the old numbers. Activations and the
  KV cache keep the float grid (unchanged). Cache keys of pruning/threshold measurements and the one-layer
  compression profile include the grid, and the RTN method version is 2, so nothing old is reused by mistake.

## 2026-10-05: "Budget plan" instead of "same size"

- The two plans that protect as many layers as fit in the standard method's memory were called "same size", but
  they come out 1-1.4% smaller (0.768 / 0.766 GB against 0.777 GB). Report text, labels and figures 3 and 6 now
  call them the budget plan and the budget plan with nothing removed. Internal keys (`allocation_same_size`, ...)
  are unchanged, so caches and saved plans still load. The fair pruning test keeps "same size": there the sizes match.

## 2026-10-04: Protection threshold sweep

- `MODE = "threshold_sweep"` (src/sdf/stage0/threshold_sweep.py): builds the Stage 0 plan at every threshold in
  `THRESHOLD_SWEEP` (0.1 to 0.9) and guard size in `GUARD_SWEEP` (0, 3, 5, 8), applies it to the real weights and
  measures perplexity, next to the standard method and the same-size plans. The report marks the best trade-offs
  (no other version both smaller and more accurate). Measurements are cached per plan and shared with
  `prune_sweep` (`prune_sweep.make_evaluator`).

## 2026-10-04: Report text for every sensitivity score

- Stage 0's report explained only `grad_x_weight`, `layer_removal` and `layer_quant`, so a single run with
  `fisher`, `taylor_ema`, `hessian` or `movement` crashed with a `KeyError` after the profile was computed. All
  seven scores now have a plain-language description, and a test checks every score in `SCORES` has one.

## 2026-10-04: Downstream tasks and the master report

- `eval/downstream.py`: multiple-choice accuracy through lm-evaluation-harness (`pip install -e .[downstream]`),
  set with `DOWNSTREAM_TASKS` / `DOWNSTREAM_LIMIT` in `main.py` (off by default). Stages 1-3 measure it on every
  row; the FP16 result is cached on its own key, and the FP16 perplexity cache key no longer includes the
  downstream settings, so existing cache entries stay valid.
- `reporting/master.py`: `all_stages_comparison.xlsx` and `master_report.md` from every `stage_<N>/results.json`,
  written at the end of `MODE = "stages"` (or any time with `write_master(run_dir)`).

## 2026-10-04: Groundwork for Stages 1-3

- **`MODE = "stages"`** (`src/sdf/stages/`): one runner for Stages 1-3. It loads the Stage 0 plans
  (`STAGE0_DIR`, or runs Stage 0 first), then for each method builds the uncompressed row (Stage 0's cache entry),
  the standard method on the uniform plan (cached by config) and the framework on each Stage 0 plan, every one from
  a fresh FP16 model, and reports to `stage_<N>/` through `StageReporter`.
- `stages/methods.py` lists every method in the spec. Round-to-nearest baselines work now (`rtn`, `rtn_act`,
  `rtn_kv`); the rest are failed rows saying "not implemented yet" until they are written.
- Search space: `smoothquant_alpha` (0-1, default 0.5) and `quarot_k_bits` (2/3/4/8, default 4). The Stage 0 sweep's
  default grid only covers the Stage 0/1 parameters.
- `docs/stage-methods-feasibility.md`: which library each method needs and whether it runs on the Windows PC and
  Colab.

## 2026-10-04: Fair pruning test

- `prune_sweep` adds the fair comparison (`stage0.prune_sweep_same_size`, on by default; rows `same_NNN`): every
  layer at `uniform_bits` and the same number of weights removed as the standard method, so the same predicted
  size and share removed, but placed by sensitivity (`planner.same_size_pruning_plan`: robust layers lose more,
  fragile ones less, guarded ones nothing). Its own report section, "Fair test: same size, same share removed".
  Where the guarded layers can't absorb the pruning (high levels) the plan stays bigger and the report says so.

## 2026-10-04: Run on Google Colab

- `notebooks/colab_run.ipynb`: mounts Drive, clones or pulls `main`, installs, runs the tests, then runs `main.py`
  with `MODE` and the output / cache folders set in the notebook (`MyDrive/thesis_compression/results` and
  `cache`). No code changes were needed: every path was already a setting.

## 2026-10-03: Hessian score fixed

- `hessian` came out negative for some TinyLlama layers (one probe per passage is mostly noise). It now uses
  `stage0.hessian_probes` (default 8) probes per passage, averages the curvature estimate per weight over all
  passages and probes, and drops the negative part before weighting by w². Scores are never negative. Costs
  1 + `hessian_probes` backward passes per passage two CPU copies of the layer weights (originals and the
  curvature sum) and one extra gradient on the device.

## 2026-10-03: Four more sensitivity scores

- `stage0.score` adds `fisher`, `taylor_ema`, `hessian` and `movement` (`profile_sensitivity(method=...)`; settings
  `taylor_ema_beta` 0.9, `movement_lr` 1e-4, `hessian_eps` 1e-3). Hessian uses Hutchinson's diagonal estimate with
  a finite-difference Hessian-vector product, so it needs no second-order graph; movement runs one plain SGD step
  per calibration passage. Both restore the weights from a CPU copy and keep per-weight state on the CPU, so GPU
  memory is about one backward pass. Layer removal stays the default.
- `compare_scores` runs all seven, survives a failing score, and opens its report with how well each score
  matches the two measured ones (agreement, top 5 vs the never-pruned set, time, peak memory).

## 2026-10-03: Pruning levels, measured

- **`MODE = "prune_sweep"`** (`stage0/prune_sweep.py`): prunes the real weights at every share in
  `PRUNE_SWEEP_RATIOS` (default 10% to 100% in 10% steps, `stage0.prune_sweep_ratios`) and measures perplexity on
  both test halves, standard method (every layer 4-bit, pruned) vs framework (Stage 0 plan, protected and guarded
  layers not pruned). Magnitude pruning per output row plus round-to-nearest at the plan's bits
  (`PRUNE_SWEEP_QUANTIZE = False` for pruning only), simulated on FP16 weights; each level is cached. Outputs in
  `stage_0_prune_sweep/` with a "Pruning curve" table. Before, pruning was only predicted, at the one fixed
  ratio (0.3).
- Pruning ratio 1.0 is now allowed in plans (the layer keeps no weights and passes its input on).
- `predict_cost` counts scale factors only for kept weights, consistent with its "pruned weights are free"
  assumption; plans that prune are predicted slightly smaller (about 0.5% at 30% pruning on TinyLlama).
- `load_fp16` shared by Stage 0 and the new study; `StageReporter(subdir=...)` for side studies.

## 2026-10-01: Annotated figures

- `docs/figures/`: six annotated figures (SVG + PNG) for the thesis: pipeline overview, how Stage 0 builds its
  plans, the weight, activation and KV cache plans with the final run's numbers, and what Stage 0 hands to each
  later stage. Drawn by `docs/figures/make_figures.py` (needs matplotlib, not a package dependency).

## 2026-10-01: Stage 0 finalised: measured activation plan, hand-off report, honest verdicts

- **Activation plan is measured** (`ACT_PLAN = "measured"`, default): each layer's Linear inputs are rounded to
  4 and 8 bits on their own (64 passages) and the perplexity rise recorded; an average of 6 bits per layer goes
  where the rise is largest. The weight-derived plan stays as `ACT_PLAN = "from_weights"` and as the comparison
  row `activations_from_weights/framework`. New metric `predicted_act_ppl_rise`, `activation_profile.json`,
  per-layer column "Activation damage at fewest bits". `allocate_bits` is now shared by the KV and activation
  plans.
- **KV cache measured on 64 passages by default** (`KV_CALIB_SAMPLES`): the 2026-09-30 rerun showed 16 passages
  were too noisy for the bit choice. The KV bit plan keeps the standard method's 4-bit average (same size); the
  hand-off report says when the bit choice predicts no gain.
- **`handoff.md`**: what Stage 0 hands to Stages 1-4 and the search, layer by layer, with predicted costs, the
  Original model section and every column explained. Linked from report.md and returned in `outputs`.
- **Report fixes:** build time no longer turns every framework row into a "trade-off" (stated as a one-off
  cost instead); with no deployment target set, rows say "no targets set" instead of "yes"; `kv_budget_gb` is
  checked against the predicted KV memory; the FP16 row is called "Original model (uncompressed)".
- Specs v0.4 (S0-15, S0-17, REP-09; SDD §5.9, §5.10) and the Stage 0 diagram updated.

## 2026-09-30: Stage 0 hands everything later stages need

- **Pruning guard.** `stage0.guard_top_k` (default 5, `GUARD_TOP_K` in main.py): the layers whose removal raises
  perplexity most are never pruned in any framework plan (threshold, same-size, same-size without pruning),
  whatever score picks the bits. On TinyLlama these are 0, 2, 7, 21, 1. When `stage0.score` is not
  `layer_removal`, a removal profile is measured and cached for the guard. Before, the critical layers were only
  safe because the default threshold happened to protect them; a higher threshold, the size-matched plans or
  the gradient score could prune layer 0, which alone takes perplexity from 14 to ~1190. The sweep guards when
  its profiles are layer-removal ones.
- **Activation plan for Stage 2** (`stage0/activation.py`): protected and guarded layers keep
  `act_protected_bits` (8) activations, the rest `act_compressed_bits` (4); original = `act_uniform_bits` (8).
  Derived from the weight plan, not measured; Stage 2 should validate it. New rows `activations/original` and
  `activations/framework` (`avg_activation_bits`), `activation_plan.json`, report section and per-layer columns
  "Never pruned" and "Activation bits".
- **Hand-off.** `Stage0Result` also returns the same-size plans, the activation plan and both KV plans;
  `CompressionPlan.load`, `KVPlan.load` and `ActivationPlan.load` read the saved JSON back.
- Specs v0.3 (S0-14 to S0-16, SDD §5.4, §5.9, §7.3) and the Stage 0 class diagram updated.

## 2026-09-30: Every report table explains its columns

- `report.md` lists "What each column means" under the plain results table, the per-layer table and the
  technical results table. Metric columns use the metric registry's plain explanation (moved there from Key
  terms); per-layer columns take an optional third element, the meaning, in `plain_layer_columns`.
- Stage 0 explains all 12 per-layer columns; the raw-score explanation follows the sensitivity score used.

## 2026-09-30: Every report describes the original model

- New `utils/model_info.py`: `describe_model(config, name, num_parameters)` reads the original model's
  parameters from its Hugging Face config (architecture, parameter count, layers, hidden size, feed-forward
  size, attention heads, key/value heads, numbers per head, vocabulary, maximum context, published number
  format, bits per number, size at 16 bits, shared word tables). Fields a model's config lacks are left out,
  so it works for any model.
- `StageReporter(original_model=...)`: an "Original model" section near the top of `report.md` (value and a
  plain-language meaning for each), `original_model.*` rows in the Config sheet, and `original_model` in
  `results.json`. The sweep and score-comparison reports show the same section.
- Stage 0 reads only the config file when everything comes from cache; the parameter count then comes from
  the sensitivity profile. The glossary no longer hardcodes TinyLlama's size.

## 2026-09-30: Specs describe layer removal and the KV cache plan

- `docs/specs/` version 0.2: S0-01 names layer removal as the default sensitivity score, with one-layer
  compression and gradient x weight selectable (SDD §5.3); new S0-10 to S0-13 and SDD §5.8 for the Stage 0 KV
  cache plan; MET-06, S3-02 and SDD §8 follow that plan; traceability updated.

## 2026-09-29: System and subsystem specifications

- **`docs/specs/`**: a System Requirements Specification following ISO/IEC/IEEE 29148:2018 (every requirement
  from the thesis spec with an ID, priority, status and verification method) and a Subsystem Design
  Description following IEEE 1016-2009 (shared core, Stages 0 to 4 and the search layer, each described from
  the same viewpoints, with a traceability table from requirement to design, code and test). Stage 0 and the
  shared core are described from the code; the rest is marked planned, consistent with `docs/diagrams/`.

## 2026-09-30: Diagrams describe layer removal and the KV cache plan

- `docs/diagrams/stage0.md`: layer removal is the default sensitivity score (grad x weight and one-layer
  compression still drawn as options); new KV cache classes (`KVProfile`, `KVPlan`, `KVLayerPlan`, `KVCost`,
  `kv_cache` module) and the three KV cache rows in the sequence diagram. `stage3.md` now reads the Stage 0 KV
  cache plan.

## 2026-09-29: Class and sequence diagrams

- **`docs/diagrams/`**: a class diagram and a sequence diagram for the project as a whole and for each stage
  and the search layer, in Mermaid so GitHub draws them. Stage 0 and the shared parts are drawn from the
  code; Stages 1 to 4 and the search are drawn from the design and labelled `<<planned>>`.

## 2026-09-30: Stage 0 plans the KV cache

- New `stage0/kv_cache.py`, on by default (`KV_CACHE` in main.py, section 4). Per decoder layer it plans:
  1. key bits and 2. value bits, chosen separately: each layer's keys (then values) alone are rounded to each
     of `KV_BITS_OPTIONS` and the calibration perplexity rise is measured. A greedy allocation spends the
     same average bits as the uniform `KV_UNIFORM_BITS` cache where the measured damage is largest. Keys are
     rounded per channel and values per token (KIVI / KVQuant), keys before RoPE.
  3. a token budget (H2O / SnapKV style eviction): the fewest of `KV_KEEP_RATIOS` whose top tokens still
     receive `KV_ATTENTION_COVERAGE` of the layer's attention (oracle top-k, measured with eager attention).
  4. predicted cache memory at `KV_CONTEXT_LEN` tokens x `KV_BATCH_SIZE`, with scale overhead per group.
- Report rows `kv_cache/original` (uniform bits, no eviction), `kv_cache/framework` (bits + budget) and
  `kv_cache_bits_only/framework` (bits alone, to separate the two effects); the FP16 row gets the 16-bit cache
  size. New metrics, per-layer columns, a "KV cache plan" section and plain-language text. Plans and the KV
  profile are saved as JSON; the profile is cached.
- Key/value layers are found by name (`KV_MODULE_NAMES`), so any model with separate key/value projections
  works; fused-QKV models need their own names.

## 2026-09-30: Default sensitivity score is layer removal

- `stage0.score` defaults to `layer_removal`, Mohammad's choice: a layer's sensitivity is the rise in
  calibration perplexity when it is skipped. `layer_quant` and `grad_x_weight` stay selectable.

## 2026-09-29: Default sensitivity score is now single-layer compression

- `stage0.score` defaults to `layer_quant`. On TinyLlama (WikiText-2, 64 passages) gradient × weight ranked
  layer 0 least sensitive, so the plan compressed and pruned it, yet skipping layer 0 raises perplexity from
  14.1 to about 1,190. Its ranking also did not agree with the measured one-layer compression damage (rank
  agreement −0.01). `layer_quant` measures the damage protection prevents directly. Old runs are unaffected:
  the score is part of the profile cache key.

## 2026-09-29: Layer-removal and one-layer-compression sensitivity scores

- **`stage0.score`** (`SENSITIVITY_SCORE` in main.py) chooses how sensitivity is measured:
  - `grad_x_weight` (default): unchanged.
  - `layer_removal`: skip one decoder layer at a time and measure the rise in perplexity on the
    calibration text.
  - `layer_quant`: compress only that layer (round-to-nearest at `compressed_bits` and the candidate's
    group size) and measure the rise in perplexity. This is closest to what the plan does.

  The ablation scores need one forward pass over the calibration text per layer. They work on any decoder
  stack through hooks and temporary weight rounding, and the model is restored afterwards. The score is
  part of the profile cache key.
- **`MODE = "compare_scores"`** in main.py profiles all three ways on the same text and writes
  `stage_0_scores/report.md`, `scores.xlsx` and `results.json`: rank agreement between the ways, the layers
  each would protect, and a per-layer table.
- The report's "How this stage works" section describes whichever score was used.

## 2026-09-29: Report explains the sensitivity-driven method step by step

- The plain part's background section is now "How this stage works". It explains in four numbered steps
  how the score is measured (weight size times gradient, summed per layer), how it is ranked onto a 0-1
  scale, what the threshold does, and how bits and pruning follow, including the two same-size versions.

## 2026-09-29: Local output folders

- Results and the cache now go to `thesis_compression/results` and `thesis_compression/cache` under the
  folder you run from, in `main.py` and `configs/tinyllama.yaml`, instead of Google Drive paths. The
  folder is git-ignored. The hardware profile in the config is now `RTX-5070-Ti`.

## 2026-09-29: Tunable parameters in main.py

- **`main.py`** lists every tunable parameter at the top of one file: the run settings, the five
  search-space parameters (with their allowed values in comments), the fixed Stage 0 settings and the sweep
  values. Edit it and run `python main.py`. `MODE` picks a single run or a sweep. Values outside the search
  space are rejected before anything runs. Settings not listed keep their defaults from `config.py`.

## 2026-09-29: Stage 0 sweep

- **`sdf-stage0-sweep`** tries several values of every search-space setting and writes one plain-language
  report plus `sweep.xlsx` (every plan) and `results.json`. It covers the threshold, the prune ratio, the
  group size, and the calibration text and amount. For each combination it shows the threshold plan, the
  same-size plan and the no-removal plan against the standard method, how much the calibration setting
  changes the layer ranking (rank agreement), and which plans are both smaller and safer than the standard
  method.
- A calibration setting that fails (for example, a dataset that can't be downloaded) is recorded in the
  report instead of stopping the sweep.
- `--profile` plans from saved sensitivity profiles, so the planning part runs without a GPU.
- Profile loading moved into `stage0.run.load_profile`, shared by the single run and the sweep.

## 2026-09-28: Report gaps from the first local run

- **Every core metric in both report tables.** Held-out perplexity, peak GPU memory and per-token decode
  latency were only in results.json; they are now columns in the plain and technical tables, along with
  sparsity.
- **Pruning caveat for the same-size plan.** On TinyLlama the same-size plan protects 5 layers and prunes
  the other 17 by 30%, while uniform prunes nothing, so the size match is bought with pruning. Sensitivity
  exposure scores 4 bits with 30% pruned as 2.8 effective bits, which likely understates the damage. The
  report now says so in both parts.
- **Same-size plan without pruning** (`allocation_same_size_no_prune`). Robust layers drop to
  `stage0.no_prune_compressed_bits` (3) instead of being pruned, and as many top layers as fit are
  protected at 8 bits within the uniform plan's memory. This isolates the effect of the sensitivity
  guidance itself. Its plan is saved as `compression_plan_budget_matched_no_prune.json`.
- **Per-layer table** now shows the share removed for the same-size plan and the bits per layer of the
  no-pruning plan.
- **Prefill latency** is a column in both report tables, next to decode latency.
- **Unused budget explained.** The same-size plans keep the protected set to the top-k layers by
  sensitivity, so packing stops at the first layer that doesn't fit (0.011 GB left on TinyLlama). The
  report now states the leftover and what the next layer would cost. Every TinyLlama decoder layer is the
  same size, so skipping ahead would not fit another layer either.
- **Unit test for the no-pruning plan**: it fits the uniform budget, removes nothing, keeps robust layers
  at 3 bits and protects the 4 most sensitive layers on TinyLlama shapes.

## 2026-09-28: Leaner code after the ponytail audit

About 220 lines were removed with no change to what a run produces:

- **One copy of every default.** The defaults live only in `config.py`. `configs/tinyllama.yaml` now holds
  only the Colab paths and the hardware profile, so it shows at a glance what a run changes.
- **Candidates are plain dicts.** The `Candidate` class, the unused sampling helpers and `stage0.score`
  (a setting with one allowed value) are gone.
- **Sensitivity is normalised once**, when planning. The saved profile holds only the raw scores, and the
  planner functions take the normalised score list.
- **Smaller helpers.** The cache has one `get_or_compute` method, the number and label helpers exist once,
  `load_texts(sources, dataset, split)` replaces the loader factory, and the package `__init__` files no
  longer re-export names (import from the modules directly).

## 2026-09-28: Report layout matches the hand-written version

- **New plain-part order.** The plain part of report.md now follows the layout Mohammad liked:
  1. a one-paragraph summary;
  2. key terms, covering the versions compared, every measure and stage-specific words;
  3. one results table with every version, units and "lower is better" in the headers;
  4. a sensitivity table for every layer;
  5. findings.

  The Stage 0 findings cover the same-size plan, the 16-bit size floor and outlier layers. The background
  explanation comes after the findings, and the technical detail follows unchanged.

## 2026-09-28: Fair same-size comparison, size floor, outlier flag

- **Same-size framework plan.** The threshold plan (0.947 GB predicted on TinyLlama) is about 22% bigger
  than uniform 4-bit (0.777 GB), so comparing its quality against uniform isn't fair. Stage 0 now adds a
  second framework row, `allocation_same_size/framework`, compared against the same uniform plan. It
  protects as many of the most sensitive layers as fit in the uniform plan's predicted memory, which works
  out to 5 of 22 layers at 0.775 GB. Its plan is saved as `compression_plan_budget_matched.json`.
- **Size floor reported.** Embeddings, the LM head and norms stay at 16 bits in every plan. The report now
  states this floor (0.26 GB on TinyLlama) in both the plain and the technical part.
- **Outlier layers flagged.** A layer whose raw sensitivity is far from the rest is flagged in the report
  and in the Per-layer sheet, together with what the plan does to it. Outliers are found by a robust
  z-score (median / MAD, above 3.5), which catches TinyLlama's layer 0. They are flagged, not
  auto-protected.

## 2026-09-28: Plain-language reports

- **Two-part report.md.** Every report now opens with a part written for readers with no AI background:
  - the short version (did the framework win);
  - what the stage does;
  - what was compared;
  - what each number means, with the three versions side by side and a "which is better/worse" reading;
  - why the framework won or lost;
  - a glossary.

  The technical tables follow unchanged under "Technical details".
- **Built into the shared reporter.** This lives in `StageReporter`, so every stage gets it. Each metric in
  the registry carries a plain name and explanation, and a stage supplies its own intro, "why" and glossary
  terms.

## 2026-09-28: Fixes from the first real run (RTX 5070 Ti)

- **Degenerate plan fixed.** Sensitivity scores are now rank-normalised by default
  (`stage0.normalization: rank`). With min-max, TinyLlama's layer 0 (raw score 2705 against 6596–8647 for
  the other layers) squeezed layers 1–21 into 0.66–1.0, so threshold 0.5 protected 21 of 22 layers. That
  plan averaged 8.97 bits against 5.65 for uniform. With rank, `sensitive_threshold` *t* protects about the
  top (1 − *t*) share of layers, whatever the score distribution. Normalisation is applied when planning,
  so cached profiles are reused.
- **Windows encoding fixed.** Every text file is now written and read as UTF-8: reports, JSON, config and
  `run.log`. Before this, Windows wrote cp1252, which garbled "—" and "Δ" in report.md.

## 2026-09-28: Dataset IDs moved to the config

- **Fix.** WikiText-2 now loads from `Salesforce/wikitext`. Current `datasets` versions no longer resolve
  the bare `wikitext` id, so Stage 0 failed before profiling.
- **Config.** All Hub dataset ids and splits now live in the config (`data.sources`), so a moved dataset is
  a config change rather than a code change.

## 2026-09-28: Stage 0 aligned with the thesis spec

- **Config.** Added a single `FrameworkConfig` (`configs/tinyllama.yaml` plus `--set key=value` overrides).
  Stage logic no longer hardcodes anything: bit-widths, the uniform baseline, eval settings and output
  paths all come from the config.
- **Search space.** It now uses the spec's parameter names and is defined in one place
  (`search_space.py`), so adding a parameter is one line. The scope is Stage 0 only, so it holds
  `sensitive_threshold`, `prune_ratio_aggressive`, `calib_dataset` and `calib_samples`, plus
  `gptq_groupsize`, which Stage 0 uses to predict memory. The Stage 2 and 3 parameters are added with
  those stages.
- **Reporting.** Added `StageReporter`, the shared reporting module. Every stage writes `report.md`,
  `stage_<N>_comparison.xlsx` and `results.json` with one schema. The reporter:
  - computes deltas against FP16 and against the original method;
  - highlights framework wins and losses with conditional formatting;
  - checks the DeploymentRequirement for each row and records the shortfall;
  - records a failed method as a failed row instead of stopping the stage.
- **FP16 baseline.** Added measurement of perplexity on the validation and held-out halves of the
  WikiText-2 test split, model size, peak GPU memory, and prefill and decode latency (warmup runs, repeated
  timings, mean and std). It is computed once per model, eval settings and hardware, then cached.
- **Stage 0 comparison.** Stage 0 now compares FP16, a uniform allocation (the original method) and the
  sensitivity plan. It reports predicted weight memory, bits per weight, sparsity, sensitivity exposure,
  build cost and per-layer allocations.
- **Profile caching.** The sensitivity profile is cached by model and calibration settings. Before this, it
  would have been recomputed for every trial, even though the searched Stage 0 thresholds don't affect it.
- **Seeding and logging.** Seeding covers python, numpy, torch and cuda, with a deterministic-algorithms
  flag, and the seed is logged. Logging is timestamped and goes to stderr and to `<run_dir>/run.log`.
- **Crash-safe output.** Writes are atomic and `results.json` is rewritten after every row, so a Colab
  disconnect keeps completed work.
- **Trial log removed.** It belongs to the search layer, which is out of scope for now.

## 2026-09-28: Initial Stage 0

- Project skeleton, gradient × weight sensitivity profile and threshold planner.
