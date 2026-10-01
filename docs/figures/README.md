# Figures

Annotated figures for the thesis, each as SVG (for the document, scales cleanly) and PNG. They use the real
numbers from the final Stage 0 run (`20261001-184158_781ac2`, TinyLlama-1.1B-Chat, RTX 5070 Ti). Stages 1-4
and the search are drawn as planned. Redraw with `pip install matplotlib && python docs/figures/make_figures.py`.

| Figure | Shows |
|---|---|
| [fig1_pipeline](fig1_pipeline.svg) | The whole framework: Stage 0 measures, Stages 1-3 compress, Stage 4 evaluates, the search tunes it all. |
| [fig2_stage0_method](fig2_stage0_method.svg) | How Stage 0 tests one layer at a time and turns the damage into a plan for weights, activations and the KV cache. |
| [fig3_weight_plan](fig3_weight_plan.svg) | Each layer's sensitivity, which layers are protected or never pruned, and the predicted size of every version. |
| [fig4_activation_plan](fig4_activation_plan.svg) | Damage from rounding each layer's inputs to 4 bits, and which layers get 8 bits in the Stage 2 plan. |
| [fig5_kv_plan](fig5_kv_plan.svg) | How much of the past each layer keeps in the KV cache, and the predicted cache memory (13.0 MB to 3.2 MB). |
| [fig6_handoff](fig6_handoff.svg) | What Stage 0 hands to Stages 1-4 and the search, with the key numbers. |

## In plain words

A language model is a stack of layers. Some layers are fragile: storing their numbers roughly breaks the
model. Others barely matter. Stage 0 finds out which is which by changing one layer at a time and measuring how
much worse the model gets at predicting the next word (its *perplexity* rises). The later stages then compress
the robust layers hard and leave the fragile ones alone. Each figure has notes on it explaining what to look at.

The numbers are copied into `make_figures.py` from that run's `handoff.md`; update them there after a new run.
