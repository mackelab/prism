# Prediction Python API

Use [`prism predict`](../cli/predict.md) for the complete folder-based prediction
workflow. For Python code that manages model inputs and inference directly, the
package root exposes `prism.load_pretrained`, `prism.list_pretrained_models`, and
`prism.PretrainedModel`.

```python
import prism

available = prism.list_pretrained_models()
bundle = prism.load_pretrained("msb3s_2_4_6_128", params_name="params_ema")

model = bundle.model
simulator = bundle.simulator
config = bundle.config
step = bundle.step
params_name = bundle.params_name
```

By default, these functions use the public `manugloeck/dmri-pretrained`
repository. Model files are downloaded and cached by `huggingface_hub`.
`load_pretrained` accepts `revision`, `cache_dir`, `local_files_only`, and a
network-forward `precision`. CPU fp16 and bf16 requests fall back to fp32.

`load_pretrained` applies `params_ema` (EMA weights) by default. Set
`params_name="params"` to select raw weights. These are the only accepted
values; missing requested weights raise `ValueError`. The returned model is
in evaluation mode, and `bundle.params_name` and `bundle.step` identify the
loaded weights. The `simulator` field is the reconstructed `MultiCompartment`
class.

Pretrained models are acquisition-specific. Verify compatibility with the input
acquisition before using their output. PRISM has not been clinically validated.

## Published checkpoint selections

The public repository at revision
[`017561f`](https://huggingface.co/manugloeck/dmri-pretrained/tree/017561ffbf77b109c9fe6c3d9c7d48831780ddc3)
contains these model bundles and saved steps:

| Model name | Best step | Regular saved step |
| --- | ---: | ---: |
| `b3s_2_4_6_128` | 1,800,000 | None |
| `b3s_2_4_6_64` | 2,640,000 | None |
| `msb3s_2_4_6_128` | 1,520,000 | 2,280,000 |

`load_pretrained` defaults to `which="best"`. `which="latest"` selects a regular
checkpoint, so the two B3S bundles above require `best` at this revision. The symbolic
loader also accepts `which="figure"`, the checkpoint a paper figure used (for example
`symbolic50_illustrative`). Pin
the full revision and record `bundle.step` and `bundle.params_name` when saving
an evaluation. `list_pretrained_models()` queries the available model names;
future revisions may contain different selections.

These bundles provide pretrained dMRI inference. Their identity with the
historical Figure 6 weights has not been established. The paper's frozen
scripts name `updated5_b3s_2_4_6_128`; its source preset uses PPP, whereas the
public `b3s_2_4_6_128` bundle uses PP. The public multishell bundle also has a
different simulator from that source preset. See the
[paper workflow map](../paper_experiments.md#checkpoints-and-saved-source-results)
for the available source records.

## Symbolic checkpoints (safetensors)

Symbolic checkpoints are stored as bare EMA weights in safetensors files with a
`config.yaml` next to them, instead of Orbax run directories. Load one with

```python
from prism.symbolic.pretrained import load_pretrained_symbolic

bundle = load_pretrained_symbolic("symbolic_sbmi_10k", which="best")
model, simulator, step = bundle.model, bundle.simulator, bundle.step
```

The loader rebuilds the model from `<preset>/config.yaml` and raises
`ValueError` before changing anything when the file's tensor names or shapes
differ from the model's. Use `path=` for a local directory laid out as
`config.yaml` and `<best|latest>/params_ema.safetensors`. The default
repository is `manugloeck/prism-symbolic-pretrained`.

## API

::: prism.hub
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - PretrainedModel
        - load_pretrained
        - list_pretrained_models
