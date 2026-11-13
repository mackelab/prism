# Training Python API

The supported training interface is the `prism train` command. The functions
below are reusable when constructing simulators, models, dataloaders, or
checkpoint bundles in Python. The functions and state classes in
`prism.train.train_script` implement the CLI loop and are not a supported
programmatic trainer or callback interface.

See the [training guide](../guides/training.md) for configuration, output, resume,
and evaluation examples.

## Builders

::: prism.train.build_simulator.build_simulator
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.train.build_model.build_model
    options:
      show_root_heading: true
      show_root_full_path: false

`build_simulator(cfg)` returns the configured simulator class (a dMRI
`MultiCompartment` or a symbolic `AdditiveModel`) and its training-data
generators, one per dMRI acquisition entry or a single one for symbolic models. `build_model(cfg, sim_type)`
builds an initialized model for that simulator class.

## Simulation data

::: prism.train.dataset.SimulationDataset
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - reset
        - close
        - set_data
        - get_stats

::: prism.train.dataset.instantiate_dataloader
    options:
      show_root_heading: true
      show_root_full_path: false

## Checkpoint loading

::: prism.train.utils.load_cfg
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.train.utils.load_checkpoint
    options:
      show_root_heading: true
      show_root_full_path: false

`load_checkpoint` accepts a local run or a Hugging Face model subfolder. It
returns `(checkpoint, model, simulator_class)`. The model is rebuilt in eval
mode, but the returned checkpoint parameters are not applied automatically.

## Checkpoint bundles

::: prism.train.utils.bundle_checkpoint
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.train.utils.upload_checkpoint_to_hub
    options:
      show_root_heading: true
      show_root_full_path: false

::: prism.train.utils.download_checkpoint_from_hub
    options:
      show_root_heading: true
      show_root_full_path: false
