"""Set explicit diffusion bounds without changing network parameters."""

from dataclasses import replace

from prism.nn.model import PrismModel


def configure_model(model):
    decoder = model.inference_decoder
    if decoder.loss_type != "v":
        raise ValueError("The paper experiment requires v-prediction")
    decoder.train_cfg = replace(decoder.train_cfg, t_min=1e-4, t_max=80.0)
    decoder.schedule = replace(decoder.schedule, t_min=1e-4, t_max=80.0)
    decoder.solver_cfg = replace(
        decoder.solver_cfg, schedule=decoder.schedule, num_steps=64
    )
    return model


class PaperDMRIModel(PrismModel):
    def _initialize_modules(self, rngs):
        super()._initialize_modules(rngs)
        configure_model(self)


def diffusion_settings(model):
    decoder = model.inference_decoder
    for config in (decoder.train_cfg, decoder.schedule, decoder.solver_cfg.schedule):
        if config.t_min != 1e-4 or config.t_max != 80.0:
            raise ValueError("Diffusion bounds differ from the paper experiment")
    return {
        "t_min": 1e-4,
        "t_max": 80.0,
        "loss": decoder.loss_type,
        "steps": decoder.solver_cfg.num_steps,
        "solver_class": type(decoder.solver_cfg).__name__,
        "ode_method": decoder.solver_cfg.ode_method,
        "lognoise_mean": decoder.train_cfg.lognoise_mean,
        "lognoise_scale": decoder.train_cfg.lognoise_scale,
    }
