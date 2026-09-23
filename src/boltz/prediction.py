"""Runtime settings and resource ownership for command-line prediction."""

import gc
import platform
from contextlib import suppress
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy


def resolve_precision(
    model: str,
    accelerator: str,
    precision: Optional[str] = None,
    low_memory: bool = False,
) -> str:
    """Resolve opt-in inference precision without changing legacy defaults."""
    if model != "boltz2":
        if low_memory or precision is not None:
            raise ValueError("--low_memory and --precision require --model boltz2.")
        return "32-true"
    if not low_memory and precision is None:
        return "bf16-mixed"
    if accelerator not in {"gpu", "cpu"}:
        raise ValueError("The new inference modes support GPU and CPU accelerators.")
    if precision not in {None, "32-true", "bf16-mixed", "16-mixed"}:
        raise ValueError(f"Unsupported inference precision: {precision}")
    if accelerator == "cpu":
        if precision not in {None, "32-true"}:
            raise ValueError("Use --precision 32-true for CPU inference.")
        return "32-true"
    if not torch.cuda.is_available():
        raise ValueError("GPU inference requires an available CUDA device.")
    bf16_supported = torch.cuda.is_bf16_supported()
    if precision == "bf16-mixed" and not bf16_supported:
        raise ValueError("This GPU does not support BF16; use --precision 16-mixed.")
    return precision or ("bf16-mixed" if bf16_supported else "16-mixed")


def run_prediction_stage(
    model_cls: type[LightningModule],
    checkpoint: Path,
    model_kwargs: dict,
    data_module: LightningDataModule,
    trainer_kwargs: dict,
    num_records: int,
) -> None:
    """Own and release one model/Trainer before the next checkpoint is loaded."""
    trainer = model_module = None
    kwargs = dict(trainer_kwargs)
    devices = kwargs["devices"]
    requested = devices if isinstance(devices, int) else len(devices)
    if requested < 1:
        raise ValueError("At least one prediction device is required.")
    count = max(1, min(requested, num_records))
    kwargs["devices"] = count if isinstance(devices, int) else devices[:count]
    try:
        trainer = Trainer(
            strategy=(
                DDPStrategy(
                    start_method="spawn" if platform.system() == "Windows" else "fork"
                )
                if count > 1
                else "auto"
            ),
            **kwargs,
        )
        model_module = model_cls.load_from_checkpoint(
            checkpoint, strict=True, map_location="cpu", **model_kwargs
        )
        model_module.eval()
        trainer.predict(model_module, datamodule=data_module, return_predictions=False)
    finally:
        # Lightning moves weights to CPU during teardown, but retains model and
        # datamodule references. Drop both sides of these cycles before loading
        # another checkpoint, including when a stage fails.
        data_module.trainer = None
        if model_module is not None:
            model_module.trainer = None
        trainer = model_module = None
        gc.collect()
        if torch.cuda.is_initialized():
            # A failed CUDA operation may also make cache cleanup fail. Keep
            # the original prediction error instead of masking it in teardown.
            with suppress(RuntimeError):
                torch.cuda.empty_cache()
