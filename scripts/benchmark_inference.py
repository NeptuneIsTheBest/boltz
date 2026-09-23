"""Measure a fresh prediction process, its workers and per-stage CUDA peaks.

Requires psutil in addition to Boltz's dependencies. Example:
python scripts/benchmark_inference.py --report baseline.json -- predict input.yaml --seed 42
python scripts/benchmark_inference.py --report compact.json -- predict input.yaml --seed 42 --low_memory
Use separate output directories (or --override) to avoid measuring skipped predictions.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_worker(report: Path, source: str, arguments: list[str]) -> None:
    """Run the actual CLI with an observer that also works with the old CLI."""
    if source:
        sys.path.insert(0, source)
    import pytorch_lightning as pl
    import torch

    stages = []

    class MemoryObserver(pl.Callback):
        def on_predict_start(self, trainer, pl_module):
            self.failed = 0
            if pl_module.device.type == "cuda":
                torch.cuda.synchronize(pl_module.device)
                torch.cuda.reset_peak_memory_stats(pl_module.device)
            self.started = time.perf_counter()

        def on_predict_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            self.failed += int(outputs.get("exception", False))
            for key in (
                "coords",
                "confidence_score",
                "affinity_pred_value",
                "affinity_probability_binary",
            ):
                if key in outputs and not torch.isfinite(outputs[key]).all():
                    raise RuntimeError(f"Non-finite prediction: {key}")

        def on_predict_end(self, trainer, pl_module):
            cuda = pl_module.device.type == "cuda"
            if cuda:
                torch.cuda.synchronize(pl_module.device)
            stages.append(
                {
                    "stage": "affinity"
                    if getattr(pl_module, "affinity_prediction", False)
                    else "structure",
                    "seconds": time.perf_counter() - self.started,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        pl_module.device
                    )
                    if cuda
                    else 0,
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        pl_module.device
                    )
                    if cuda
                    else 0,
                    "failed_batches": self.failed,
                }
            )
            report.write_text(json.dumps(stages, indent=2), encoding="utf-8")

    class ObservedTrainer(pl.Trainer):
        def __init__(self, *args, **kwargs):
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), MemoryObserver()]
            kwargs["enable_progress_bar"] = False
            kwargs["logger"] = False
            super().__init__(*args, **kwargs)

    pl.Trainer = ObservedTrainer
    from boltz.main import cli

    cli.main(args=arguments, standalone_mode=False)
    if not stages or any(stage["failed_batches"] for stage in stages):
        raise RuntimeError("Benchmark did not complete all requested predictions.")


def main() -> None:
    """Measure aggregate RSS in the parent so imports and loading are included."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--source", default="", help="Optional src directory for a baseline checkout"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.worker:
        run_worker(args.report, args.source, arguments)
        return

    import psutil

    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    stage_report = report.with_suffix(".stages.json")
    # A failed or skipped rerun must not inherit successful stage measurements.
    stage_report.write_text("[]", encoding="utf-8")
    log = report.with_suffix(".log")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--report",
        str(stage_report),
    ]
    if args.source:
        command.extend(["--source", str(Path(args.source).resolve())])
    command.extend(["--", *arguments])
    peak_rss = 0
    monitor_errors = []
    started = time.perf_counter()
    # The parent starts no CUDA context; all GPU memory belongs to the child.
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        owned_processes = {process.pid: process}
        try:
            while True:
                rss = 0
                try:
                    processes = [process, *process.children(recursive=True)]
                except psutil.NoSuchProcess:
                    processes = []
                except (psutil.AccessDenied, OSError) as exc:
                    # Keep waiting for this child if Windows enumeration fails
                    # under commit pressure, so the next run cannot overlap it.
                    processes = [process]
                    if str(exc) not in monitor_errors:
                        monitor_errors.append(str(exc))
                for entry in processes:
                    owned_processes[entry.pid] = entry
                    try:
                        rss += entry.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                    except (psutil.AccessDenied, OSError) as exc:
                        if str(exc) not in monitor_errors:
                            monitor_errors.append(str(exc))
                peak_rss = max(peak_rss, rss)
                if child.poll() is not None:
                    break
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                # On interruption, stop only this benchmark and its workers.
                # psutil checks process identity before terminating a reused PID.
                for entry in reversed(list(owned_processes.values())):
                    try:
                        entry.terminate()
                    except psutil.Error:
                        pass
                child.wait(timeout=10)
    result = {
        "arguments": arguments,
        "source": args.source,
        "exit_code": child.returncode,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "memory_measurement_complete": not monitor_errors,
        "monitor_errors": monitor_errors,
        "stages": json.loads(stage_report.read_text(encoding="utf-8"))
        if stage_report.exists()
        else [],
        "log": str(log),
    }
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(child.returncode)


if __name__ == "__main__":
    main()
