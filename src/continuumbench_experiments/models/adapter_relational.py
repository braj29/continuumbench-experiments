from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..continuumbench.specs import TaskSpec
from .adapter_common import BaseViewModel


class ExternalRelationalAdapter(BaseViewModel):
    def __init__(
        self,
        name: str,
        fit_fn: Callable[..., tuple[Any, dict[str, Any]]],
        predict_fn: Callable[..., np.ndarray],
        **kwargs: Any,
    ):
        self._name = name
        self._view_name = "relational"
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.kwargs = kwargs
        self.model_obj: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        self.model_obj, metadata = self.fit_fn(
            train_data=train_data,
            val_data=val_data,
            task=task,
            **self.kwargs,
        )
        return metadata

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        return self.predict_fn(model_obj=self.model_obj, data=data, task=task, **self.kwargs)


class OfficialRelationalTransformerAdapter(BaseViewModel):
    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        target_col: str,
        rt_repo_path: str,
        name: str = "rt-official",
        python_executable: Optional[str] = None,
        columns_to_drop: Optional[Sequence[str]] = None,
        project: str = "continuumbench-rt",
        eval_splits: Sequence[str] = ("val", "test"),
        eval_freq: int = 1000,
        eval_pow2: bool = False,
        max_eval_steps: int = 40,
        load_ckpt_path: Optional[str] = None,
        save_ckpt_dir: Optional[str] = None,
        compile_: bool = False,
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 8,
        max_bfs_width: int = 256,
        lr: float = 1e-4,
        wd: float = 0.0,
        lr_schedule: bool = False,
        max_grad_norm: float = 1.0,
        max_steps: int = 2**12 + 1,
        embedding_model: str = "all-MiniLM-L12-v2",
        d_text: int = 384,
        seq_len: int = 1024,
        num_blocks: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
    ):
        self._name = name
        self._view_name = "relational"
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.target_col = target_col
        self.rt_repo_path = str(rt_repo_path)
        self.python_executable = python_executable or sys.executable
        self.columns_to_drop = list(columns_to_drop or [])
        self.project = project
        self.eval_splits = list(eval_splits)
        self.eval_freq = int(eval_freq)
        self.eval_pow2 = bool(eval_pow2)
        self.max_eval_steps = int(max_eval_steps)
        self.load_ckpt_path = load_ckpt_path
        self.save_ckpt_dir = save_ckpt_dir
        self.compile_ = bool(compile_)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_bfs_width = int(max_bfs_width)
        self.lr = float(lr)
        self.wd = float(wd)
        self.lr_schedule = bool(lr_schedule)
        self.max_grad_norm = float(max_grad_norm)
        self.max_steps = int(max_steps)
        self.embedding_model = embedding_model
        self.d_text = int(d_text)
        self.seq_len = int(seq_len)
        self.num_blocks = int(num_blocks)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

    @property
    def name(self) -> str:
        return self._name

    @property
    def view_name(self) -> str:
        return self._view_name

    def _rt_main_config(self) -> dict[str, Any]:
        task_tuple = (
            self.dataset_name,
            self.task_name,
            self.target_col,
            self.columns_to_drop,
        )
        # Zero-shot sentinel: max_steps=0 means "evaluate the loaded checkpoint
        # without any fine-tuning".  RT's training loop is `while steps < max_steps`,
        # so with max_steps=0 the loop never runs and evaluate() is never called.
        # Work-around: run exactly 1 step with eval_freq=1 so that the evaluation
        # fires at step=0 (before the single gradient update) and prints the metric.
        if self.max_steps == 0:
            effective_max_steps = 1
            effective_eval_freq = 1
        else:
            effective_max_steps = self.max_steps
            effective_eval_freq = self.eval_freq

        return {
            "project": self.project,
            "eval_splits": self.eval_splits,
            "eval_freq": effective_eval_freq,
            "eval_pow2": self.eval_pow2,
            "max_eval_steps": self.max_eval_steps,
            "load_ckpt_path": self.load_ckpt_path,
            "save_ckpt_dir": self.save_ckpt_dir,
            "compile_": self.compile_,
            "seed": self.seed,
            "train_tasks": [task_tuple],
            "eval_tasks": [task_tuple],
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "max_bfs_width": self.max_bfs_width,
            "lr": self.lr,
            "wd": self.wd,
            "lr_schedule": self.lr_schedule,
            "max_grad_norm": self.max_grad_norm,
            "max_steps": effective_max_steps,
            "embedding_model": self.embedding_model,
            "d_text": self.d_text,
            "seq_len": self.seq_len,
            "num_blocks": self.num_blocks,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
        }

    def _extract_test_metric(self, logs: str) -> Optional[float]:
        numeric = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        dataset = re.escape(self.dataset_name)
        task = re.escape(self.task_name)
        patterns = [
            rf"(?:auc|r2|mae)/{dataset}/{task}/test:\s*{numeric}",
            rf"{dataset}/{task}/test:\s*{numeric}",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, logs)
            if matches:
                return float(matches[-1])
        return None

    def _validate_runtime_environment(self, repo_path: Path) -> None:
        if platform.system() != "Linux":
            raise RuntimeError(
                "Official Relational Transformer is not runnable on this machine. "
                "The upstream runtime expects Linux + CUDA. "
                "Use Snellius or another Linux GPU box."
            )

        probe = subprocess.run(
            [
                self.python_executable,
                "-c",
                (
                    "import json, platform\n"
                    "import torch\n"
                    "print(json.dumps({'platform': platform.system(), "
                    "'cuda_available': bool(torch.cuda.is_available())}))\n"
                ),
            ],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            tail = "\n".join(((probe.stdout or "") + "\n" + (probe.stderr or "")).splitlines()[-40:])
            raise RuntimeError(
                "Official RT runtime probe failed for the configured python executable.\n"
                f"--- tail ---\n{tail}"
            )

        try:
            payload = json.loads((probe.stdout or "").strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Official RT runtime probe returned unexpected output.\n"
                f"stdout={probe.stdout!r}\nstderr={probe.stderr!r}"
            ) from exc

        runtime_platform = payload.get("platform")
        cuda_available = bool(payload.get("cuda_available"))
        if runtime_platform != "Linux" or not cuda_available:
            raise RuntimeError(
                "Official Relational Transformer requires a Linux + CUDA runtime "
                f"in the target python environment. Detected platform={runtime_platform!r}, "
                f"cuda_available={cuda_available}. "
                "Use Snellius or another Linux GPU box."
            )

    def fit(self, train_data: Any, val_data: Optional[Any], task: TaskSpec) -> dict[str, Any]:
        repo_path = Path(self.rt_repo_path).expanduser().resolve()
        rt_main_path = repo_path / "rt" / "main.py"
        if not rt_main_path.is_file():
            raise FileNotFoundError(
                f"Official RT repo not found at '{repo_path}'. Missing rt/main.py."
            )
        self._validate_runtime_environment(repo_path)

        env = os.environ.copy()
        env["RT_MAIN_CONFIG"] = json.dumps(self._rt_main_config())
        command = [
            self.python_executable,
            "-c",
            (
                "import json, os\n"
                "from rt.main import main\n"
                "main(**json.loads(os.environ['RT_MAIN_CONFIG']))\n"
            ),
        ]
        result = subprocess.run(
            command,
            cwd=str(repo_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        logs = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        if result.returncode != 0:
            tail = "\n".join(logs.splitlines()[-120:])
            dependency_hint = ""
            missing_module_to_package = {
                "wandb": "wandb",
                "maturin_import_hook": "maturin-import-hook",
                "maturin": "maturin[patchelf]",
                "strictfire": "strictfire",
                "orjson": "orjson",
                "ml_dtypes": "ml-dtypes",
                "einops": "einops",
                "sentence_transformers": "sentence-transformers",
            }
            missing_packages = []
            for module_name, package_name in missing_module_to_package.items():
                marker = f"ModuleNotFoundError: No module named '{module_name}'"
                if marker in logs:
                    missing_packages.append(package_name)

            if missing_packages:
                unique_packages = ", ".join(sorted(set(missing_packages)))
                dependency_hint = (
                    "\nHint: install missing RT runtime dependencies in the same python env, e.g. "
                    f"`python -m pip install --upgrade {unique_packages}`."
                )
            elif "ModuleNotFoundError: No module named 'rustler'" in logs:
                dependency_hint = (
                    "\nHint: RT's rust sampler extension is missing. "
                    "Install/build it in the RT environment (requires Rust toolchain), e.g. "
                    "`python -m pip install -e <rt_repo_path>/rustler`."
                )
            raise RuntimeError(
                "Official RT run failed. "
                f"returncode={result.returncode}{dependency_hint}\n--- tail ---\n{tail}"
            )

        metric = self._extract_test_metric(logs)
        if metric is None:
            tail = "\n".join(logs.splitlines()[-120:])
            raise RuntimeError(
                "Official RT run succeeded but test metric was not found in logs.\n"
                f"--- tail ---\n{tail}"
            )

        return {
            "backend": "official_relational_transformer",
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "target_col": self.target_col,
            "precomputed_test_metric": float(metric),
        }

    def predict(self, data: Any, task: TaskSpec) -> np.ndarray:
        raise RuntimeError(
            "OfficialRelationalTransformerAdapter does not emit row-wise predictions "
            "in this integration. Use precomputed_test_metric instead."
        )


def stub_relational_fit_fn(
    train_data: Any,
    val_data: Any,
    task: TaskSpec,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    model_obj = {"mean": float(train_data["task_df"][task.target_col].mean())}
    return model_obj, {"backend": "stub_relational"}


def stub_relational_predict_fn(
    model_obj: Any,
    data: Any,
    task: TaskSpec,
    **kwargs: Any,
) -> np.ndarray:
    return np.full(len(data["task_df"]), model_obj["mean"], dtype=float)
