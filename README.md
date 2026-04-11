# ContinuumBench Experiments

Standalone repo for the ContinuumBench tri-view harness and its tabular-model runner.

This repo contains:

- `src/continuumbench_experiments/continuumbench/harness.py`: thin compatibility facade over the core ContinuumBench modules.
- `src/continuumbench_experiments/continuumbench/specs.py`: task, table, split, and experiment data contracts.
- `src/continuumbench_experiments/continuumbench/views.py`: joined-table and graphified view construction.
- `src/continuumbench_experiments/continuumbench/runner.py`: Protocol A / B execution and result summaries.
- `src/continuumbench_experiments/continuumbench/examples.py`: synthetic demo problem.
- `src/continuumbench_experiments/cli/continuumbench.py`: primary ContinuumBench runner for TabICL and TabPFN.
- `src/continuumbench_experiments/models/tabular.py`: TabICL and TabPFN builders.
- `src/continuumbench_experiments/models/adapters.py`: model adapters and external model wrappers.
- `continuumbench_tabfm_run.py`: thin repo-root entry point for the main ContinuumBench runner.
- `tests/`: focused unit tests for the extracted code.
- `scripts/run_local_experiments.sh`: local convenience wrapper.

This extraction intentionally leaves out unrelated KG, LimiX, SAINT, TAG, standalone
benchmark-runner code, and Kubernetes job code from the original workspace.

## Setup

Install the project, tests, and the optional dataset-source backend:

```bash
uv sync --extra dev --extra task-source
make setup
```

## Local Runs

List the available shortcuts:

```bash
make help
```

Quick smoke test:

```bash
make continuumbench-smoke
./scripts/run_local_experiments.sh continuumbench-smoke
```

Dataset-backed ContinuumBench:

```bash
make continuumbench
./scripts/run_local_experiments.sh continuumbench
```

TabPFN-only run:

```bash
MODELS=tabpfn make continuumbench
MODELS=tabpfn ./scripts/run_local_experiments.sh continuumbench
```

Official relational-transformer track:

```bash
RT_REPO_PATH=/path/to/relational-transformer make continuumbench-rt
RT_REPO_PATH=/path/to/relational-transformer \
./scripts/run_local_experiments.sh continuumbench-rt
```

## Snellius

The repo includes a small Snellius workflow for `tabpfn` on one A100 GPU:

- `scripts/setup_snellius_env.sh`: create a project venv and warm the RelBench / TabPFN caches.
- `scripts/run_snellius_continuumbench.sh`: run the benchmark inside an existing Snellius allocation.
- `scripts/snellius_tabpfn_a100.sbatch`: example `sbatch` file for one `gpu_a100` quarter-node job.

Suggested flow on a Snellius login node:

```bash
cd /scratch-shared/$USER/continuumbench-experiments
module purge
module load 2024
module avail Python
export SNELLIUS_PYTHON_MODULE=Python/<version-from-module-avail>
./scripts/setup_snellius_env.sh
```

Submit a TabPFN run:

```bash
MODELS=tabpfn \
DATASET_NAME=rel-f1 \
TASK_NAME=driver-top3 \
sbatch --account <your-project-id> scripts/snellius_tabpfn_a100.sbatch
```

Notes:

- The job script keeps caches and scratch outputs under `/scratch-shared/$USER`, then copies run artifacts back to `outputs/snellius/<jobid>` in the repo.
- If you skip `scripts/setup_snellius_env.sh`, set `DOWNLOAD_ARTIFACTS=1` at submit time so the compute job may fetch missing RelBench artifacts.
- For interactive sanity checks, request a short GPU allocation first and run `bash scripts/run_snellius_continuumbench.sh` inside that shell.
- The SURF documentation currently lists `gpu_a100` with a minimum allocation of `1 GPU + 18 CPU cores + 120 GiB memory`, and `/scratch-shared/<user>` is shared across nodes with automatic cleanup after 14 days:
  https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting
  https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems
  https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660234/Example+job+scripts

## Tests

```bash
make test

uv run python -m unittest -q \
  tests.test_continuumbench_harness \
  tests.test_continuumbench_cli
```
