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

The local wrapper also passes `--tabpfn-ignore-pretraining-limits` by default, since the
current `rel-f1/driver-top3` preprocessing path can exceed TabPFN's nominal feature cap.

Official relational-transformer track:

```bash
RT_REPO_PATH=/path/to/relational-transformer make continuumbench-rt
RT_REPO_PATH=/path/to/relational-transformer \
./scripts/run_local_experiments.sh continuumbench-rt
```

## Snellius

The simplest Snellius workflow is:

- `scripts/setup_snellius_env.sh`: create a project venv and warm the RelBench / TabPFN caches.
- `scripts/snellius_tabpfn_a100.sbatch`: final self-contained batch script for one A100 GPU job.

Suggested flow on a Snellius login node:

```bash
cd ~/continuumbench-experiments
git pull
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
./scripts/setup_snellius_env.sh
```

Submit a TabPFN run:

```bash
sbatch scripts/snellius_tabpfn_a100.sbatch
```

Watch the job:

```bash
squeue -u $USER
tail -f slurm-cb-tabpfn-<jobid>.out
```

Notes:

- The batch script follows SURF's "Final job script" pattern directly: `#SBATCH` header, load modules, use `$TMPDIR` for scratch output, run Python, then copy results back.
- Defaults are embedded in the job script: dataset `rel-f1`, task `driver-top3`, model `tabpfn`, seed `7`, device `cuda`.
- The batch script also enables `--tabpfn-ignore-pretraining-limits` by default because this benchmark can exceed TabPFN's nominal 500-feature limit after preprocessing.
- Results are copied to `outputs/snellius/<jobid>` in the repo after the job finishes.
- If you skip `./scripts/setup_snellius_env.sh`, edit `DOWNLOAD_ARTIFACTS=1` near the top of the batch script before submitting.
- The SURF documentation currently lists `gpu_a100` with a minimum allocation of `1 GPU + 18 CPU cores + 120 GiB memory`, and `/scratch-shared/<user>` is shared across nodes with automatic cleanup after 14 days:
  https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660226/Final+job+script
  https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660221/SLURM+batch+system
  https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting
  https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems

## Tests

```bash
make test

uv run python -m unittest -q \
  tests.test_models_adapters \
  tests.test_models_device_resolution \
  tests.test_models_tabular \
  tests.test_continuumbench_harness \
  tests.test_continuumbench_cli
```
