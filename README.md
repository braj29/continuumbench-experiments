# ContinuumBench Experiments

Standalone repo for the ContinuumBench tri-view harness and its tabular-model runner.

This repo contains:

- `src/continuumbench_experiments/continuumbench/harness.py`: core ContinuumBench harness.
- `src/continuumbench_experiments/cli/continuumbench.py`: primary ContinuumBench runner for TabICL and TabPFN.
- `src/continuumbench_experiments/models/tabular.py`: TabICL and TabPFN builders.
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

Official relational-transformer track:

```bash
RT_REPO_PATH=/path/to/relational-transformer make continuumbench-rt
RT_REPO_PATH=/path/to/relational-transformer \
./scripts/run_local_experiments.sh continuumbench-rt
```

## Tests

```bash
make test

uv run python -m unittest -q tests.test_continuumbench_harness
```
