# ContinuumBench Experiments

Standalone repo for the ContinuumBench tri-view harness and the related RelBench and
GraphBench experiment runners.

This repo contains:

- `continuumbench_relbench.py`: core ContinuumBench harness.
- `continuumbench_tabfm_run.py`: ContinuumBench runner for TabICL and TabPFN.
- `relbench_run.py`: direct RelBench runner.
- `graphbench_run.py`: direct GraphBench runner.
- `src/graphbench_adapter.py`: GraphBench-to-tabular adapters.
- `src/graphbench_experiment.py`: GraphBench experiment loop.
- `src/relbench_experiment.py`: RelBench experiment loop.
- `src/model.py`: minimal TabICL and TabPFN builders.
- `tests/`: focused unit tests for the extracted code.
- `scripts/run_local_experiments.sh`: local convenience wrapper.

This extraction intentionally leaves out unrelated KG, LimiX, SAINT, TAG, and Kubernetes
job code from the original workspace.

## Setup

Install the project and GraphBench extras:

```bash
uv sync --extra dev --extra graphbench
```

## Local Runs

Quick smoke test:

```bash
./scripts/run_local_experiments.sh continuumbench-smoke
```

RelBench-backed ContinuumBench:

```bash
./scripts/run_local_experiments.sh continuumbench
```

Official relational-transformer track:

```bash
RT_REPO_PATH=/path/to/relational-transformer \
./scripts/run_local_experiments.sh continuumbench-rt
```

Direct RelBench runs:

```bash
./scripts/run_local_experiments.sh relbench-tabicl
DEVICE=cuda ./scripts/run_local_experiments.sh relbench-tabpfn
```

Direct GraphBench runs:

```bash
GRAPHBENCH_DATASET=socialnetwork GRAPHBENCH_TASK=node \
./scripts/run_local_experiments.sh graphbench-tabicl

GRAPHBENCH_DATASET=socialnetwork GRAPHBENCH_TASK=node DEVICE=cuda \
./scripts/run_local_experiments.sh graphbench-tabpfn
```

## Tests

```bash
uv run python -m unittest -q \
  tests.test_continuumbench_relbench \
  tests.test_relbench_experiment \
  tests.test_graphbench_experiment \
  tests.test_graphbench_adapter
```
