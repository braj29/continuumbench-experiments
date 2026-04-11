SHELL := /bin/bash

.DEFAULT_GOAL := help

RUN_SCRIPT := ./scripts/run_local_experiments.sh

PYTHON_BIN ?= python
UV_BIN ?= uv
DATASET_NAME ?= rel-f1
TASK_NAME ?= driver-top3
MODELS ?= tabicl,tabpfn
DEVICE ?= auto
SEED ?= 7
OUTPUT_ROOT ?= outputs/local_runs
RT_REPO_PATH ?= $(CURDIR)/relational-transformer
DOWNLOAD_ARTIFACTS ?= true
TABPFN_IGNORE_PRETRAINING_LIMITS ?= true

export PYTHON_BIN
export UV_BIN
export DATASET_NAME
export TASK_NAME
export MODELS
export DEVICE
export SEED
export OUTPUT_ROOT
export RT_REPO_PATH
export DOWNLOAD_ARTIFACTS
export TABPFN_IGNORE_PRETRAINING_LIMITS

.PHONY: help setup test continuumbench-smoke continuumbench continuumbench-rt

help:
	$(RUN_SCRIPT)

setup test continuumbench-smoke continuumbench continuumbench-rt:
	$(RUN_SCRIPT) $@
