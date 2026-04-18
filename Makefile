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
TABRED_DATA_DIR ?= $(CURDIR)/data/tabred/homecredit-default
SPLIT ?= default
INCLUDE_META ?= 0
MAX_TRAIN_ROWS ?=
DRY_RUN ?= 0

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
export TABRED_DATA_DIR
export SPLIT
export INCLUDE_META
export MAX_TRAIN_ROWS
export DRY_RUN

.PHONY: help setup test continuumbench-smoke continuumbench continuumbench-rt tabred-homecredit-tabpfn

help:
	$(RUN_SCRIPT)

setup test continuumbench-smoke continuumbench continuumbench-rt tabred-homecredit-tabpfn:
	$(RUN_SCRIPT) $@
