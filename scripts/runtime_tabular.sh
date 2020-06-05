#!/usr/bin/env bash

python -m experiments.runtime_tabular_experiment \
--config_dir=experiments/runtime_tabular_configs \
--log_out=experiments/logs/runtime_tabular.log \
--save_dir=experiments/runtime_tabular_results

