#!/usr/bin/env bash

python -m experiments.precision_tabular_experiment \
--config_dir=experiments/precision_tabular_configs \
--log_out=experiments/logs/precision_tabular.log \
--save_dir=experiments/precision_tabular_results

