#!/usr/bin/env bash

python -m experiments.consistency_tabular_experiment \
--config_dir=experiments/consistency_tabular_configs \
--log_out=experiments/logs/consistency_tabular.log \
--save_dir=experiments/consistency_tabular_results
