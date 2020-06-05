#!/usr/bin/env bash

python -m experiments.coverage_tabular_experiment \
--config_dir=experiments/coverage_tabular_configs \
--log_out=experiments/logs/coverage_tabular.log \
--save_dir=experiments/coverage_tabular_results

