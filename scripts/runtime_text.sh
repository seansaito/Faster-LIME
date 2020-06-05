#!/usr/bin/env bash

python -m experiments.runtime_text_experiment \
--config_dir=experiments/runtime_text_configs \
--log_out=experiments/logs/runtime_text.log \
--save_dir=experiments/runtime_text_results

