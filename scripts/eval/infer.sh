#!/bin/bash
export TRANSFORMERS_CACHE=/hf_cache/
export HF_DATASETS_CACHE=/hf_cache/

num_gpus=8

model_path=/models/echo/pytorch_model.bin
model_name=echo

inf_input_dir=/your/path/to/dataset/
inf_output_dir=/your/path/to/dataset/${model_name}/

# Run inference
python /wangbenyou/xinyuan/AudioCaption/src/GAMA/gama_inf_dataset.py  \
  --gpu_count ${num_gpus} \
  --json_input ${inf_input_dir} \
  --json_output ${inf_output_dir}output.json \
  --model_path ${model_path} \
  --do_sample True \
  --num_beams 0 \
  --temperature 0.1 \
  --top_p 0.95 \
  --top_k 300 \
  --max_new_tokens 1024 \