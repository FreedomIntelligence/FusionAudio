#!/bin/bash
export TRANSFORMERS_CACHE=/hf_cache/
export HF_DATASETS_CACHE=/hf_cache/

num_gpus=8

model_path=/wangbenyou/xinyuan/models/GAMA4-2/pytorch_model.bin
model_name=gama

inf_input_dir=/wangbenyou/xinyuan/ECHO-AQA/data/eval/instruction_ref/fsd50k_eval.json
inf_output_dir=/wangbenyou/xinyuan/ECHO-AQA/results/FSD50K/fsd50k_${model_name}/
eval_output_dir=${inf_output_dir}t-0.01.json

# Run inference
python /wangbenyou/xinyuan/AudioCaption/src/GAMA/gama_inf_dataset.py  \
  --gpu_count ${num_gpus} \
  --json_input ${inf_input_dir} \
  --json_output ${inf_output_dir}output.json \
  --model_path ${model_path} \
  --do_sample True \
  --num_beams 0 \
  --temperature 0.01 \
  --top_p 0.95 \
  --top_k 300 \
  --max_new_tokens 1024 \

# align output format
python /wangbenyou/xinyuan/AudioCaption/src/eval/Cls_task/modify_json.py \
 --input_dir ${inf_output_dir}output.json \
 --output_dir ${eval_output_dir} \
 --ref_dir ${inf_input_dir} \