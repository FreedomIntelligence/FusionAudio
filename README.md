# ECHO-1.6M, Towards Fine-grained Audio Captioning with Multimodal Contextual Cues 

### Setup Environment

```shell
conda create -n echo python=3.10
conda activate echo
pip install -r requirements.txt
pip install -e src/GAMA/hf-dev-train/transformers-main
pip install -e src/GAMA/peft-main
```

----

### Training

The format of the dataset is a JSON file of a list of dicts, in the following format:

```json
[
 {
  "audio_id": "path_to_audio_file",
  "instruction": "Question",
  "input":"",
  "dataset": "dataset_name",
  "task": "type_of_task",
  "output": "correct_answer"
 },
  ...
]
```

- Download the Llama-2-7b-chat-hf-qformer from [GAMA 'README' Traning](https://github.com/Sreyan88/GAMA).
- Update the path of the dowloaded Llama-2-7b-chat-hf-qformer in [gama_finetune.py](.src/GAMA/gama_finetune.py) on line 96 and 101.

Use the following commands to train the model:

```shell
conda activate echo
cd scripts/train/
bash train.sh
```

----

### Inference & Evaluation

You can use the scripts to evaluation ECHO on classification tasks directly, just change the model and dataset name to start. Of course, you need to change the API_base_url and API_KEY  if you want to use the evaluation model.

```shell
cd scripts/eval
bash eval_cls.sh
```

If you want to evaluation ECHO on other benchmark or your own dataset, you need to change the data path and use the corresponding code, like [AudioCapsQA_eval.py](.src/eval/AudioBench/AudioCapsQA/AudioCapsQA_eval.py).

```shell
cd scripts/eval
bash infer.sh
```

