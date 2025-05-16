# FusionAudio-1.6M, Towards Fine-grained Audio Captioning with Multimodal Contextual Cues 

Download FusionAudio-1.6M text [here](https://huggingface.co/datasets/SatsukiVie/FusionAudio).

### Setup Environment

```shell
conda create -n FusionAudio python=3.10
conda activate FusionAudio
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

- Download the Llama-2-7b-chat-hf-qformer from [GAMA 'README' Training Setting](https://github.com/Sreyan88/GAMA).
- Update the path of the dowloaded Llama-2-7b-chat-hf-qformer in [gama_finetune.py](.src/GAMA/gama_finetune.py) on line 96 and 101.

Use the following commands to train the model:

```shell
conda activate FusionAudio
cd scripts/train/
bash train.sh
```

----

### Inference & Evaluation

You can use the scripts to evaluation FusionAudio on classification tasks directly, just change the model and dataset name to start. Of course, you need to change the API_base_url and API_KEY  if you want to use the evaluation model.

```shell
cd scripts/eval
bash eval_cls.sh
```
Download FusionAudio-25k and FusionAudio-25k-high checkpoint [here](https://huggingface.co/SatsukiVie/FusionAudio/tree/main).

If you want to evaluation FusionAudio on other benchmark or your own dataset, you need to change the data path and use the corresponding code, like [AudioCapsQA_eval.py](.src/eval/AudioBench/AudioCapsQA/AudioCapsQA_eval.py).

```shell
cd scripts/eval
bash infer.sh
```
If you want to evaluate an audio-text retrieval model trained on the FusionAudio dataset, you can run the following code.What you should prepare:
1.Install the correct environment and prepare val dataset according to https://github.com/XinhaoMei/WavCaps/tree/master/retrieval   
2.set the "ckpt_path" in inference.yaml to your model path
```shell
cd scripts/retrieval
python eval.py
```
