# FusionAudio-1.2M, Towards Fine-grained Audio Captioning with Multimodal Contextual Cues 



---

🚀🚀🚀 Official implementation of **FusionAudio-1.2M**: Towards Fine-grained Audio Captioning with Multimodal Contextual Cues

![sample](imgs\sample.png)

* **Authors**: [Shunian Chen*](https://github.com/Shunian-Chen), [Xinyuan Xie*](https://github.com/satsuki2486441738), [Zheshu Chen*](https://github.com/kawagebo12), [Liyan Zhao](https://github.com/Apostasi0225cuhksz), [Owen Lee](https://github.com/KaiTheSkyWalker), [Zhan Su](https://scholar.google.com/citations?user=VzEpVpoAAAAJ), [Qilin Sun](https://scholar.google.com/citations?user=igqPS8sAAAAJ), [Benyou Wang](https://scholar.google.com.hk/citations?user=Jk4vJU8AAAAJ)
* **Institutions**: The Chinese University of Hong Kong, Shenzhen
* **Resources**: [📄Paper](https://arxiv.org/abs/2506.01111)  [🤗Dataset](https://huggingface.co/datasets/SatsukiVie/FusionAudio)
* **Models**: [🤗FusionAudio](https://huggingface.co/SatsukiVie/FusionAudio)

## 💡 Highlights

* 🔥 **Large-scale high-quality** audio captioning dataset **FusionAudio-1.2M**
* 🔥 **Multimodal context fusion** for more fine-grained audio understanding
* 🔥 **SOTA performance** achieving state-of-the-art results on multiple audio understanding benchmarks

## 📜 News
**\[2025/06/01\]** 🚀 Our papar [FusionAudio-1.2M, Towards Fine-grained Audio Captioning with Multimodal Contextual Cues](https://arxiv.org/abs/2506.01111) is available!

**\[2025/05/16\]** 🚀 Released FusionAudio-1.2M [dataset](https://huggingface.co/datasets/SatsukiVie/FusionAudio), [model](https://huggingface.co/SatsukiVie/FusionAudio/tree/main), and code!

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n FusionAudio python=3.10
conda activate FusionAudio

# Install dependencies
pip install -r requirements.txt
pip install -e src/GAMA/hf-dev-train/transformers-main
pip install -e src/GAMA/peft-main
```

### Quick Inference

We provide an easy-to-use inference script `quick_inference.py` that supports both command-line and Python API usage.

#### Command Line Usage

```bash
python quick_inference.py \
    --base_model /path/to/Llama-2-7b-chat-hf-qformer \
    --model_path /path/to/fusionaudio_checkpoint.pth \
    --audio /path/to/your/audio.wav \
    --question "Please describe this audio in detail."
```

#### Python API Usage

```python
from quick_inference import FusionAudioInference

# Initialize inferencer
inferencer = FusionAudioInference(
    base_model_path="/path/to/Llama-2-7b-chat-hf-qformer",
    model_path="/path/to/fusionaudio_checkpoint.pth",
    device="cuda:0"
)

# Audio captioning
response = inferencer.predict(
    audio_path="/path/to/your/audio.wav",
    question="Please describe this audio in detail."
)
print(f"Audio description: {response}")
```

For detailed parameter descriptions, run `python quick_inference.py --help`.

## 📊 Dataset

### FusionAudio-1.2M

We constructed a large-scale dataset containing 1.2 million high-quality audio-text pairs.

**Download Link**: [🤗 Hugging Face](https://huggingface.co/datasets/SatsukiVie/FusionAudio)

#### Data Format

```json
[
  {
    "audio_id": "path_to_audio_file",
    "instruction": "Question",
    "input": "",
    "dataset": "dataset_name", 
    "task": "type_of_task",
    "output": "correct_answer"
  }
]
```

## 🏋️ Training

### Preprocessing

1. Download Llama-2-7b-chat-hf-qformer model (refer to [GAMA README](https://github.com/Sreyan88/GAMA))
2. Update the model path in `src/GAMA/gama_finetune.py` at lines 96 and 101

### Start Training

```bash
conda activate FusionAudio
cd scripts/train/
bash train.sh
```

## 📈 Evaluation

### Classification Task Evaluation

```bash
cd scripts/eval
bash eval_cls.sh
```

### Captioning Evaluation

```bash
cd scripts/eval  
bash infer.sh
```

### Retrieval Task Evaluation

```bash
# Environment preparation (refer to WavCaps repository)
# 1. Configure environment according to https://github.com/XinhaoMei/WavCaps/tree/master/retrieval
# 2. Set ckpt_path in inference.yaml
# 3. Put eval_retrieval.py into the downloaded retrieval folder

cd scripts
python eval_retrieval.py
```

## 📋 Data Statistics

![statistics](imgs\statistics.png)

## 🛠️ Model Downloads

| Model Name | Purpose | Download Link |
|---------|------|----------|
| FusionAudio-25k/FusionAudio-25k-high | General audio understanding | [🤗 HuggingFace](https://huggingface.co/SatsukiVie/FusionAudio) |
| FusionAudio-Retrieval | Audio retrieval | [🤗 HuggingFace](https://huggingface.co/Zheshu/FusionAudio-Retrieval) |


## ❤️ Acknowledgments

* **GAMA**: Thanks for providing excellent infrastructure
* **WavCaps**: Thanks for pioneering work in audio captioning
* **Llama**: Thanks for providing powerful language model foundation

## ✒️ Citation

If our work is helpful for your research, please consider giving a star ⭐ and citing our paper 📝

```bibtex
@misc{chen2025fusionaudio12mfinegrainedaudiocaptioning,
      title={FusionAudio-1.2M: Towards Fine-grained Audio Captioning with Multimodal Contextual Fusion}, 
      author={Shunian Chen and Xinyuan Xie and Zheshu Chen and Liyan Zhao and Owen Lee and Zhan Su and Qilin Sun and Benyou Wang},
      year={2025},
      eprint={2506.01111},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2506.01111}, 
}
```

## 📄 License

**Usage License**: This dataset and models are intended for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and other related models. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

---

<div align="center">

**🌟 If this project helps you, please give us a Star! 🌟**

</div>
