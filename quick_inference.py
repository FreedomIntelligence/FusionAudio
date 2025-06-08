#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FusionAudio 快速推理脚本
基于 GAMA 架构的音频理解模型推理工具
"""

import torch
import torchaudio
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model
import sys
import os

# 添加 GAMA 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'GAMA'))
from utils.prompter import Prompter


class FusionAudioInference:
    def __init__(self, base_model_path, model_path, device="cuda:0"):
        """
        初始化 FusionAudio 推理器
        
        Args:
            base_model_path: Llama-2-7b-chat-hf-qformer 基础模型路径
            model_path: FusionAudio 微调权重路径
            device: 推理设备
        """
        self.device = device
        self.model, self.tokenizer = self._load_model(base_model_path, model_path)
        self.prompter = Prompter('alpaca_short')
        
    def _load_model(self, base_model_path, model_path):
        """Load model and tokenizer"""
        print(f"Loading base model from {base_model_path}...")
        
        # Load tokenizer and base model
        tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        model = LlamaForCausalLM.from_pretrained(
            base_model_path, 
            device_map={"": self.device}, 
            torch_dtype=torch.float32
        )
        
        # Configure LoRA
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        
        # Set token configuration
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        
        # Load fine-tuned weights
        print(f"Loading FusionAudio weights from {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print("Model loaded successfully!")
        return model, tokenizer
    
    def _load_audio(self, filename):
        """Audio preprocessing"""
        # Set audio backend
        try:
            import soundfile
            torchaudio.set_audio_backend("soundfile")
        except:
            try:
                torchaudio.set_audio_backend("sox_io")
            except:
                print("Warning: Unable to set audio backend, may affect loading certain audio formats")
        
        # Load audio
        waveform, sr = torchaudio.load(filename)
        
        # Resample to 16kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        
        # Audio preprocessing
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type='hanning',
            num_mel_bins=128, dither=0.0, frame_shift=10
        )
        
        # Pad or truncate to fixed length
        target_length = 1024
        n_frames = fbank.shape[0]
        if n_frames < target_length:
            pad = torch.nn.ZeroPad2d((0, 0, 0, target_length - n_frames))
            fbank = pad(fbank)
        else:
            fbank = fbank[:target_length, :]
        
        # Normalize
        fbank = (fbank + 5.081) / 4.4849
        return fbank
    
    def predict(self, audio_path, question, temperature=0.1, top_p=0.95, top_k=300, max_new_tokens=1024):
        """
        执行推理
        
        Args:
            audio_path: 音频文件路径，如果为 None 或 'empty' 则只处理文本
            question: 问题或指令
            temperature: 生成温度
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            max_new_tokens: 最大生成token数
            
        Returns:
            str: 模型回答
        """
        # 准备prompt
        prompt = self.prompter.generate_prompt(question, None)
        
        # 文本输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Audio input
        if audio_path and audio_path != 'empty':
            try:
                audio_input = self._load_audio(audio_path).unsqueeze(0).to(self.device)
                print(f"Audio loaded: {audio_path}")
            except Exception as e:
                print(f"Failed to load audio: {e}")
                audio_input = None
        else:
            audio_input = None
            print("No audio provided, performing text-only inference")
        
        # Generation configuration
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
        )
        
        # Inference
        print("Generating response...")
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                audio_input=audio_input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        
        # Decode output
        output_sequence = generation_output.sequences[0]
        response = self.tokenizer.decode(output_sequence)[len(prompt)+6:-4]
        return response


def main():
    parser = argparse.ArgumentParser(
        description="FusionAudio Quick Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audio captioning
  python quick_inference.py --base_model /path/to/Llama-2-7b-chat-hf-qformer \\
                           --model_path /path/to/fusionaudio_checkpoint.pth \\
                           --audio /path/to/audio.wav \\
                           --question "Please describe this audio in detail."
  
  # Audio question answering
  python quick_inference.py --base_model /path/to/Llama-2-7b-chat-hf-qformer \\
                           --model_path /path/to/fusionaudio_checkpoint.pth \\
                           --audio /path/to/audio.wav \\
                           --question "What is the main sound in this audio?"
  
  # Text-only conversation (without audio)
  python quick_inference.py --base_model /path/to/Llama-2-7b-chat-hf-qformer \\
                           --model_path /path/to/fusionaudio_checkpoint.pth \\
                           --question "Hello, how are you?"

Parameter Details:
  --base_model    Path to the Llama-2-7b-chat-hf-qformer base model directory
  --model_path    Path to the FusionAudio fine-tuned checkpoint (.pth file)
  --audio         Path to audio file (supports .wav, .mp3, .flac, etc.)
                  If not provided, performs text-only inference
  --question      Question or instruction for the model
  --device        Inference device (cuda:0, cuda:1, cpu, etc.)
  --temperature   Sampling temperature (0.0-1.0, lower = more deterministic)
  --top_p         Nucleus sampling parameter (0.0-1.0)
  --top_k         Top-k sampling parameter (positive integer)
  --max_tokens    Maximum number of tokens to generate
        """
    )
    
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to Llama-2-7b-chat-hf-qformer base model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to FusionAudio fine-tuned weights")
    parser.add_argument("--audio", type=str, default=None,
                       help="Path to audio file (optional)")
    parser.add_argument("--question", type=str, default="Please describe this audio in detail.",
                       help="Question or instruction")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Inference device (default: cuda:0)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature (default: 0.1)")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Nucleus sampling parameter (default: 0.95)")
    parser.add_argument("--top_k", type=int, default=300,
                       help="Top-k sampling parameter (default: 300)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate (default: 1024)")
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    
    # Initialize inferencer
    try:
        inferencer = FusionAudioInference(args.base_model, args.model_path, args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Perform inference
    try:
        response = inferencer.predict(
            args.audio, 
            args.question,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens
        )
        
        print("\n" + "="*50)
        print("Question:", args.question)
        if args.audio:
            print("Audio:", args.audio)
        print("Response:", response)
        print("="*50)
        
    except Exception as e:
        print(f"Inference failed: {e}")


if __name__ == "__main__":
    main() 