# gama_inf_dataset.py
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch.multiprocessing as mp
import numpy as np
import torchaudio
import json
import torch
from tqdm import tqdm
import subprocess
import torchvision
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter
from functools import partial

def get_available_gpus(num_gpus=None):
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split(), universal_newlines=True)

        gpus = []
        for line in output.strip().split('\n'):
            index, used_mem, total_mem = map(float, line.split(','))
            mem_usage = used_mem / total_mem
            if mem_usage < 0.1: 
                gpus.append(int(index))
        
        if num_gpus is not None:
            return gpus[:num_gpus]
        else:
            return gpus
    except Exception as e:
        print(f"Error: {e}")
        return []

def process_subset(gpu_id, data_subset, base_model, model_path, args):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"{os.getpid()} use {device}", flush=True)
    
    checkpoint_path = os.path.join(os.path.dirname(args.json_output), f"checkpoint_gpu_{gpu_id}.json")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Load {checkpoint_path},  {len(results)} samples", flush=True)
        processed_count = len(results)
        data_subset = data_subset[processed_count:]
    else:
        results = []
        processed_count = 0
    
    prompter = Prompter('alpaca_short')
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(base_model, device_map={"": gpu_id}, torch_dtype=torch.float32)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    
    model.is_parallelizable = True
    model.model_parallel = True
    
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    model.eval()

    state_dict = torch.load(model_path, map_location='cpu')
    msg = model.load_state_dict(state_dict, strict=False)

    for idx, sample in enumerate(tqdm(data_subset, desc=f"GPU-{gpu_id} inference", unit="it"), start=0):
        total_index = processed_count + idx
        audio_id = sample.get("audio_id", "")
        instruction = sample.get("instruction", "")
        audio_info, response = predict(audio_id, instruction, model, tokenizer, prompter, device, args)
        if audio_info is None and response is None:
            continue
        sample_result = sample.copy()
        sample_result["output"] = response
        results.append(sample_result)
        if (total_index + 1) % 20 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Save {checkpoint_path}, {total_index + 1} samples", flush=True)
    

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def load_audio(filename):
    try:
        import soundfile
        torchaudio.set_audio_backend("soundfile")
    except Exception as e_sf:
        print(f"Soundfile backend not available or failed to set: {e_sf}. Trying sox_io.")
        try:
            torchaudio.set_audio_backend("sox_io")
        except Exception as e_sox:
            print(f"Sox_io backend not available or failed to set: {e_sox}.")
            print("Warning: Could not set soundfile or sox_io backend. MP3 loading might fail.")
            print("Consider installing soundfile (`pip install soundfile`) or SoX.")

    waveform, sr = torchaudio.load(filename)
    audio_info = 'Original input audio length {:.2f} seconds, number of channels: {:d}, sampling rate: {:d}.'.format(waveform.shape[1]/sr, waveform.shape[0], sr)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform=waveform, orig_freq=sr, new_freq=16000)
        sr = 16000
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # normalize the fbank
    fbank = (fbank + 5.081) / 4.4849
    return fbank, audio_info
    
def predict(audio_path, question, model, tokenizer, prompter, device, args):
    instruction = question
    prompt = prompter.generate_prompt(instruction, None)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)


    if (audio_path != 'empty'):
        try:
            cur_audio_input, audio_info = load_audio(audio_path)
        except Exception as e:
            print(f"Fail to load {audio_path}, skip: {e}")
            return None, None
        cur_audio_input = cur_audio_input.unsqueeze(0)
        if torch.cuda.is_available():
            cur_audio_input = cur_audio_input.to(device)
    else:
        cur_audio_input = None
        audio_info = 'Audio is not provided, answer pure language question.'

    if args.num_beams!=0:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            num_beams=args.num_beams, 
            temperature=args.temperature,
            top_p=args.top_p, 
            top_k=args.top_k, 
            max_new_tokens=args.max_new_tokens,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )
    else:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,   
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k, 
            max_new_tokens=args.max_new_tokens,   
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=args.max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)[len(prompt)+6:-4]
    return audio_info, output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_count", type=str, default="0")
    parser.add_argument("--json_input", type=str, required=True)
    parser.add_argument("--json_output", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--num_beams", type=int)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    try:
        gpu_count = int(args.gpu_count)
        available_gpus = get_available_gpus(gpu_count)
        if not available_gpus or len(available_gpus) < gpu_count:
            print(f"Warning: no enough GPU, require {gpu_count}, but {len(available_gpus)} in fact")
            if not available_gpus:
                available_gpus = list(range(min(gpu_count, torch.cuda.device_count())))
        gpu_count = len(available_gpus)
    except ValueError:
        available_gpus = [int(x) for x in args.gpu_count.split(',')]
        gpu_count = len(available_gpus)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in available_gpus])
    print(f"Use GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    with open(args.json_input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if gpu_count == 0 or not torch.cuda.is_available():
        print("Warning: no GPU available, use CPU for inference")
        base_model = "/wangbenyou/xinyuan/models/Llama-2-7b-chat-hf-qformer"
        process_subset(0, data, base_model, args.model_path, args)
    else:
        mp.set_start_method('spawn', force=True)

        chunk_size = len(data) // gpu_count
        remainder = len(data) % gpu_count
        
        data_chunks = []
        start = 0
        for i in range(gpu_count):
            end = start + chunk_size + (1 if i < remainder else 0)
            data_chunks.append(data[start:end])
            start = end
        
        print(f"Data chunks: {[len(chunk) for chunk in data_chunks]}")

        base_model = "/wangbenyou/xinyuan/models/Llama-2-7b-chat-hf-qformer"
        processes = []

        with mp.Pool(processes=gpu_count) as pool:
            torch.cuda.empty_cache() 
            process_func = partial(process_subset, 
                                  base_model=base_model, 
                                  model_path=args.model_path, 
                                  args=args)

            process_args = [(gpu_id, data_chunks[i]) for i, gpu_id in enumerate(range(gpu_count))]

            results = pool.starmap(process_func, process_args)

        all_results = []
        for res in results:
            all_results.extend(res)

        output_dir = os.path.dirname(args.json_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"make output dir: {output_dir}")
        
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        checkpoint_dir = os.path.dirname(args.json_output)
        for gpu in range(gpu_count):
            cp_path = os.path.join(checkpoint_dir, f"checkpoint_gpu_{gpu}.json")
            if os.path.exists(cp_path):
                os.remove(cp_path)
        
        print(f"Save to {args.json_output}")
