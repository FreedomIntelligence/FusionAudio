# -*- coding: utf-8 -*-
import json
import argparse
import re 
import ast
import logging
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI, APIError

def prompter(input_item):
    system_prompt = 'There is a question "QUESTION", and one response is "ANSWER". Among the following choices, which option best matches this response? You may only response with the letter of the option. If the letter of an option has already appeared in the Answer, then simply return that letter. If there are no matching options, then choose one at random. The following lines are the options.'
    
    instruction_text = input_item['instruction']
    response = input_item['output']
    if response is None:
        return None

    instruction_prefix = "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D."

    
    question = instruction_text.replace(instruction_prefix, "").strip()

    options_match = question.split("Options: ")[-1].strip()
    choices = [item.strip() for item in options_match[1:-1].split(",")]

    choice_a = choices[0] if len(choices) > 0 else None
    choice_b = choices[1] if len(choices) > 1 else None
    choice_c = choices[2] if len(choices) > 2 else None
    choice_d = choices[3] if len(choices) > 3 else None

    if not question or not choice_a or not choice_b: 
         print(f"Warning: Could not parse instruction correctly: {instruction_text}")

    system_prompt = system_prompt.replace("ANSWER", response).replace("QUESTION", question)
    attention = 'To reiterate, the response you provide to me must be a single letter, either A, B, C, or D.'

    content_parts = [system_prompt]
    if choice_a:
        content_parts.append(f"A. {choice_a}")
    if choice_b:
        content_parts.append(f"B. {choice_b}")
    if choice_c:
        content_parts.append(f"C. {choice_c}")
    if choice_d:
        content_parts.append(f"D. {choice_d}")
        
    content_parts.append(attention)
    
    content = '\n'.join(content_parts)
    return content

DEFAULT_llm_API_BASE = "YOUR_API_BASE"
DEFAULT_APIKEY = "YOUR_API_KEY"
DEFAULT_llm_MODEL = "gpt-4.1-mini"
DEFAULT_NUM_PROCESSES = 100
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gama_alignment")

def call_llm_completion(api_key, api_base_url, model_name, messages, temperature, max_tokens, max_retries=10):
    api_base_url = api_base_url or DEFAULT_llm_API_BASE
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                temperature=temperature
            )
            output = completion.choices[0].message.content.strip()
            return output
        except APIError as e:
            retries += 1
            logger.warning(f"API Error: {e}. Retrying ({retries}/{max_retries})...")
            if retries >= max_retries:
                logger.error(f"API call failed after {max_retries} retries.")
                return None
        except Exception as e:
            retries += 1
            logger.warning(f"An unexpected error occurred during API call: {e}. Retrying ({retries}/{max_retries})...")
            if retries >= max_retries:
                logger.error(f"API call failed after {max_retries} retries due to unexpected errors.")
                return None
    return None

def api_call(content,
             api_key=DEFAULT_APIKEY,
             api_base_url=DEFAULT_llm_API_BASE,
             model_name=DEFAULT_llm_MODEL,
             temperature=DEFAULT_TEMPERATURE,
             max_tokens=DEFAULT_MAX_TOKENS):
    messages = [{"role": "user", "content": content}]
    output = call_llm_completion(api_key, api_base_url, model_name, messages, temperature, max_tokens)
    return output if output is not None else "Error: API call failed."

def _api_call_one_sample(args):
    input_item, api_key, api_base_url, model_name, temperature, max_tokens = args
    content = prompter(input_item)
    if content is None:
        return None
    response = api_call(content, api_key, api_base_url, model_name, temperature, max_tokens)
    output_item = {
        "audio_id": input_item["audio_id"],
        "instruction": input_item["instruction"],
        "input": "",
        "dataset": input_item["dataset"],
        "task": input_item["task"],
        "output": response,
    }
    return output_item

def main(args):
    with open(args.input_file, "r") as f:
        # sound
        #input_data = json.load(f)[:896] # AudioGrounding
        #input_data = json.load(f)[896:1896] # vocal_sound_classification
        #input_data = json.load(f)[1896:3766] # Acoustic_Scene_Classification
        #input_data = json.load(f)[3766:] # Sound_AQA

        # music
        #input_data = json.load(f)[:2000] # Music_Instruments_Classfication
        #input_data = json.load(f)[2000:4000] # Music_Genre_Recognition
        #input_data = json.load(f)[4000:5000] # Music_Midi_Pitch_Analysis
        #input_data = json.load(f)[5000:6000] # Music_Midi_Velocity_Analysis
        #input_data = json.load(f)[6000:6814] # Music_AQA
        input_data = json.load(f)[6814:] # Music_Mood_Recognition,

    output_data = []
    num_samples = len(input_data)
    actual_num_processes = min(DEFAULT_NUM_PROCESSES, num_samples)
    logger.info(f"Using {actual_num_processes} processes for {num_samples} samples (LLM API alignment).")

    map_args = [
        (item, DEFAULT_APIKEY, DEFAULT_llm_API_BASE, DEFAULT_llm_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS)
        for item in input_data
    ]

    desc = "LLM API Alignment"
    with Pool(processes=actual_num_processes) as pool:
        try:
            for output_item in tqdm(pool.imap(_api_call_one_sample, map_args), total=num_samples, desc=desc):
                if output_item is not None:
                    output_data.append(output_item)
        except Exception as e:
            logger.error(f"Error during multiprocessing ({desc}): {e}", exc_info=True)

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gama Alignment")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/wangbenyou/xinyuan/ECHO-AQA/results/AIR-Bench/foundation_music_echo/t-0.1.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/wangbenyou/xinyuan/ECHO-AQA/results/AIR-Bench/foundation_echo/t-0.1-alignment-4.1mini.json",
        help="Path to the output JSON file",
    )
    args = parser.parse_args()
    main(args)