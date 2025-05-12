#!/usr/bin/env python
# LLM_judge.py
# -*- coding:utf-8 -*-

import os
import logging
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI, APIError

# Default Configuration
DEFAULT_llm_API_BASE = "YOUR_API_BASE"
DEFAULT_APIKEY = "YOUR_API_KEY"
DEFAULT_llm_MODEL = "gpt-4.1-mini"
DEFAULT_NUM_PROCESSES = 100
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 1024

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_judge")

# --- Core API Call Function (Modified to accept hyperparameters) ---
def call_llm_completion(api_key, api_base_url, model_name, messages, temperature, max_tokens, max_retries=10):
    """
    (Function to call llm API with retries, accepting temperature and max_tokens)
    """
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

def _gpt4_llm_as_judge_one_sample_1_to_10_score_chat_exact_with_meta(args):
    # Unpack arguments (add meta_info, temperature, max_tokens)
    question, references, prediction, meta_info, api_key, api_base_url, model_name, temperature, max_tokens = args

    # Use the first reference answer as Assistant 1
    assistant1_answer = references[0] if references else "N/A"
    # Model prediction as Assistant 2
    assistant2_answer = prediction
    # Audio description
    audio_description = meta_info if meta_info is not None else "N/A"

    # Prompt template for 1-10 scale evaluation (Exactly aligned with score_chat.py, includes Audio Description)
    PROMPT_TEMPLATE = """\
You are a helpful and precise assistant for checking the quality of the answer.
[Detailed Audio Description]
{audio_description}
[Question]
{question}
[The Start of Assistant 1s Answer]
{assistant1_answer}
[The End of Assistant 1s Answer]
[The Start of Assistant 2s Answer]
{assistant2_answer}
[The End of Assistant 2s Answer]
[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.
Assistant 1 represents a reference answer, and Assistant 2 represents the model's prediction.
Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses.
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively.
The two scores are separated by a space.""" # Exactly from score_chat.py

    # Format the evaluation prompt
    evaluation_prompt = PROMPT_TEMPLATE.format(
        audio_description=audio_description,
        question=question,
        assistant1_answer=assistant1_answer,
        assistant2_answer=assistant2_answer
    )
    messages = [{"role": "user", "content": evaluation_prompt}]

    # Call the llm API (pass hyperparameters, remove hardcoded max_tokens)
    output = call_llm_completion(api_key, api_base_url, model_name, messages, temperature, max_tokens)

    # --- Result Handling: Only record raw output and API call status (unchanged) ---
    judge_response = ""
    api_success = 0
    if output is not None:
        judge_response = output
        api_success = 1
        # Log raw response, containing two scores
        logger.debug(f"API call successful. Raw output: '{output}'. Q: {question[:50]}...")
    else:
        logger.error(f"API call failed for sample. Q: {question[:50]}...")
        judge_response = "Error: API call failed."
        api_success = 0

    # Return a dictionary containing the raw response and API status (add meta_info)
    return {
        'question'        : question,
        'references'      : references,
        'model_prediction': prediction,
        'meta_info'       : meta_info,
        'judge_response'  : judge_response,
        'rate_score'      : None,
        'success'         : api_success,
    }


def gpt4_llm_as_judge_1_to_10_score_chat_exact_with_meta(input_data, api_key=DEFAULT_APIKEY,
                                                               api_base_url=DEFAULT_llm_API_BASE,
                                                               model_name=DEFAULT_llm_MODEL,
                                                               num_processes=DEFAULT_NUM_PROCESSES,
                                                               temperature=DEFAULT_TEMPERATURE,
                                                               max_tokens=DEFAULT_MAX_TOKENS):
    # Unpack input_data (add meta_info_list)
    questions, references_list, predictions, meta_info_list = input_data
    num_samples = len(questions)
    # Update length check
    if not (num_samples == len(references_list) == len(predictions) == len(meta_info_list)):
        raise ValueError("(LLM_judge) Input lists (questions, references_list, predictions, meta_info_list) must have the same length.")
    if num_samples == 0:
        logger.warning("Input data is empty for 1-10 scale evaluation (score_chat exact style with meta). Returning empty list.")
        return []

    # Check if references_list contains valid reference answers (unchanged)
    for i, refs in enumerate(references_list):
        if not refs:
            logger.error(f"Sample at index {i} has an empty list of references. Cannot proceed with score_chat exact style prompt.")
            raise ValueError(f"Empty reference list found at index {i}, required for Assistant 1 in score_chat exact style prompt.")
    # Check if meta_info_list contains valid info (e.g., not None)
    if not all(info is not None for info in meta_info_list):
        logger.error("Input meta_info_list contains None values. Cannot proceed.")
        raise ValueError("Input meta_info_list contains None values.")


    actual_num_processes = min(num_processes, num_samples)
    logger.info(f"Using {actual_num_processes} processes for {num_samples} samples (GPT-4 Judge - Temp: {temperature}, MaxTokens: {max_tokens} - score_chat Exact Style with Meta - Raw Output).") # 更新日志

    # Prepare arguments (add temperature, max_tokens)
    map_args = zip(
        questions,
        references_list,
        predictions,
        meta_info_list,
        [api_key] * num_samples,
        [api_base_url] * num_samples,
        [model_name] * num_samples,
        [temperature] * num_samples,
        [max_tokens] * num_samples
    )

    all_details = []
    # Update tqdm description
    desc = "GPT-4 Judge (1-10 Scale - score_chat Exact Style with Meta - Raw Output)"
    with Pool(processes=actual_num_processes) as pool:
        try:
             all_details = list(
                tqdm(
                    # Call the updated internal function
                    pool.imap(_gpt4_llm_as_judge_one_sample_1_to_10_score_chat_exact_with_meta, map_args),
                    total=num_samples,
                    desc=desc
                )
            )
        except Exception as e:
            logger.error(f"Error during multiprocessing ({desc}): {e}", exc_info=True)

    if not all_details:
        logger.warning(f"No samples were successfully processed for {desc}.")
        return []

    # No longer calculate scores, return details with raw responses (unchanged)
    logger.info(f"{desc} finished. Returning {len(all_details)} detailed results with raw judge responses.")
    return all_details
