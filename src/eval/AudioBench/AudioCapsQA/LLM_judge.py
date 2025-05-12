#!/usr/bin/env python
# LLM_judge.py
# -*- coding:utf-8 -*-

import os
import logging
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI, APIError

# Default Configuration
DEFAULT_NUM_PROCESSES = 313

DEFAULT_API_BASE = "YOUR_API_BASE"
DEFAULT_APIKEY = "YOUR_API_KEY"
DEFAULT_MODEL = "gpt-4.1-mini"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_judge")

# --- Core API Call Function ---
def call_llm_api(api_key, api_base_url, model_name, messages, max_retries=10):
    api_base_url = api_base_url or DEFAULT_API_BASE
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
                max_tokens=512,
                n=1,
                temperature=0.0
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

def _format_references(reference_list):
    """Formats a list of reference answers into a string suitable for the prompt."""
    if not reference_list:
        return "N/A"
    if len(reference_list) == 1:
        return reference_list[0]
    else:
        # If multiple references, display them clearly with numbering and separators
        formatted = "Multiple reference answers are provided:\n"
        formatted += "\n---\n".join([f"Reference {i+1}: {ref}" for i, ref in enumerate(reference_list)])
        return formatted

def _llm_as_judge_one_sample_graded(args):
    """
    Internal function for graded (0-5) evaluation of a single sample
    """
    # Unpack arguments
    question, references, prediction, api_key, api_base_url, model_name = args

    # Format reference answers
    reference_text = _format_references(references)

    # Prompt template for graded evaluation
    PROMPT_TEMPLATE = """\
[Reference Answers]
{reference_text}

[Model Answer]
{prediction}

[Question]
{question}

[Task]
Rate the model's answer based on its alignment with the provided reference answer(s), focusing on accuracy and relevance. Consider all reference answers if multiple are given. Please be critical on the details. If the model response is something like 'cannot decide', please rate as 0.
Criteria: Assess if the model's response mirrors the content, accuracy, and relevance of *any* of the provided reference answers.
Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference(s).
Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference(s).
Score2: The answer recognizes the topic but diverges significantly from the reference(s) in accuracy or relevance.
Score3: The answer aligns with the reference(s) generally but lacks detail or precise accuracy in some aspects.
Score4: The answer is mostly accurate and relevant, closely following at least one reference but could be clearer or more detailed.
Score5: The answer is highly accurate, detailed, and matches the essence and detail of at least one reference answer perfectly.

Your response should be formatted as follows:
Explanation: (Provide a concise explanation of your rating, comparing the reference answer(s) with the model's response. "The reference answer(s) state [XXX], while the model's answer is [YYY]. I think ...")
Rating: (int)"""

    # Format the evaluation prompt
    evaluation_prompt = PROMPT_TEMPLATE.format(reference_text=reference_text, prediction=prediction, question=question)
    messages = [{"role": "user", "content": evaluation_prompt}]

    # Call the LLM API
    output = call_llm_api(api_key, api_base_url, model_name, messages)

    # Result parsing
    rate_score = 0.0
    success = 0
    judge_response = ""
    if output:
        judge_response = output
        try:
            parts = output.split()
            if parts:
                last_part = parts[-1].strip('()')
                rate_score = float(last_part)
                if 0 <= rate_score <= 5:
                     success = 1
                else:
                     logger.warning(f"Parsed score {rate_score} out of expected range (0-5). Q: {question[:50]}...")
                     rate_score = 0.0
                     success = 0
            else:
                logger.warning(f"Judge response was empty after splitting. Q: {question[:50]}...")
                success = 0
        except ValueError:
            last_part_for_log = parts[-1] if parts else "N/A"
            logger.warning(f"Could not parse float from last part: '{last_part_for_log}'. Full output: '{output}'. Q: {question[:50]}...")
            rate_score = 0.0
            success = 0
        except Exception as e:
            logger.error(f"Error parsing judge response: {e}. Output: '{output}'. Q: {question[:50]}...")
            rate_score = 0.0
            success = 0
    else:
        logger.error(f"API call failed for sample. Q: {question[:50]}...")
        judge_response = "Error: API call failed."
        rate_score = 0.0
        success = 0

    # Return results
    return {
        'question'        : question,
        'references'      : references,
        'model_prediction': prediction,
        'judge_response'  : judge_response,
        'rate_score'      : rate_score,
        'success'         : success,
    }

# --- Public Evaluation Function ---
def llm_as_judge(input_data, api_key=DEFAULT_APIKEY,
                              api_base_url=DEFAULT_API_BASE,
                              model_name=DEFAULT_MODEL,
                              num_processes=DEFAULT_NUM_PROCESSES):

    questions, references_list, predictions = input_data
    num_samples = len(questions)
    if not (num_samples == len(references_list) == len(predictions)):
        raise ValueError("(LLM_judge) Input lists (questions, references_list, predictions) must have the same length.")
    if num_samples == 0:
        logger.warning("Input data is empty for graded evaluation. Returning default results.")
        return {'judge_score': 0, 'success_rate': 0}, []

    actual_num_processes = min(num_processes, num_samples)
    logger.info(f"Using {actual_num_processes} processes for {num_samples} samples (Graded Evaluation - Multi-Ref).")

    # Prepare arguments
    map_args = zip(
        questions,
        references_list,
        predictions,
        [api_key] * num_samples,
        [api_base_url] * num_samples,
        [model_name] * num_samples
    )

    all_details = []
    desc = "LLM Judge (Graded - Multi-Ref)"
    with Pool(processes=actual_num_processes) as pool:
        try:
             all_details = list(
                tqdm(
                    pool.imap(_llm_as_judge_one_sample_graded, map_args),
                    total=num_samples,
                    desc=desc
                )
            )
        except Exception as e:
            logger.error(f"Error during multiprocessing ({desc}): {e}", exc_info=True)

    if not all_details:
        logger.warning(f"No samples were successfully processed for {desc}.")
        return {'judge_score': 0, 'success_rate': 0}, []

    # Calculate results
    all_scores = [detail['rate_score'] for detail in all_details]
    total_success = sum([detail['success'] for detail in all_details])
    success_rate = total_success / len(all_details) if all_details else 0
    avg_score = (sum(all_scores) / len(all_scores)) * 20 if all_scores else 0

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate}
    logger.info(f"{desc} Results - Score (0-100): {avg_score:.2f}, Success Rate: {success_rate:.2%}")

    return judge_results, all_details
