#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ClothoAQA_eval.py
import os
import json
import random
import logging
from collections import defaultdict
# openai == 1.78.0
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(filepath):
    """Safely load a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file: {filepath} - {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

# --- Modified align_data function ---
def align_data(reference_data, prediction_data):
    """
    Align reference answers and model predictions based on audio_id and instruction.
    Now aggregates multiple reference answers for the same (audio_id, instruction) into a list.
    """
    aligned_data = []
    predictions_map = {}
    missing_pred_keys = 0
    duplicate_pred_keys = 0
    for i, pred_item in enumerate(prediction_data):
        audio_id = pred_item.get('audio_id')
        instruction = pred_item.get('instruction')
        output = pred_item.get('output')
        if audio_id and instruction and output is not None:
            key = (audio_id, instruction)
            if key in predictions_map:
                 duplicate_pred_keys += 1
            predictions_map[key] = output
        else:
            missing_pred_keys += 1
            logging.warning(f"Prediction item at index {i} missing 'audio_id', 'instruction', or 'output'. Skipping. Item: {pred_item}")
    if missing_pred_keys > 0:
        logging.warning(f"Total prediction items skipped due to missing keys: {missing_pred_keys}")
    if duplicate_pred_keys > 0:
        logging.warning(f"Found {duplicate_pred_keys} duplicate prediction keys (audio_id, instruction). Used the last occurrence for each.")


    references_grouped = defaultdict(list)
    missing_ref_keys = 0
    for i, ref_item in enumerate(reference_data):
        audio_id = ref_item.get('audio_id')
        instruction = ref_item.get('instruction')
        reference_answer = ref_item.get('output')
        if audio_id and instruction and reference_answer is not None:
            key = (audio_id, instruction)
            references_grouped[key].append(reference_answer)
        else:
             missing_ref_keys += 1
             logging.warning(f"Reference item at index {i} missing 'audio_id', 'instruction', or 'output'. Skipping. Item: {ref_item}")
    if missing_ref_keys > 0:
         logging.warning(f"Total reference items skipped due to missing keys: {missing_ref_keys}")
    logging.info(f"Grouped references for {len(references_grouped)} unique (audio_id, instruction) keys.")


    no_match_found = 0
    for key, ref_list in references_grouped.items():
        audio_id, instruction = key
        model_prediction = predictions_map.get(key)

        if model_prediction is not None:
            aligned_data.append({
                "question": instruction,    
                "references": ref_list,     
                "prediction": model_prediction, 
                "audio_id": audio_id,      
                "key": key            
            })
        else:
            no_match_found += 1
            logging.debug(f"No prediction found for reference key: {key}. Skipping this group of references.")

    if no_match_found > 0:
         logging.warning(f"Total reference groups skipped because no matching prediction was found: {no_match_found}")

    logging.info(f"Data alignment and grouping completed. Found {len(aligned_data)} groups with matching predictions.")
    return aligned_data

class Clotho_aqa_test_dataset(object):
    """
    Class to handle evaluation (compatible with previous version, but data structure handled has changed).
    """
    def __init__(self, number_of_samples=-1, sample_seed=42):
        self.number_of_samples = number_of_samples
        self.sample_seed = sample_seed
        logging.info(f"Evaluator initialized. Sample limit: {'All' if number_of_samples == -1 else number_of_samples}. Sample seed: {sample_seed}.")

    def select_samples(self, aligned_grouped_data):
        """
        Select and shuffle sample groups based on the quantity set during initialization.
        """
        if self.number_of_samples != -1 and self.number_of_samples < len(aligned_grouped_data):
            logging.info(f"Shuffling and selecting {self.number_of_samples} out of {len(aligned_grouped_data)} aligned sample groups.")
            random.seed(self.sample_seed)
            selected_data = random.sample(aligned_grouped_data, self.number_of_samples)
        else:
            logging.info(f"Using all {len(aligned_grouped_data)} aligned sample groups.")
            selected_data = aligned_grouped_data

        logging.info('Number of sample groups selected for evaluation: {}'.format(len(selected_data)))
        return selected_data

    def compute_score(self, data_for_scoring, metrics=None):
        supported_metrics = ['llm_judge_graded']
        if metrics not in supported_metrics:
            logging.error(f"Invalid metrics specified: '{metrics}'. Supported metrics are: {supported_metrics}")
            return {"error": f"Invalid metrics: '{metrics}'. Supported: {supported_metrics}"}

        if not data_for_scoring:
             logging.error("No data provided for scoring.")
             return {"error": "No data with model predictions provided."}

        # --- Extract data needed for evaluation (note references is a list of lists) ---
        questions   = [item.get("question") for item in data_for_scoring]
        # The value corresponding to the 'references' key is itself a list, so the result is a list of lists
        references_list = [item.get("references") for item in data_for_scoring]
        predictions = [item.get("prediction") for item in data_for_scoring]

        # Check if extracted data is valid
        if not (len(questions) == len(references_list) == len(predictions) == len(data_for_scoring)):
             logging.error("Data extraction failed or resulted in inconsistent lengths.")
             return {"error": "Failed to extract data correctly for scoring."}
        if not all(isinstance(ref, list) for ref in references_list):
            logging.error("Internal structure error: 'references' field should contain lists.")
            for i, ref in enumerate(references_list):
                 if not isinstance(ref, list):
                      logging.error(f"Sample at index {i} has non-list in 'references': {ref}")
                      break
            return {"error": "Invalid data structure for references."}

        logging.info(f"Prepared {len(questions)} sample groups for evaluation using metric: {metrics}")

        # --- Select evaluation method based on metrics parameter ---
        input_lists = [questions, references_list, predictions]

        # --- LLM Judge Evaluation ---
        try:
            if metrics == 'llm_judge_graded':
                logging.info("Using LLM Judge (Graded - Multi-Ref via LLM_judge.py)...")
                from LLM_judge import llm_as_judge
                eval_func = llm_as_judge
            else:
                 raise ValueError(f"Unsupported metric '{metrics}' reached evaluation stage.")

            # Call the evaluation function
            judge_results, all_details = eval_func(
                input_data=input_lists,
            )
            return {metrics: judge_results, 'details': all_details}

        except ImportError:
            logging.error("Could not import evaluation functions from LLM_judge.py.")
            return {"error": "Failed to import LLM judge from LLM_judge.py."}
        except Exception as e:
             logging.error(f"Error during LLM Judge ({metrics}) evaluation: {e}", exc_info=True)
             return {"error": f"LLM Judge ({metrics}) evaluation failed: {e}"}


if __name__ == '__main__':
    reference_file = '/wangbenyou/xinyuan/AudioCaption/data/eval/clotho_aqa_test.json'

    output_dir = '/wangbenyou/xinyuan/ECHO-AQA/results/ClothoQA/Clothoqa_echo/'
    prediction_file = output_dir+'t-0.1.json'
    os.makedirs(output_dir, exist_ok=True)
    # Update output filenames
    results_output_file = os.path.join(output_dir, 'Clotho_aqa_llm_judge_results.json')
    details_output_file = os.path.join(output_dir, 'Clotho_aqa_llm_judge_details.json')

    logging.info(f"Loading reference data from: {reference_file}")
    reference_data = load_json(reference_file)
    logging.info(f"Loading prediction data from: {prediction_file}")
    prediction_data = load_json(prediction_file)

    if reference_data is None or prediction_data is None:
        logging.error("Failed to load necessary data files. Exiting.")
        exit(1)

    # --- Align and Group Data ---
    logging.info("Aligning and grouping reference/prediction data...")
    # aligned_samples now contains grouped data
    aligned_samples_grouped = align_data(reference_data, prediction_data)

    if not aligned_samples_grouped:
        logging.error("No sample groups were successfully aligned. Cannot proceed. Check formats and keys ('audio_id', 'instruction', 'output').")
        exit(1)

    # --- Initialize Evaluator and Select Sample Groups ---
    evaluator = Clotho_aqa_test_dataset(number_of_samples=-1)
    # data_for_scoring is now a list containing grouped information
    data_for_scoring = evaluator.select_samples(aligned_samples_grouped)

    if not data_for_scoring:
        logging.error("No sample groups selected for evaluation. Exiting.")
        exit(1)

    all_results = {}

    print("\n--- Computing score using LLM Judge (Graded - Multi-Reference) ---")
    results = evaluator.compute_score(
        data_for_scoring=data_for_scoring,
        metrics='llm_judge_graded'
    )
    # Check if the returned results contain error information
    if 'error' in results:
        print(f"Evaluation Error: {results['error']}")
        logging.error(f"Evaluation failed: {results['error']}")
    else:
        print("Results:", results.get('llm_judge_graded', 'No score returned'))
        all_results['results'] = results
        # Save detailed results
        if 'details' in results:
            try:
                with open(details_output_file, 'w', encoding='utf-8') as f:
                    json.dump(results['details'], f, indent=4, ensure_ascii=False)
                logging.info(f"Detailed evaluation results saved to: {details_output_file}")
            except Exception as e:
                logging.error(f"Failed to save detailed results: {e}")

    # --- Save final summary results ---
    summary_results_to_save = {}
    # Only add to summary if scores were successfully obtained
    if 'results' in all_results and 'llm_judge_graded' in all_results['results']:
        summary_results_to_save['llm_judge_graded'] = all_results['results']['llm_judge_graded']

    if summary_results_to_save:
        try:
            with open(results_output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_results_to_save, f, indent=4, ensure_ascii=False)
            logging.info(f"Summary evaluation results saved to: {results_output_file}")
        except Exception as e:
            logging.error(f"Failed to save summary results: {e}")
    else:
        logging.warning("No valid summary results were obtained to save.")

    logging.info("Evaluation script finished.")