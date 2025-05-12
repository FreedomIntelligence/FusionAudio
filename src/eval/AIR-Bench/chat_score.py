#!/usr/bin/env python
# -*- coding:utf-8 -*-
# gama_eval.py
import os
import json
import random
import logging
from collections import defaultdict 

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

def align_data(reference_data, prediction_data):
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

    references_grouped = defaultdict(lambda: {'refs': [], 'meta': None})
    missing_ref_keys = 0
    for i, ref_item in enumerate(reference_data):
        audio_id = ref_item.get('audio_id')
        instruction = ref_item.get('instruction')
        reference_answer = ref_item.get('output')
        meta_info = ref_item.get('meta_info')

        # Check if key fields exist
        if audio_id and instruction and reference_answer is not None and meta_info is not None:
            key = (audio_id, instruction)
            references_grouped[key]['refs'].append(reference_answer)
            # Assume meta_info is the same for the same key, store only once
            if references_grouped[key]['meta'] is None:
                references_grouped[key]['meta'] = meta_info
            elif references_grouped[key]['meta'] != meta_info:
                 # If meta_info is inconsistent, log a warning but continue using the first one found
                 logging.warning(f"Inconsistent meta_info found for key {key}. Using the first one encountered.")
        else:
             missing_ref_keys += 1
             # Update warning message to include meta_info
             logging.warning(f"Reference item at index {i} missing 'audio_id', 'instruction', 'output', or 'meta_info'. Skipping. Item: {ref_item}")
    if missing_ref_keys > 0:
         logging.warning(f"Total reference items skipped due to missing keys: {missing_ref_keys}")
    logging.info(f"Grouped references and meta_info for {len(references_grouped)} unique (audio_id, instruction) keys.")


    no_match_found = 0
    for key, data_group in references_grouped.items():
        audio_id, instruction = key
        ref_list = data_group['refs']
        meta_info = data_group['meta']
        model_prediction = predictions_map.get(key)

        if model_prediction is not None:
            # If a matching prediction is found, add to the aligned data list
            aligned_data.append({
                "question": instruction,
                "references": ref_list,
                "prediction": model_prediction,
                "meta_info": meta_info,
                "audio_id": audio_id,
                "key": key
            })
        else:
            # If no prediction is found, log a warning
            no_match_found += 1
            logging.debug(f"No prediction found for reference key: {key}. Skipping this group of references.")

    if no_match_found > 0:
         logging.warning(f"Total reference groups skipped because no matching prediction was found: {no_match_found}")

    logging.info(f"Data alignment and grouping completed. Found {len(aligned_data)} groups with matching predictions.")
    return aligned_data # 

class clotho_aqa_test_dataset(object):
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

    # --- Modified compute_score function (1-10 scale, aligned with score_chat structure, includes meta_info) ---
    def compute_score(self, data_for_scoring, metrics=None):
        # Update supported metric
        supported_metrics = ['gpt4_llm_judge_1_to_10_score_chat_exact_with_meta']
        if metrics not in supported_metrics:
            logging.error(f"Invalid metrics specified: '{metrics}'. Supported metrics are: {supported_metrics}")
            return {"error": f"Invalid metrics: '{metrics}'. Supported: {supported_metrics}"}

        if not data_for_scoring:
             logging.error("No data provided for scoring.")
             return {"error": "No data with model predictions provided."}

        # --- Extract data needed for evaluation (add meta_info) ---
        questions       = [item.get("question") for item in data_for_scoring]
        references_list = [item.get("references") for item in data_for_scoring]
        predictions     = [item.get("prediction") for item in data_for_scoring]
        meta_info_list  = [item.get("meta_info") for item in data_for_scoring]

        # Check if extracted data is valid (add meta_info_list check)
        if not (len(questions) == len(references_list) == len(predictions) == len(meta_info_list) == len(data_for_scoring)):
             logging.error("Data extraction failed or resulted in inconsistent lengths.")
             return {"error": "Failed to extract data correctly for scoring."}
        if not all(isinstance(ref, list) for ref in references_list):
            logging.error("Internal structure error: 'references' field should contain lists.")
            for i, ref in enumerate(references_list):
                 if not isinstance(ref, list):
                      logging.error(f"Sample at index {i} has non-list in 'references': {ref}")
                      break
            return {"error": "Invalid data structure for references."}
        # Check if meta_info_list contains valid info (e.g., not None)
        if not all(info is not None for info in meta_info_list):
            logging.error("Internal structure error: 'meta_info' field should not be None.")
            for i, info in enumerate(meta_info_list):
                 if info is None:
                      logging.error(f"Sample at index {i} has None in 'meta_info'.")
                      break
            return {"error": "Invalid data structure for meta_info."}


        logging.info(f"Prepared {len(questions)} sample groups for evaluation using metric: {metrics}")

        # --- Select evaluation method based on metrics parameter ---
        # Add meta_info_list to input_lists
        input_lists = [questions, references_list, predictions, meta_info_list]

        all_details = []
        try:
            # Update metric name check and function call
            if metrics == 'gpt4_llm_judge_1_to_10_score_chat_exact_with_meta':
                # Update log message
                logging.info("Calling llm GPT-4 Judge (1-10 Scale - score_chat Exact Style with MetaInfo - Raw Output)...")
                # Call the updated judge function
                from LLM_judge import gpt4_llm_as_judge_1_to_10_score_chat_exact_with_meta
                # Call does not use temperature and max_tokens arguments, will use defaults from LLM_judge.py (1.0, 1024)
                all_details = gpt4_llm_as_judge_1_to_10_score_chat_exact_with_meta(input_data=input_lists)
                score_range = (1, 10)
            else:
                 raise ValueError(f"Unsupported metric '{metrics}' reached evaluation stage.")

            if not all_details:
                 logging.warning(f"LLM_judge returned no details for metric {metrics}.")
                 return {"error": f"LLM Judge did not return any results for {metrics}."}

            # --- Parse scores here (1-10 scale, extract the second score) ---
            parsed_scores = []
            parse_success_count = 0
            for i, detail in enumerate(all_details):
                raw_response = detail.get('judge_response', '')
                parsed_score = None
                parse_successful = False
                if detail.get('success') == 1:
                    try:
                        # Try to extract two numbers separated by space from raw response
                        parts = raw_response.split()
                        if len(parts) == 2:
                            # We care about the second score (Assistant 2 / Model Prediction)
                            score_str = parts[1]
                            score = float(score_str)
                            # Check if score is within expected range (1-10)
                            if score_range[0] <= score <= score_range[1]:
                                parsed_score = score
                                parse_successful = True
                                parse_success_count += 1
                            else:
                                logging.warning(f"Parsed second score {score} out of range {score_range} from response: '{raw_response}'. Sample index: {i}")
                        else:
                             logging.warning(f"Could not find two space-separated scores in judge response: '{raw_response}'. Sample index: {i}")
                    except ValueError:
                        logging.warning(f"Could not parse float from second part of judge response: '{raw_response}'. Sample index: {i}")
                    except Exception as e:
                        logging.error(f"Unexpected error parsing judge response: '{raw_response}'. Error: {e}. Sample index: {i}")
                else:
                    logging.warning(f"Skipping score parsing due to API failure for sample index: {i}. Response: '{raw_response}'")

                # Update the entry in all_details to include the parsed score and status
                detail['rate_score'] = parsed_score
                detail['parse_successful'] = parse_successful
                if parsed_score is not None:
                    parsed_scores.append(parsed_score)

            # --- Calculate final score and success rate (logic unchanged) ---
            if not all_details:
                 final_score = 0
                 parse_success_rate = 0
            else:
                 total_samples = len(all_details)
                 parse_success_rate = parse_success_count / total_samples if total_samples > 0 else 0
                 if parsed_scores:
                     avg_raw_score = sum(parsed_scores) / len(parsed_scores)
                     # Map 1-10 scale to 10-100 scale (multiply by 10)
                     final_score = avg_raw_score * 10
                 else:
                     final_score = 0

            judge_results = {'judge_score': final_score, 'parse_success_rate': parse_success_rate}
            logging.info(f"Score parsing complete. Final Score ({metrics}, 10-100 scale): {final_score:.2f}, Parse Success Rate: {parse_success_rate:.2%}")

            return {metrics: judge_results, 'details': all_details}

        except ImportError:
            logging.error("Could not import evaluation functions from LLM_judge.py.")
            return {"error": "Failed to import llm GPT-4 judge from LLM_judge.py."}
        except Exception as e:
             logging.error(f"Error during llm GPT-4 ({metrics}) judging or score parsing: {e}", exc_info=True)
             return {"error": f"llm GPT-4 ({metrics}) judging or parsing failed: {e}"}


# --- Usage Example (main logic modified: metric name, filenames) ---
if __name__ == '__main__':
    reference_file = '/wangbenyou/xinyuan/ECHO-AQA/data/eval/AIR-Bench/chat_sound_music_input.json'

    output_dir = '/wangbenyou/xinyuan/ECHO-AQA/results/AIR-Bench/chat_sound_music_echo/'
    prediction_file = output_dir+ 't-0.1.json'
    os.makedirs(output_dir, exist_ok=True)
    # Update output filenames to reflect the new scoring method (gpt4, score_chat exact style with meta)
    metric_name = 'gpt4_llm_judge_1_to_10_score_chat_exact_with_meta'
    results_output_file = os.path.join(output_dir, f'clotho_aqa_{metric_name}_results.json')
    details_output_file = os.path.join(output_dir, f'clotho_aqa_{metric_name}_details.json')


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
    evaluator = clotho_aqa_test_dataset(number_of_samples=-1)
    # data_for_scoring is now a list containing grouped information
    data_for_scoring = evaluator.select_samples(aligned_samples_grouped)

    if not data_for_scoring:
        logging.error("No sample groups selected for evaluation. Exiting.")
        exit(1)


    all_results = {}
    # Update print message
    print(f"\n--- Computing score using llm GPT-4 ({metric_name}) ---")
    # compute_score now handles score parsing
    results = evaluator.compute_score(
        data_for_scoring=data_for_scoring,
        metrics=metric_name
    )
    # Check if the returned results contain error information
    if 'error' in results:
        print(f"Evaluation Error: {results['error']}")
        logging.error(f"Evaluation failed: {results['error']}")
    else:
        # Results are now nested under the metrics key
        scores = results.get(metric_name, {})
        print(f"Results: Score (10-100 scale)={scores.get('judge_score', 'N/A'):.2f}, Parse Success Rate={scores.get('parse_success_rate', 'N/A'):.2%}")
        all_results[metric_name] = scores
        # Save detailed results (including raw response and parsed score)
        if 'details' in results:
            try:
                with open(details_output_file, 'w', encoding='utf-8') as f:
                    json.dump(results['details'], f, indent=4, ensure_ascii=False)
                logging.info(f"Detailed evaluation results saved to: {details_output_file}")
            except Exception as e:
                logging.error(f"Failed to save detailed results: {e}")

    # --- Save final summary results (now saves results for only one metric) ---
    summary_results_to_save = {}
    # Only add to summary if scores were successfully obtained
    if metric_name in all_results:
        summary_results_to_save[metric_name] = all_results[metric_name]

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