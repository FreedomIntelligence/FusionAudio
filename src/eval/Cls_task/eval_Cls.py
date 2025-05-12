# -*- coding: utf-8 -*-
# @Time    : 4/10/23 5:05 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : eval_llm_cla.py

# evaluation classification based on gpt/bert embeddings
import os.path
import openai
import numpy as np
import math
import json
import string
import torch
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report
import collections
import csv
from stats import calculate_stats
from tqdm import tqdm  # 添加tqdm用于进度可视化
import concurrent.futures
import threading
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# openai == 0.28.0

API_BASE_URL = 'YOUR_API_BASE_URL'
API_KEY = 'YOUR_API_KEY'

dataset = 'FSDnoisy18K'
llm_task = 'caption'
text_embed_setting = 'gpt'


num_class = 20
label_dict_path = '/wangbenyou/xinyuan/AudioCaption/data/eval/vocabulary/FSDnoisy18k.csv'
base_path = '/wangbenyou/xinyuan/AudioCaption/src/eval/Cls_task/'
eval_file_list = ['/wangbenyou/xinyuan/ECHO-AQA/results/FSDnoisy18K/FSDnoisy18K_audio-5k/t-0.01.json']


eval_file_list = [ x for x in eval_file_list]
for x in eval_file_list:
    assert os.path.exists(x) == True

device = "cuda" if torch.cuda.is_available() else "cpu"


if text_embed_setting == 'bert':
    bert_mdl_size = 'bert-large-uncased'
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_mdl_size, model_max_length=512)
    bert_model = BertModel.from_pretrained(bert_mdl_size).to(device)
    print(f"Using device: {device}")
else:
    bert_tokenizer = None
    bert_model = None

api_executor = concurrent.futures.ThreadPoolExecutor(max_workers=300)
embed_cache_lock = threading.Lock()

for eval_file in eval_file_list:
    try:
        def get_bert_embedding(input_text):
            if bert_tokenizer is None or bert_model is None:
                raise RuntimeError("BERT model not loaded but BERT embedding was requested.")
            input_text = remove_punctuation_and_lowercase(input_text)
            #print(input_text)
            inputs = bert_tokenizer(input_text, return_tensors="pt")
            if inputs['input_ids'].shape[1] > 512:
                inputs['input_ids'] = inputs['input_ids'][:, :512]
                inputs['token_type_ids'] = inputs['token_type_ids'][:, :512]
                inputs['attention_mask'] = inputs['attention_mask'][:, :512]
            outputs = bert_model(**inputs.to(device))
            last_hidden_states = torch.mean(outputs.last_hidden_state[0], dim=0).cpu().detach().numpy()
            return last_hidden_states

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
        def embedding_with_backoff(**kwargs):
            return openai.Embedding.create(**kwargs)

        def get_gpt_embedding(input_text, mdl_size='text-embedding-3-large'):
            openai.api_base = API_BASE_URL
            openai.api_key = API_KEY
            response = embedding_with_backoff(
                input=input_text,
                model=mdl_size
            )
            embeddings = response['data'][0]['embedding']
            return embeddings

        def cosine_similarity(vector1, vector2):
            dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vector1))
            magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vector2))
            return dot_product / (magnitude1 * magnitude2)

        def remove_punctuation_and_lowercase(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.lower()
            return text

        def gen_cm(all_truth, all_pred, save_name):
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import numpy as np

            # list of label names
            label_names = list(label_dict.keys())

            # generate confusion matrix
            cm = confusion_matrix(all_truth, all_pred)

            # plot confusion matrix as a figure
            plt.imshow(cm, cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(label_names))
            plt.xticks(tick_marks, label_names, rotation=90, fontsize=6)
            plt.yticks(tick_marks, label_names, fontsize=6)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            # add label values to the confusion matrix cells
            for i in range(len(label_names)):
                for j in range(len(label_names)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="white")

            plt.savefig(save_name, dpi=300)

        def make_name_dict(label_csv):
            name_lookup = collections.OrderedDict()
            with open(label_csv, 'r') as f:
                csv_reader = csv.DictReader(f)
                line_count = 0
                for row in csv_reader:
                    name_lookup[row['mid']] = row['display_name']
                    line_count += 1
            return name_lookup

        label_list = np.loadtxt(label_dict_path, delimiter=',', dtype=str)
        print(f"Loaded label list, total {len(label_list)} categories")
        print(label_list)

        # load cached label embedding dict
        os.makedirs(f'{base_path}label_embed_dict', exist_ok=True)
        if os.path.exists(f'{base_path}label_embed_dict/{dataset}_{llm_task}_{text_embed_setting}.json'):
            print("Loading cached label embeddings...")
            with open(f'{base_path}label_embed_dict/{dataset}_{llm_task}_{text_embed_setting}.json', 'r') as f:
                json_str = f.read()
            label_dict = json.loads(json_str, object_pairs_hook=OrderedDict)
            print(f"Loaded {len(label_dict)} label embeddings")
        else:
            print("Creating new label embeddings...")
            label_dict = OrderedDict()
            # Use parallel to get label embeddings
            with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
                future_to_class = {}
                for i in range(label_list.shape[0]):
                    class_name = label_list[i, 1]
                    if text_embed_setting == 'gpt':
                        future = executor.submit(get_gpt_embedding, 'sound of ' + class_name.lower())
                    elif text_embed_setting == 'bert':
                        future = executor.submit(get_bert_embedding, 'sound of ' + class_name.lower())
                    future_to_class[future] = class_name
                
                for future in tqdm(concurrent.futures.as_completed(future_to_class), 
                                   total=len(future_to_class), desc="Creating label embeddings"):
                    class_name = future_to_class[future]
                    try:
                        embedding = future.result()
                        label_dict[class_name] = embedding
                    except Exception as exc:
                        print(f"Error processing {class_name}: {exc}")

            os.makedirs(f'{base_path}label_embed_dict', exist_ok=True)
            with open(f'{base_path}label_embed_dict/{dataset}_{llm_task}_{text_embed_setting}.json', 'w') as f:
                json_str = json.dumps(label_dict)
                f.write(json_str)
            print("Label embeddings saved")

        print(f"Label dictionary contains {len(label_dict.keys())} categories")

        print(f"Loading evaluation file: {eval_file}")
        with open(eval_file, 'r') as fp:
            eval_data = json.load(fp)
        print(f"Evaluation data loaded, total {len(eval_data)} samples")

        os.makedirs(f'{base_path}embedding_cache', exist_ok=True)
        if os.path.exists(f'{base_path}embedding_cache/{dataset}_{llm_task}_{text_embed_setting}.json') == True:
            print("Loading embedding cache...")
            with open(f'{base_path}embedding_cache/{dataset}_{llm_task}_{text_embed_setting}.json') as f:
                embed_cache = f.read()
            embed_cache = json.loads(embed_cache)
            print(f"Loaded {len(embed_cache)} cached embeddings")
        else:
            print("Creating new embedding cache")
            embed_cache = {}

        def get_pred(cur_pred_list, label_dict, mode='accu'):
            # at beginning, all zero scores
            score = np.zeros(num_class)
            label_embed_list = list(label_dict.values())
            
            # Concurrently query missing predictions' embeddings
            missing = []
            futures_map = {}
            with embed_cache_lock:
                for cur_pred in cur_pred_list:
                    if cur_pred not in embed_cache:
                        missing.append(cur_pred)
                
                for pred in missing:
                    if text_embed_setting == 'gpt':
                        futures_map[pred] = api_executor.submit(get_gpt_embedding, pred)
                    else:
                        futures_map[pred] = api_executor.submit(get_bert_embedding, pred)
            
            for pred, future in futures_map.items():
                result = future.result()
                with embed_cache_lock:
                    embed_cache[pred] = result
            
            # Process each prediction text using updated embed_cache
            for cur_pred in cur_pred_list:
                with embed_cache_lock:
                    cur_pred_embed = embed_cache[cur_pred]
                for i in range(num_class):
                    if mode == 'accu':
                        score[i] = score[i] + cosine_similarity(cur_pred_embed, label_embed_list[i])
                    elif mode == 'max':
                        score[i] = max(score[i], cosine_similarity(cur_pred_embed, label_embed_list[i]))
            return score

        # Define function to process a single sample for parallel processing
        def process_sample(i, sample):
            try:
                if llm_task == 'cla':
                    cur_pred_list = sample['pred'].split(': ')[-1].split('; ')
                    cur_pred_list = ['sound of ' + x.lower().lstrip() for x in cur_pred_list]
                elif llm_task == 'caption':
                    cur_pred_list = sample['pred']#.replace('"', '').split('Audio caption')[-1][2:]
                    cur_pred_list = ['sound of ' + cur_pred_list.lower()]
                    #print(f"Processing sample {i}: {cur_pred_list}")

                # Check if ref field exists
                if 'ref' not in sample or sample['ref'] is None:
                    print(f"Warning: Sample {i} has no ref field or ref is None, using empty label")
                    cur_truth_item = ""
                else:
                    # Directly use ref as single label
                    cur_truth_item = sample['ref'].strip()
                
                cur_truth = np.zeros(num_class)
                label_keys = list(label_dict.keys())
                
                try:
                    cur_truth_idx = label_keys.index(cur_truth_item)
                    cur_truth[cur_truth_idx] = 1.0
                except ValueError:
                    print(f"Warning: Cannot find true label '{cur_truth_item}' in dictionary, sample {i}")
                
                cur_pred = get_pred(cur_pred_list, label_dict)
                #print(f"Now idx{i}: pred:{cur_pred}, truth: {cur_truth}")
                return i, cur_pred, cur_truth
            except Exception as e:
                import traceback
                print(f"Error processing sample {i}: {str(e)}")
                print(f"Sample data: {sample}")
                print(traceback.format_exc())
                # Return empty result to avoid interrupting the whole process
                return i, np.zeros(num_class), np.zeros(num_class)

        # Add data validation step
        print("Validating evaluation data format...")
        for i, sample in enumerate(eval_data[:5]):  # Only check the first 5 samples
            if 'pred' not in sample:
                print(f"Warning: Sample {i} is missing 'pred' field")
            if 'ref' not in sample:
                print(f"Warning: Sample {i} is missing 'ref' field")
            if i < 3:  # Print the first 3 samples for debugging
                print(f"Sample {i} content: {sample}")

        num_sample = len(eval_data)
        print('Number of samples: {:d}'.format(num_sample))
        all_pred = np.zeros([num_sample, num_class])
        all_truth = np.zeros([num_sample, num_class])
        
        print("Starting parallel sample processing...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            future_sample = {executor.submit(process_sample, i, eval_data[i]): i 
                              for i in range(num_sample)}
            
            for future in tqdm(concurrent.futures.as_completed(future_sample), 
                                total=num_sample, desc="Processing samples"):
                i, cur_pred, cur_truth = future.result()
                all_pred[i] = cur_pred
                all_truth[i] = cur_truth
        
        print("Sample processing completed")
        
        # Create result save directory
        save_fold = f"{base_path}{'.'.join(eval_file.split('/')[-1].split('.')[:-1])}_{llm_task}_{text_embed_setting}_cla_report"
        os.makedirs(save_fold, exist_ok=True)
        print(f"Saving results to {save_fold}")

        np.save(save_fold + '/all_pred.npy', all_pred)
        np.save(save_fold + '/all_truth.npy', all_truth)
        stats = calculate_stats(all_pred, all_truth)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        np.savetxt(save_fold + '/result_summary.csv', [mAP, mAUC, acc], delimiter=',')
        print(f"Result summary - mAP: {mAP:.4f}, mAUC: {mAUC:.4f}, Accuracy: {acc:.4f}")

        print("Saving embedding cache...")
        embed_cache_json = json.dumps(embed_cache)
        save_cache_path = f'{base_path}embedding_cache/{dataset}_{llm_task}_{text_embed_setting}.json'
        with open(save_cache_path, 'w') as f:
            f.write(embed_cache_json)
        print(f"Embedding cache saved to: {save_cache_path}")
    except Exception as e:
        import traceback
        print(f"Error processing file {eval_file}: {str(e)}")
        print(traceback.format_exc())

api_executor.shutdown(wait=True)
print("Evaluation completed")