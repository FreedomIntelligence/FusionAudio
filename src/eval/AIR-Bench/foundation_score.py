import json

with open('/wangbenyou/xinyuan/ECHO-AQA/results/AIR-Bench/foundation_music_echo/t-0.1-alignment-4.1mini.json','r') as f:
    data = json.load(f)

with open('/wangbenyou/xinyuan/ECHO-AQA/data/eval/AIR-Bench/Foundation_meta.json','r') as f:
    ref_data = json.load(f)

ref_map = {}
for ref_item in ref_data:
    audio_id = '/wangbenyou/xinyuan/ECHO-AQA/data/eval/AIR-Bench/Foundation/' + ref_item['task_name'] + "_" + ref_item['dataset_name'] +'/'+ ref_item['path']#[:-4]+".flac"
    ref_map[audio_id] = ref_item

num = 0
num_true = 0
for item in data:
    audio_id = item['audio_id']
    ref_item = ref_map.get(audio_id)
    if not ref_item:
        continue
    gt = None
    if ref_item['choice_a'] == ref_item['answer_gt']:
        gt = 'A'
    elif ref_item['choice_b'] == ref_item['answer_gt']:
        gt = 'B'
    elif ref_item['choice_c'] == ref_item['answer_gt']:
        gt = 'C'
    elif ref_item['choice_d'] == ref_item['answer_gt']:
        gt = 'D'
    if gt is not None:
        if gt == item.get("output"):
            num_true += 1
        num += 1

print(f"acc:{num_true/num}")