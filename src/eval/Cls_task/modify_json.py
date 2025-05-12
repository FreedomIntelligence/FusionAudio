import json
import os
import argparse

parser = argparse.ArgumentParser(description="Script to modify JSON files")
parser.add_argument("--input_dir", required=True, help="Input JSON file path")
parser.add_argument("--output_dir", required=True, help="Output JSON file path")
parser.add_argument("--ref_dir", required=True , help="Reference JSON file path")
args = parser.parse_args()

input_file_path = args.input_dir
output_file_path = args.output_dir

reference_file_path = "/wangbenyou/xinyuan/ECHO-AQA/data/eval/instruction_ref/TAU_urban_mobile_test.json" 

# Check if file exists
if not os.path.exists(input_file_path):
    print(f"Error: {input_file_path} does not exist.")
    exit(1)

try:
    # Read original file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        stage5_data = json.load(f)
    
    # Load reference data if file exists
    ref_data = {}
    if os.path.exists(reference_file_path):
        with open(reference_file_path, 'r', encoding='utf-8') as f:
            esc50_data = json.load(f)
            
        # Create mapping from audio ID to output
        for item in esc50_data:
            if 'audio_id' in item and 'output' in item:
                ref_data[item['audio_id']] = item['output']
    else:
        print(f"Warning: Reference file {reference_file_path} does not exist, only key names will be modified.")
    
    # Modify data
    modified_count = 0
    ref_added_count = 0
    
    for item in stage5_data:
        # Change 'output' to 'pred'
        if 'output' in item:
            item['pred'] = item.pop('output')
            modified_count += 1
        
        # Add reference output
        audio_id = item.get('audio_id')
        if audio_id and audio_id in ref_data:
            item['ref'] = ref_data[audio_id]
            ref_added_count += 1
    
    # Save modified file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(stage5_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed!")
    print(f"- Changed {modified_count} 'output' keys to 'pred'")
    print(f"- Added {ref_added_count} reference 'ref' entries")
    print(f"- Modified file saved to: {output_file_path}")

except Exception as e:
    print(f"Error occurred during processing: {str(e)}")
    exit(1)
