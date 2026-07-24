import torch
import os
import argparse
import re
from collections import OrderedDict

def merge_checkpoints(checkpoint_dir, output_file, num_to_merge):
    """
    Merges the last N model checkpoints from a directory using Simple Moving Average.
    """
    try:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            print(f"Error: No checkpoint files (.pt) found in directory '{checkpoint_dir}'")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{checkpoint_dir}' does not exist.")
        return

    def sort_key(filename):
        match = re.search(r'step_(\d+)\.pt', filename)
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=sort_key)

    if num_to_merge > 0 and len(checkpoint_files) >= num_to_merge:
        checkpoints_to_merge = checkpoint_files[-num_to_merge:]
        print(f"Selected the last {len(checkpoints_to_merge)} checkpoints for merging.")
    else:
        checkpoints_to_merge = checkpoint_files
        print(f"Warning: Number to merge ({num_to_merge}) is >= total checkpoints ({len(checkpoint_files)}). Merging all files.")

    checkpoints_to_merge_paths = [os.path.join(checkpoint_dir, f) for f in checkpoints_to_merge]
    
    summed_state_dict = torch.load(checkpoints_to_merge_paths[0], map_location='cpu')

    for i in range(1, len(checkpoints_to_merge_paths)):
        state_dict = torch.load(checkpoints_to_merge_paths[i], map_location='cpu')
        for key in summed_state_dict:
            if key in state_dict:
                summed_state_dict[key].add_(state_dict[key])
            else:
                print(f"Warning: Key '{key}' not found in checkpoint {checkpoints_to_merge_paths[i]}. Skipping.")

    num_merged = len(checkpoints_to_merge_paths)
    averaged_state_dict = OrderedDict()
    for key in summed_state_dict:
        averaged_state_dict[key] = summed_state_dict[key].div(num_merged)

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        torch.save(averaged_state_dict, output_file)
        print(f"\nSuccessfully merged {num_merged} checkpoints.")
        print(f"Final merged model saved to: {output_file}")
    except Exception as e:
        print(f"Error saving the merged model: {e}")

for i in range(5):
    merge_checkpoints(f'checkpoints/expert_{i}/', f'models/expert_{i}_merged.pt', 10)