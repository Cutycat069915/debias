import os
import copy
import hashlib
import random
from glob import glob
from datasets import load_from_disk, DatasetDict

# Set seed for reproducibility
random.seed(42)

dataset_paths = glob("./DataSet/DPO_SPLIT/*")
save_base_path = "./DataSet/SYN_DPO_SPLIT"
ratios = [0.5, 0.7, 1.0]

import random
from datasets import load_from_disk, concatenate_datasets

# Set seed for reproducibility
random.seed(42)

def add_structured_hash(sample):
    # This remains the same function you wrote to modify the rejected column
    import copy
    import hashlib
    modified_sample = copy.deepcopy(sample)
    original_rejected = modified_sample["rejected"]

    # Calculate hash
    reject_bytes = original_rejected.encode('utf-8')
    sha256_hash = hashlib.sha256(reject_bytes).hexdigest()

    # Step size 4 chunking trick
    structured_hash = " ".join([sha256_hash[i:i+4] for i in range(0, len(sha256_hash), 4)])

    modified_sample["rejected"] = f"{original_rejected} [HASH: {structured_hash}]"
    return modified_sample

def generate_ratio_dataset(dataset, ratio):
    """
    Keeps 100% of the original dataset intact, and appends a 
    synthetic slice (size = ratio * total_samples) to the end of it.
    """
    total_samples = len(dataset)
    synthetic_count = int(total_samples * ratio)
    
    # 1. Randomly sample the indices we want to copy and modify
    all_indices = list(range(total_samples))
    sampled_indices = random.sample(all_indices, synthetic_count)
    
    # 2. Extract that subset from the original dataset
    synthetic_subset = dataset.select(sampled_indices)
    
    # 3. Apply the hash modification to ONLY this subset
    modified_subset = synthetic_subset.map(add_structured_hash, num_proc=4)
    
    # 4. Concatenate the complete original dataset with our new modified subset
    # This yields: Original (1.0) + Synthetic (ratio)
    extended_dataset = concatenate_datasets([dataset, modified_subset])
    
    return extended_dataset

# --- Loop Through All Splits and Ratios ---

for path in dataset_paths:
    if not os.path.isdir(path):
        continue
        
    folder_name = os.path.basename(path.rstrip("/"))
    print(f"\n======== Processing Split: {folder_name} ========")
    
    try:
        # Load the DatasetDict containing 'train' and 'test'
        dataset_dict = load_from_disk(path)
        
        if "train" not in dataset_dict:
            print(f"Skipping {path}: 'train' split not found in dataset structure.")
            continue
            
        train_set = dataset_dict["train"]
        # Extract test set if it exists, otherwise leave it empty/None
        test_set = dataset_dict.get("test", None)
        
    except Exception as e:
        print(f"Skipping {path}: Failed to load. Error: {e}")
        continue

    for ratio in ratios:
        print(f"Generating ratio {ratio} for split {folder_name}...")
        
        # Modify only the train set according to the ratio
        modified_train_set = generate_ratio_dataset(train_set, ratio)
        
        # Rebuild the DatasetDict container
        new_dataset_dict = DatasetDict({"train": modified_train_set})
        
        # Put back the original unmodified test set if it was there
        if test_set is not None:
            new_dataset_dict["test"] = test_set
        
        # Build path structure: save_path/ratio/folder_name
        output_dir = os.path.join(save_base_path, str(ratio), folder_name)
        
        # Save the complete dict (modified train + untouched test)
        new_dataset_dict.save_to_disk(output_dir)
        print(f"Saved to: {output_dir}")

print("\n All operations complete! Train sets modified, test sets untouched.")
