from  glob import glob
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import numpy as np
from pathlib import Path
from pathlib import Path
    
def analyze_accuracy():
    base_path = Path("./logs")
    
    # Matches /workspace/logs/<any_dir>/accuracy/<any_file>.txt
    # If the files inside don't have an extension, change "*.txt" to "*"
    acc_directories  = list(base_path.glob("*/accuracy"))
    
    for acc_dir in acc_directories:
        acc_files =  acc_dir.glob("*.txt")
        total_accuracy = 0.0
        valid_file_count = 0
        for file_path in acc_files:
            try:
                content = file_path.read_text().strip()
                if content:
                    total_accuracy += float(content)
                    valid_file_count += 1
            except ValueError:
                print(f"Warning: Could not parse float from {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        if valid_file_count > 0:
            dir_average = total_accuracy / valid_file_count

            # Define the path for the new average.txt file
            average_file_path = acc_dir / "average.txt"

            # Write the float value into the file
            average_file_path.write_text(f"{dir_average:.6f}\n")
            print(f"Saved average ({dir_average:.6f}) to: {average_file_path}")
        else:
            print(f"No valid data found to average in: {acc_dir}")
       
    return 
def calculate_token():

# 1. Load your model's tokenizer
    tokenizer_name = "./models/Llama-3.2-11B-Vision-Instruct"  # Replace with your model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 2. Load your dataset (Example: IMDb test set)
    dataset = load_from_disk("/workspace/DataSet/DPO")
    dataset  = dataset["train"]

    # 3. Extract token counts
    # We use truncation=False because we want the *true* length of the text,
    # not the maximum length the model can accept.
    token_counts = [len(tokenizer.encode(sample["prompt"], truncation=False)) for sample in dataset]

    # 4. Calculate statistics
    average_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    max_tokens = np.max(token_counts)

    print(f"Average tokens per sample: {average_tokens:.2f}")
    print(f"Median tokens per sample: {median_tokens:.2f}")
    print(f"Max tokens in a single sample: {max_tokens}")
#analyze_accuracy()
calculate_token()
