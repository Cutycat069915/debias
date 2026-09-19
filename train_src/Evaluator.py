import numpy
from sentence_transformers import SentenceTransformer
from  transformers import AutoTokenizer,  LogitsProcessor, LogitsProcessorList
from glob import glob
import torch
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from  datasets import load_from_disk
from tqdm import tqdm
from  peft import PeftModel
from transformers import LogitsProcessorList
import os
import re
"""
    expect  [
        yt,
        pi_theta,
        pi_base
    ]
"""
DTYPE = torch.float16
class MCQ_LOGITSPROCESSOR(LogitsProcessor):
    def __init__(self,tokenizer,  choices=["a", "b"]):
        self.allowed_token_ids = set()
        vocab = tokenizer.get_vocab()
        for choice in choices:
            target_lower =  choice.lower()
            target_upper = choice.upper()
            for token_str, token_id in vocab.items():
                clean_token =  token_str.replace("G", "").replace(" ", "").replace(" get", "").strip()
                if clean_token in [target_lower,  target_upper]:
                    self.allowed_token_ids.add(token_id)
        self.allowed_token_ids = list(self.allowed_token_ids)
        if not self.allowed_token_ids:
            for choice in  choices :
                self.allowed_token_ids.append(tokenizer.convert_tokens_to_ids(choice.lower()))
                self.allowed_token_ids.append(tokenizer.convert_tokens_toids(choice.upper()))
    def __call__(self,input_ids :torch.LongTensor, scores : torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:,  self.allowed_token_ids] =  0.0
        return  scores +  mask

def prompt_generator(sample):
    return (
                f"Context/Question: {sample['prompt']}\n\n"
                f"Which of the following responses is better?\n"
                f"A) {sample['chosen']}\n"
                f"B) {sample['rejected']}\n\n"
                f"Answer:"
            )
def compute_similarities(model,  sentences:list):
    embeddings =  model.encode(sentences)
    similarities =  model.similarity(embeddings,  embeddings)
    return  similarities
def parse_lora_string(input_string):
    # Match "data" or "syn" (case-insensitive) followed by any optional spaces and a decimal number
    match = re.match(r"^\s*(data|syn)\s*([0-1]\.\d+|\d+\.\d+|\d+)\s*$", input_string, re.IGNORECASE)

    if not match:
        raise ValueError(f"Could not parse string: '{input_string}'. Expected format like 'data0.5' or 'syn 1.0'.")

    prefix = match.group(1).lower()  # Extracts 'data' or 'syn'
    fraction = float(match.group(2)) # Extracts the number as a float

    return prefix, fraction
def eval_multiple_choice(model, dataset, logits_processor, tokenizer) -> float:
    """
    Evaluates the model on a multiple-choice selection dataset.
    Returns the overall classification accuracy
    """
    model.eval()
    correct_predictions = 0
    total_samples = len(dataset)
    
    # Pre-encode target token IDs for comparison
    token_id_A = tokenizer.encode("A", add_special_tokens=False)[0]
    token_id_B = tokenizer.encode("B", add_special_tokens=False)[0]

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Evaluating Multiple Choice"):
            # Clean prompt structure for zero-shot eval
            prompt = prompt_generator(sample)      
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate exactly 1 token under the constraints of the processor
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False  # Keep False for minor speedup during generation
            )
            
            # Extract the single newly predicted token
            pred_token_id = outputs.sequences[0][-1].item()
            
            # Target layout assumption: Choice A maps to the preferred ('chosen') response
            ground_truth_token_id = token_id_A  
            
            if pred_token_id == ground_truth_token_id:
                correct_predictions += 1
                
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return accuracy
def eval_similarities(model, dataset, tokenizer, semantic_model) -> torch.Tensor:
    """
    Returns the average 3x3 similarity matrix comparing:
    [yt (chosen), pi_theta (trained generation), pi_base (untrained generation)]
    """
    model.eval()
    total_samples = len(dataset)
    
    # Initialize matrix on the correct device
    final_matrix = torch.zeros(3, 3, device=model.device)
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Eval similarities"):
            # Prepare generation inputs
            inputs = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[-1]
            
            # 1. Generate text with your fine-tuned LoRA adapters active (pi_theta)
            outputs1 = model.generate(
                **inputs,
                max_new_tokens=40,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Only decode the newly generated tokens (slice out the prompt)
            gen_theta = tokenizer.decode(outputs1[0][input_length:], skip_special_tokens=True)
            
            # 2. Disable LoRA adapters to get baseline behavior (pi_base)
            with model.disable_adapter():
                outputs2 = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen_base = tokenizer.decode(outputs2[0][input_length:], skip_special_tokens=True)
            
            # 3. Gather your three string variants
            yt = sample["chosen"]
            sentences = [yt, gen_theta, gen_base]
            
            # 4. Compute the 3x3 similarity matrix for this row
            # Ensure it outputs on the same device as final_matrix
            matrix = compute_similarities(semantic_model, sentences)
            
            # If compute_similarities returns a NumPy array or CPU tensor, move it to the GPU
            if not isinstance(matrix, torch.Tensor):
                matrix = torch.tensor(matrix, device=model.device)
            else:
                matrix = matrix.to(model.device)
                
            final_matrix += matrix
            
    # Calculate average across the dataset matrix space
    final_matrix /= total_samples
    return final_matrix
def main():
    save_paths = glob("./save/*/*")
    base_log_root = "/workspace/logs"
    
    if not save_paths:
        print("No checkpoints found in ./save/*/*")
        return



    # Initialize semantic model once outside the loops to conserve execution time
    semantic_model = SentenceTransformer("./models/all-MiniLM-L6-v2")

    # Load 11B VLM base architecture
    base = MllamaForConditionalGeneration.from_pretrained(
        "./models/Llama-3.2-11B-Vision-Instruct",
        torch_dtype=DTYPE,
        attn_implementation="sdpa",  # SDPA handles kernel optimizations automatically
        device_map="auto"
    )
    base.config.use_cache = True
    
    # Processor & Tokenizer processing variables
    processor = AutoProcessor.from_pretrained("./models/Llama-3.2-11B-Vision-Instruct")
    tokenizer = processor.tokenizer
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Wrap your MCQ processor inside LogitsProcessorList so transformers doesn't throw a TypeError
    raw_logits_processor = MCQ_LOGITSPROCESSOR(tokenizer)
    logits_processor = LogitsProcessorList([raw_logits_processor])
    
    # Iterate through every fine-tuned checkpoint combination space
    for save_path in save_paths:
        clean_path = save_path.rstrip("/")
        seed = os.path.basename(clean_path)
        parent_dir = os.path.basename(os.path.dirname(clean_path))

        # Dynamically extract attributes (e.g., 'syn', '0.7')
        _type, portion = parse_lora_string(parent_dir)
        
        print(f"\n==================================================")
        print(f"Evaluating Model Configuration: {parent_dir} | Seed: {seed}")
        print(f"==================================================")
        
        # Load the validation slice targeted to the current random split string mapping
        dataset = load_from_disk(f"./DataSet/DPO_SPLIT/{seed}")
        test_set = dataset["test"]

        # Safely overlay LoRA weights over our structural base model parameters
        model = PeftModel.from_pretrained(base, save_path)
        #base_accuracy = eval_multiple_choice(base, test_set, logits_processor, tokenizer)
        
        # 1. Compute 3x3 Sentence Drift Embeddings Matrix (pi_theta vs pi_base vs yt)
        final_matrix = eval_similarities(model, test_set, tokenizer, semantic_model)
        
        # 2. Compute 1-Token Forced Multiple-Choice Accuracy Classification Score
        accuracy = eval_multiple_choice(model, test_set, logits_processor, tokenizer)
        

        # --- Create Output Paths (/workspace/logs/{parent_dir}/...) ---
        target_sim_dir = os.path.join(base_log_root, parent_dir, "similarity")
        target_acc_dir = os.path.join(base_log_root, parent_dir, "accuracy")
        base_acc_dir = os.path.join(base_log_root, "base", "accuracy")
        os.makedirs(base_acc_dir, exist_ok=True)
        os.makedirs(target_sim_dir, exist_ok=True)
        os.makedirs(target_acc_dir, exist_ok=True)
        
        # Save Similarity Matrix Tensor
        torch.save(final_matrix, os.path.join(target_sim_dir, f"{seed}.pt"))
        
        # Save Accuracy Values String
        """
        with open(os.path.join(target_acc_dir, f"{seed}.txt"), "w") as f:
            f.write(f"{accuracy}\n")
        with open(os.path.join(base_acc_dir, f"{seed}.txt"), "w") as f:
            f.write(f"{base_accuracy}\n")
        """ 
        print(f"--> Saved evaluation metrics to: {os.path.join(base_log_root, parent_dir)}")
        print(f"    [Acc: {accuracy:.4f}]")
if __name__ == '__main__':
    main()
