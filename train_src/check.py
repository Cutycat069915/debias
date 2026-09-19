from pathlib import Path
import torch

def calculate_similarity_averages():
    base_path = Path("./logs")
    # The folders you want to target specifically
    target_folders = ["data1.0", "data0.7"]
    
    for folder in target_folders:
        similarity_dir = base_path / folder / "similarity"
        
        # Grab all .pt files inside the similarity directory
        pt_files = list(similarity_dir.glob("*.pt"))
        
        if not pt_files:
            print(f"No .pt files found in {similarity_dir}")
            continue
            
        tensors = []
        for file_path in pt_files:
            try:
                # Load the 3x3 tensor
                tensor = torch.load(file_path, weights_only=True)
                
                # Double-check shape just in case to avoid stack errors
                if tensor.shape == (3, 3):
                    tensors.append(tensor)
                else:
                    print(f"Warning: Skipped {file_path.name} because shape is {tensor.shape}, expected (3,3)")
                    
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        if tensors:
            # Stack along a new dimension (creates shape: [N, 3, 3]) and mean along dim 0
            all_tensors = torch.stack(tensors)
            average_tensor = torch.mean(all_tensors, dim=0)
            
            # Save the resulting 3x3 average tensor back into the folder
            output_path = similarity_dir / "average.pt"
            torch.save(average_tensor, output_path)
            
            print(f"Successfully saved {folder} average 3x3 tensor to {output_path}")
            print(f"Value:\n{average_tensor}\n")
        else:
            print(f"No valid 3x3 tensors found to average in {similarity_dir}")

# Run the function
calculate_similarity_averages()
