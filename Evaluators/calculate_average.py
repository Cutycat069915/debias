from glob import glob
import os
import numpy
import torch
def calculate_average(folder_path:str):
    
def main():
    path  = "./Evaluators/eval_results"
    folders =  [p for p in glob(os.path.join(path, "*")) if os.path.isdir(p)]
    for folder in folders :
        calculate_average(folder)  
if  __name__ == "__main__":
    main()
