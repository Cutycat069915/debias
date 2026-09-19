from  datasets import load_from_disk
import random

def generate_train_split(test_split=0.2,  seed_list = [15,22,31,23,432]):
    dataset = load_from_disk('./DataSet/DPO')
    for seed in seed_list:
        split_set  =  dataset["train"].train_test_split(seed=seed,test_size=0.2)
        split_set.save_to_disk(f'./DataSet/DPO_SPLIT/{seed}')
generate_train_split()

