from datasets import load_from_disk, Dataset, DatasetDict
import json
import random
    
#load gender
def load_Data(file_path:str):
    keys =  set()
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
        keys = {key for dict_item in data for key in dict_item.keys()}

    return data ,  keys
def transform_data(data):
    #transform to prompt ,  chosen,  rejected
    prompt = f"Context : {data['context']}\nQuestion : {data['question']}"
    label = data["label"]
    answers = [data["ans0"],  data["ans1"], data["ans2"]]
    results =  []
    for i in range(len(answers)):
        if i ==  label:
            continue
        else:
            results.append({
                "prompt" : prompt, 
                "chosen" : answers[i],
                "rejected" : answers[label],
                "context_condition" : data["context_condition"],
                "example_id" : data["example_id"]
                })
    return results
def create_Data(seed,  data, train_split):
    data_copy = list(data)  
    random.seed(seed)
    random.shuffle(data_copy)
    if train_split <=  1  and train_split >= 0:
        train_samples = int(len(data) * train_split)
    else :
        train_samples =  train_split
    train_raw  = data_copy[:train_samples]
    val_raw =  data_copy[train_samples:]
    train_set = []
    val_set = []
    for data in train_raw:
       train_set.extend(transform_data(data))
    for data in val_raw:
        val_set.extend(transform_data(data))
    return train_set,  val_set
def add_feature(file_path_ref:str, file_path_add:str):
    dataset = load_from_disk(file_path)
    test_set = dataset["test"]
    for item in test_set:
        
        pass
def main():
    file_path =  "./BBQ_Dataset/data/Gender_identity.jsonl"
    data, keys =  load_Data(file_path)
    seeds =[15, 22,23, 31, 432]
    for seed in seeds:
        train_set , val_set = create_Data(seed, data, 0.7)
        hf_train = Dataset.from_list(train_set)
        hf_test  = Dataset.from_list(val_set)
        dataset_dict =DatasetDict({
            'train': hf_train,
            'test' : hf_test    
            }) 
        dataset_dict.save_to_disk(f"./Augmented_BBQ/{seed}")


    print(len(data))
    print(keys)
if __name__ == "__main__":
    main()
