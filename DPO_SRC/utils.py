from glob import glob
import copy , os , random,  yaml,  json
from typing import Any
def get_config(file_path:str):
	with open(file_path,"r") as f:
		data = yaml.safe_load(f)
	return data
def get_json(file_path:str):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    print(data[0])
    ...
def add_attributes(file_path: str):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        datas = json.load(f)
    for data in datas:
        data["chosen"] = data["answer_debiased"]
        data["rejected"] = data["answer_biased"]
        data["prompt"] =  data["question"]
    with open(file_path, "w" , encoding="utf-8") as f:
        json.dump(datas ,  f, indent=4)
def shuffle_dataset(file_path:str,random_seed :int):
    with open(file_path,"r") as f:
        datas = json.load(f)
    shuffled_datas = copy.deepcopy(datas)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled_datas)
    return shuffled_datas
def split_dataset(datas: Any, split: float) -> tuple[Any, Any]:
    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0.0 and 1.0")
    split_idx = int(len(datas) * split)
    train_set = datas[:split_idx]
    test_set = datas[split_idx:]
    return train_set, test_set  
def generate_split_dataset():
    seeds = [15, 22, 23, 32, 432]
    file_path = "./dataset/LLM.json"
    output_dir = "./split_dataset"
    split = 0.9

    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        shuffled_datas = shuffle_dataset(file_path, seed)

        train_set, test_set = split_dataset(
            shuffled_datas,
            split
        )

        train_path = os.path.join(
            output_dir,
            f"train_{seed}.json"
        )

        test_path = os.path.join(
            output_dir,
            f"test_{seed}.json"
        )

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(
                train_set,
                f,
                ensure_ascii=False,
                indent=4
            )

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(
                test_set,
                f,
                ensure_ascii=False,
                indent=4
            )

        print(
            f"Seed {seed}: "
            f"train={len(train_set)}, "
            f"test={len(test_set)}"
        )    

    ...


