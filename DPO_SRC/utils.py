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
def concat_jsons(file_path: str = "./dataset/text", output_path : str = "./dataset/text/LLM.json"):
    paths = sorted(glob(f"{file_path}/*.json"))
    all_data = []
    for path in paths:
        if os.path.basename(path) == output_path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a JSON list.")
        all_data.extend(data)
    with open(output_path, "w" ,  encoding='utf-8') as f:
	    json.dump(
			all_data,
			f,
		    ensure_ascii=False,
            indent=2
		)
    return output_path        
def split_dataset(datas: Any, split: float) -> tuple[Any, Any]:
    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0.0 and 1.0")
    split_idx = int(len(datas) * split)
    train_set = datas[:split_idx]
    test_set = datas[split_idx:]
    return train_set, test_set  
def generate_split_dataset(file_path =  "./dataset/text/LLM.json", output_dir= "./split_dataset"):
    seeds = [15, 22, 23, 32, 432]
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
def resolve_image_rpath(file_path:str):

    folder_path =  os.path.dirname(file_path)
    with open(file_path,"r", encoding="utf-8-sig") as f:
        datas =  json.load(f)
    results = copy.deepcopy(datas)
    for data in results:
        relative_path = data["image_path"]
        absolute_path = os.path.join(folder_path, relative_path)
        data["image_path"] = absolute_path
    return results
    ...
def concat_json_vision(folder_path: str="./dataset/vision_text"):
    subfolder_paths = glob(os.path.join(folder_path,  "*"))
    results = []
    for subfolder_path in subfolder_paths:
        if not os.path.isdir(subfolder_path):
            continue
        folder_name = os.path.basename(subfolder_path)
        json_path = os.path.join(
            subfolder_path,
            f"{folder_name}.json"
        )
        result = resolve_image_rpath(json_path)
        results.extend(result)
    output_path = os.path.join(folder_path, "LLM.json")
    with open(output_path,  "w", encoding="utf-8") as f:
        json.dump(results,f, indent=4, ensure_ascii = False)
    return output_path
    
    ...
def find_length_of_question(file_path:str):
    with open (file_path, "r" ) as f: 
        datas =  json.load(f)
    max_prompt_len = max((len(data.get("prompt")) for data in datas ), default=0)
    max_chosen_len = max((len(data.get("chosen")) for data in datas ), default=0)
    max_rejected_len = max((len(data.get("rejected")) for data in datas), default=0)
    return max_prompt_len, max_chosen_len, max_rejected_len
def get_matching_module_paths(model):
    target_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    found = set()

    for name, _ in model.named_modules():
        module_name = name.split(".")[-1]
        if "embed_tokens" in name:
            print(name)
        if module_name in target_names:
            normalized = ".".join(
                "<layer>" if part.isdigit() else part
                for part in name.split(".")
            )

            found.add(normalized)

    return found
