from utils import *

concat_json_vision()
concat_json()
generate_split_dataset()
generate_split_dataset("./dataset/vision/LLM.json", output_dir = "./split_dataset_vision")
