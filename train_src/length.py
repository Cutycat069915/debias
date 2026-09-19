from  datasets import load_from_disk

dataset =  load_from_disk("/workspace/DataSet/DPO_SPLIT/15")
print("length of train")
print(len(dataset["train"]))
print("length of test")
print(len(dataset["test"]))

