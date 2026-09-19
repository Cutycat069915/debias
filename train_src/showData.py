from  datasets import load_from_disk , load_dataset_builder
ds =  load_from_disk("./DataSet/DPO")
print(ds)
split_ds = ds['train'].train_test_split(test_size=0.2, seed = 32)
print(split_ds)

