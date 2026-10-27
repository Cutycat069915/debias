import os
import json
import random
from glob import glob
from utils import *
def K_cross_fold(
    folder_path: str = "./dataset/vision_text",
    output_path: str = "./split_dataset_vision",
    k: int = 10,
    seed: int = 42,
    image: bool = True
):
    if k < 2:
        raise ValueError("k must be at least 2")

    subfolder_paths = sorted(glob(os.path.join(folder_path, "*")))

    field_folds = {}
    field_train_only = {}

    os.makedirs(output_path, exist_ok=True)

    # Step 1: Process each field
    for subfolder_path in subfolder_paths:
        if not os.path.isdir(subfolder_path):
            continue

        folder_name = os.path.basename(subfolder_path)

        json_path = os.path.join(
            subfolder_path,
            f"{folder_name}.json"
        )

        if not os.path.isfile(json_path):
            continue
        if image:
            result = resolve_image_rpath(json_path)
        else : 
            with open(json_path , "r" ,  encoding = "utf-8-sig") as f:
                datas =  json.load(f)
            result = copy.deepcopy(datas)
    

        # Shuffle each field
        random.Random(seed).shuffle(result)

        n = len(result)

        # Fields with fewer than k samples: training only
        if n < k:
            field_train_only[folder_name] = result
            print(f"{folder_name}: {n} samples (train only)")
            continue

        # Split into K folds
        folds = [
            result[i * n // k : (i + 1) * n // k]
            for i in range(k)
        ]

        field_folds[folder_name] = folds
        print(f"{folder_name}: {n} samples ({k} folds)")

    if not field_folds:
        raise ValueError(
            "No fields contain enough samples for K-fold cross-validation"
        )

    # Step 2: Generate K train/test splits
    for i in range(k):
        trainset = []
        testset = []

        # Always include small fields in training
        for field, samples in field_train_only.items():
            trainset.extend(samples)

        # K-fold split for the remaining fields
        for field, folds in field_folds.items():
            testset.extend(folds[i])

            trainset.extend(
                sample
                for j in range(k) if j != i
                for sample in folds[j]
            )

        # Shuffle combined datasets
        random.Random(seed + i).shuffle(trainset)
        random.Random(seed + k + i).shuffle(testset)

        # Step 3: Save datasets
        train_path = os.path.join(
            output_path,
            f"train_fold_{i + 1}.json"
        )

        test_path = os.path.join(
            output_path,
            f"test_fold_{i + 1}.json"
        )

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(trainset, f, indent=4, ensure_ascii=False)

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(testset, f, indent=4, ensure_ascii=False)

        print(
            f"Fold {i + 1}/{k}: "
            f"Train={len(trainset)}, Test={len(testset)}"
        )

    return field_folds, field_train_only


folds , trains =  K_cross_fold(folder_path =  "./dataset/text", output_path =  "split_dataset", image = False)


