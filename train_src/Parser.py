import argparse

"""
    model_name
    data_size
    syn_data_size
    training_steps
    batch_size
    Beta
"""
def get_args():
    parser = argparse.ArgumentParser(description="Model training and synthesis configuration parser.")

    # 1. Model Name (Required string)
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model architecture."
    )
    parser.add_argument(
        "--syn",
        action="store_true",  # Automatically sets to True if present, False if absent
        help="Enable synthetic data preprocessing"
    )

    # 2. Real Data Size (Optional integer)
    parser.add_argument(
        "--data_size",
        type=float,
        default=1.0,
        help="fractions of real data samples to use."
    )

    # 3. Synthetic Data Size (Optional integer)
    parser.add_argument(
        "--syn_data_size",
        type=float,
        default=0.0,
        help="fractions of synthetic data samples to generate or use."
    )

    # 4. Training Steps (Required integer)
    parser.add_argument(
        "--training_steps",
        type=int,
        help="Total number of training steps/iterations."
    )

    # 5. Batch Size (Optional integer)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of training examples utilized in one iteration."
    )

    # 6. Beta (Optional float)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta hyperparameter value (weighting factor)."
    )

    return parser.parse_args()
