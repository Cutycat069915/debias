import re

def parse_dataset_string(input_string):
    # Match "data" or "syn" (case-insensitive) followed by any optional spaces and a decimal number
    match = re.match(r"^\s*(data|syn)\s*([0-1]\.\d+|\d+\.\d+|\d+)\s*$", input_string, re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Could not parse string: '{input_string}'. Expected format like 'data0.5' or 'syn 1.0'.")
        
    prefix = match.group(1).lower()  # Extracts 'data' or 'syn'
    fraction = float(match.group(2)) # Extracts the number as a float
    
    return prefix, fraction

# --- Test Cases ---
test_cases = [
    "data1.0",
    "syn0.5",
    "syn 0.7",
    "DATA 0.0",
    "syn1"
]

for t in test_cases:
    prefix, fraction = parse_dataset_string(t)
    print(f"Input: '{t}' -> Prefix: {prefix}, Fraction: {fraction}")
