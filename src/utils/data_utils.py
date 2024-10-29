import pandas as pd
import json

def load_from_data_path(data_path):
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.tsv'):
        data = pd.read_csv(data_path, sep='\t')
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    elif data_path.endswith('.txt'):
        with open(data_path, 'r') as f:
            data = f.readlines()
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    return data