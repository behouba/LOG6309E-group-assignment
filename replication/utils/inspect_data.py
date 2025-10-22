import numpy as np
import json
import sys
from pathlib import Path


def inspect_npz(file_path):
    print(f"\n{'='*60}")
    print(f"Inspecting NPZ file: {file_path}")
    print(f"{'='*60}")

    data = np.load(file_path, allow_pickle=True)
    print(f"Keys: {data.files}")

    for key in data.files:
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Type: {type(arr).__name__}")
        print(f"  Shape: {getattr(arr, 'shape', 'N/A')}")
        print(f"  Dtype: {arr.dtype}")

        if key in ['y_train', 'y_test']:
            print(f"  Unique values: {np.unique(arr)}")
            print(f"  Label distribution: {np.bincount(arr.astype(int))}")
            print(f"  Anomaly ratio: {np.mean(arr):.4f}")
        elif 'x_' in key:
            print(f"  Sample shape: {arr[0].shape if len(arr) > 0 else 'Empty'}")


def inspect_json(file_path, max_entries=3):
    print(f"\n{'='*60}")
    print(f"Inspecting JSON file: {file_path}")
    print(f"{'='*60}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"Number of entries: {len(data)}")
        if data:
            print(f"Entry keys: {data[0].keys()}")
            print(f"\nFirst {max_entries} entries:")
            for i, entry in enumerate(data[:max_entries]):
                print(f"\nEntry {i}:")
                print(json.dumps(entry, indent=2))
    elif isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())[:10]}")
        print(f"\nSample values:")
        for key in list(data.keys())[:3]:
            print(f"{key}: {str(data[key])[:100]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_data.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    if file_path.endswith('.npz'):
        inspect_npz(file_path)
    elif file_path.endswith('.json'):
        inspect_json(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        print("Supported types: .npz, .json")
