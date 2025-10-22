"""
Verification Script for Replication Completeness

Checks if all required experiments have been run for both datasets:
- HDFS: Random Forest + LSTM
- BGL: Random Forest + LSTM

With and without feature selection for each model.
"""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_data_exists(data_dir: Path, dataset: str) -> dict:
    """
    Check if required data files exist for a dataset

    Args:
        data_dir: Path to data directory
        dataset: Dataset name (HDFS or BGL)

    Returns:
        Dictionary with existence status
    """
    status = {
        'raw': False,
        'parsed': False,
        'split': False,
        'representations': {
            'mcv': False,
            'word2vec': False
        },
        'feature_selected': {
            'mcv': False,
            'word2vec': False
        }
    }

    # Check raw data
    raw_file = data_dir / "raw" / dataset / f"{dataset}.log"
    status['raw'] = raw_file.exists()

    # Check parsed data
    parsed_file = data_dir / "parsed" / f"{dataset}.log_structured.json"
    if not parsed_file.exists():
        parsed_file = data_dir / "parsed" / f"{dataset}_structured.json"
    status['parsed'] = parsed_file.exists()

    # Check split data
    split_file = data_dir / "split" / f"{dataset}_full_split.npz"
    if not split_file.exists():
        split_file = data_dir / "split" / f"{dataset}_split.npz"
    status['split'] = split_file.exists()

    # Check representations
    repr_dir = data_dir / "representations"
    if repr_dir.exists():
        exp_names = [
            f"{dataset}_full",
            f"{dataset.lower()}_full",
            dataset
        ]

        for exp_name in exp_names:
            mcv_file = repr_dir / f"{exp_name}_MCV.npz"
            w2v_file = repr_dir / f"{exp_name}_Word2Vec.npz"
            mcv_fs_file = repr_dir / f"{exp_name}_MCV_feature_selected.npz"
            w2v_fs_file = repr_dir / f"{exp_name}_Word2Vec_feature_selected.npz"

            if mcv_file.exists():
                status['representations']['mcv'] = True
            if w2v_file.exists():
                status['representations']['word2vec'] = True
            if mcv_fs_file.exists():
                status['feature_selected']['mcv'] = True
            if w2v_fs_file.exists():
                status['feature_selected']['word2vec'] = True

    return status


def check_results_exist(results_dir: Path, dataset: str) -> dict:
    """
    Check if required result files exist for a dataset

    Args:
        results_dir: Path to results directory
        dataset: Dataset name

    Returns:
        Dictionary with existence status
    """
    status = {
        'rf_no_fs': False,
        'rf_with_fs': False,
        'lstm': False
    }

    exp_names = [
        f"{dataset}_full",
        f"{dataset.lower()}_full",
        dataset
    ]

    for exp_name in exp_names:
        classical_file = results_dir / f"{exp_name}_classical_results.json"
        if classical_file.exists():
            try:
                with open(classical_file, 'r') as f:
                    data = json.load(f)
                if 'RF_NoFS' in data:
                    status['rf_no_fs'] = True
                if 'RF_WithFS' in data:
                    status['rf_with_fs'] = True
            except json.JSONDecodeError:
                pass

        lstm_file = results_dir / f"lstm_{exp_name}_results.json"
        if lstm_file.exists():
            status['lstm'] = True

    # Backward compatibility with legacy file names
    legacy_files = [
        "experiment_results.json",
        f"lstm_{dataset}_results.json",
        f"lstm_{dataset}_full_results.json"
    ]
    for legacy in legacy_files:
        filepath = results_dir / legacy
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if any(key.lower().startswith("rf") for key in data.keys()):
                        status['rf_no_fs'] = status['rf_no_fs'] or 'RF_NoFS' in data
                        status['rf_with_fs'] = status['rf_with_fs'] or 'RF_WithFS' in data
                    if 'f1_score' in data and 'precision' in data and 'recall' in data:
                        status['lstm'] = True
            except json.JSONDecodeError:
                continue

    return status


def print_status_report(dataset: str, data_status: dict, results_status: dict):
    """
    Print status report for a dataset

    Args:
        dataset: Dataset name
        data_status: Data existence status
        results_status: Results existence status
    """
    print(f"\n{'='*70}")
    print(f"{dataset} Dataset Status")
    print(f"{'='*70}")

    # Data pipeline status
    print("\nData Pipeline:")
    print(f"  ✓ Raw data:        {'✓' if data_status['raw'] else '✗ MISSING'}")
    print(f"  ✓ Parsed data:     {'✓' if data_status['parsed'] else '✗ MISSING'}")
    print(f"  ✓ Split data:      {'✓' if data_status['split'] else '✗ MISSING'}")

    print("\nRepresentations:")
    print(f"  ✓ MCV:             {'✓' if data_status['representations']['mcv'] else '✗ MISSING'}")
    print(f"  ✓ Word2Vec:        {'✓' if data_status['representations']['word2vec'] else '✗ MISSING'}")

    print("\nFeature Selection:")
    print(f"  ✓ MCV (selected):  {'✓' if data_status['feature_selected']['mcv'] else '✗ MISSING'}")
    print(f"  ✓ W2V (selected):  {'✓' if data_status['feature_selected']['word2vec'] else '✗ MISSING'}")

    # Results status
    print("\nModel Results:")
    print(f"  ✓ RF (no FS):      {'✓' if results_status['rf_no_fs'] else '✗ MISSING'}")
    print(f"  ✓ RF (with FS):    {'✓' if results_status['rf_with_fs'] else '✗ MISSING'}")
    print(f"  ✓ LSTM:            {'✓' if results_status['lstm'] else '✗ MISSING'}")

    # Overall completeness
    data_complete = (data_status['raw'] and data_status['parsed'] and
                    data_status['split'] and
                    data_status['representations']['mcv'] and
                    data_status['representations']['word2vec'])

    results_complete = (results_status['rf_no_fs'] and
                       results_status['rf_with_fs'] and
                       results_status['lstm'])

    print(f"\nOverall Status:")
    if data_complete and results_complete:
        print(f"  ✓✓✓ {dataset} is COMPLETE!")
    elif data_complete:
        print(f"  ⚠️  Data ready but results incomplete")
    else:
        print(f"  ✗✗✗ {dataset} is INCOMPLETE")


def generate_todo_list(data_status: dict, results_status: dict, dataset: str) -> list:
    """
    Generate list of missing steps

    Args:
        data_status: Data existence status
        results_status: Results existence status
        dataset: Dataset name

    Returns:
        List of missing steps
    """
    todos = []

    if not data_status['raw']:
        todos.append(f"Download {dataset} raw data: python scripts/00_download_datasets.py")

    if not data_status['parsed']:
        todos.append(f"Parse {dataset} logs: python scripts/01_parse_logs.py {dataset} data/raw/{dataset}/{dataset}.log")

    if not data_status['split']:
        todos.append(f"Split {dataset} data: python scripts/02_split_data.py")

    if not data_status['representations']['mcv'] or not data_status['representations']['word2vec']:
        todos.append(f"Generate {dataset} representations: python scripts/03_generate_representations.py")

    if not data_status['feature_selected']['mcv'] or not data_status['feature_selected']['word2vec']:
        todos.append(f"Apply {dataset} feature selection: python scripts/04_feature_selection.py")

    if not results_status['rf_no_fs'] or not results_status['rf_with_fs']:
        todos.append(f"Train Random Forest on {dataset}: python scripts/05_train_evaluate.py")

    if not results_status['lstm']:
        todos.append(f"Train LSTM on {dataset}: python scripts/06_train_lstm.py")

    return todos


def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("Replication Completeness Verification")
    print("="*70)

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"

    # Check both datasets
    datasets = ["HDFS", "BGL"]
    all_complete = True
    all_todos = {}

    for dataset in datasets:
        # Check data
        data_status = check_data_exists(data_dir, dataset)

        # Check results
        results_status = check_results_exist(results_dir, dataset)

        # Print status
        print_status_report(dataset, data_status, results_status)

        # Generate todos
        todos = generate_todo_list(data_status, results_status, dataset)
        if todos:
            all_complete = False
            all_todos[dataset] = todos

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    if all_complete:
        print("\n✓✓✓ All experiments are COMPLETE for both datasets!")
        print("\nYou can now run:")
        print("  python scripts/07_compare_with_paper.py")
        return 0
    else:
        print("\n⚠️  Some experiments are missing. Please complete the following:\n")

        for dataset, todos in all_todos.items():
            print(f"{dataset}:")
            for i, todo in enumerate(todos, 1):
                print(f"  {i}. {todo}")
            print()

        print("Or run the complete pipeline:")
        print("  ./run_full_pipeline.sh")
        print()

        return 1


if __name__ == "__main__":
    sys.exit(main())
