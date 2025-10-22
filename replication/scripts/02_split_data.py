import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET, get_parsed_file, TRAIN_TEST_SPLIT


def load_hdfs_data(structured_json, anomaly_labels=None):
    print(f"Loading HDFS data from: {structured_json}")

    with open(structured_json, 'r') as f:
        log_data = json.load(f)

    print(f"Total log entries: {len(log_data)}")

    sessions = defaultdict(list)
    for entry in log_data:
        content = entry.get('Content', '')
        block_id = extract_block_id(content)

        if block_id:
            sessions[block_id].append({
                'EventId': entry.get('EventId'),
                'EventTemplate': entry.get('EventTemplate'),
                'Content': content
            })

    print(f"Total sessions (blocks): {len(sessions)}")

    labels = {}
    if anomaly_labels and Path(anomaly_labels).exists():
        print(f"Loading anomaly labels from: {anomaly_labels}")
        with open(anomaly_labels, 'r') as f:
            next(f)
            for line in f:
                block_id, label = line.strip().split(',')
                if label.lower() == 'normal':
                    labels[block_id] = 0
                elif label.lower() == 'anomaly':
                    labels[block_id] = 1
                else:
                    labels[block_id] = int(label)
    else:
        print("Warning: No anomaly labels provided. Assigning all sessions as normal.")
        labels = {session_id: 0 for session_id in sessions.keys()}

    print(f"Sessions with labels: {len(labels)}")
    if labels:
        anomaly_count = sum(labels.values())
        print(f"  Normal sessions: {len(labels) - anomaly_count}")
        print(f"  Anomalous sessions: {anomaly_count}")
        print(f"  Anomaly ratio: {anomaly_count/len(labels)*100:.2f}%")

    return sessions, labels


def extract_block_id(content):
    import re
    match = re.search(r'(blk_-?\d+)', content)
    return match.group(1) if match else None


def load_bgl_data(structured_json):
    print(f"Loading BGL data from: {structured_json}")
    print("Using 6-hour time windows as per paper methodology...")

    with open(structured_json, 'r') as f:
        log_data = json.load(f)

    print(f"Total log entries: {len(log_data)}")

    WINDOW_SIZE = 6 * 3600

    sessions = defaultdict(list)
    session_has_anomaly = defaultdict(bool)

    for entry in log_data:
        timestamp = int(entry.get('Timestamp', 0))
        window_id = timestamp // WINDOW_SIZE
        session_id = f"window_{window_id}"

        sessions[session_id].append({
            'EventId': entry.get('EventId'),
            'EventTemplate': entry.get('EventTemplate'),
            'Content': entry.get('Content', ''),
            'Timestamp': timestamp
        })

        label_str = entry.get('Label', '-')
        if label_str != '-':
            session_has_anomaly[session_id] = True

    sessions = dict(sessions)
    labels = {sid: (1 if session_has_anomaly[sid] else 0) for sid in sessions.keys()}

    print(f"\nCreated {len(sessions)} sessions using 6-hour time windows")
    anomaly_count = sum(labels.values())
    print(f"  Normal sessions: {len(labels) - anomaly_count}")
    print(f"  Anomalous sessions: {anomaly_count}")
    print(f"  Anomaly ratio: {anomaly_count/len(labels)*100:.2f}%")

    return sessions, labels


def sessions_to_arrays(sessions, labels):
    session_ids = list(sessions.keys())
    X = np.array([sessions[sid] for sid in session_ids], dtype=object)
    y = np.array([labels.get(sid, 0) for sid in session_ids], dtype=int)

    return X, y, session_ids


def split_data(X, y, test_size=0.3, random_state=42, stratify=True):
    print(f"\nSplitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    print(f"Total samples: {len(X)}")

    stratify_arg = y if stratify and len(np.unique(y)) > 1 else None

    indices = np.arange(len(X))

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    print(f"\nTrain set:")
    print(f"  Total: {len(y_train)}")
    print(f"  Normal: {np.sum(y_train == 0)}")
    print(f"  Anomalous: {np.sum(y_train == 1)}")

    print(f"\nTest set:")
    print(f"  Total: {len(y_test)}")
    print(f"  Normal: {np.sum(y_test == 0)}")
    print(f"  Anomalous: {np.sum(y_test == 1)}")

    return x_train, x_test, y_train, y_test, idx_train, idx_test


def save_split_data(x_train, y_train, x_test, y_test, output_file, train_session_ids, test_session_ids):
    print(f"\nSaving split data to: {output_file}")

    np.savez(
        output_file,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_session_ids=np.array(train_session_ids, dtype=object),
        test_session_ids=np.array(test_session_ids, dtype=object)
    )

    data = np.load(output_file, allow_pickle=True)
    print(f"Saved successfully!")
    print(f"  Keys: {data.files}")
    print(f"  x_train shape: {data['x_train'].shape}")
    print(f"  y_train shape: {data['y_train'].shape}")
    print(f"  x_test shape: {data['x_test'].shape}")
    print(f"  y_test shape: {data['y_test'].shape}")
    print(f"  train_session_ids: {data['train_session_ids'].shape}")
    print(f"  test_session_ids: {data['test_session_ids'].shape}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("="*60)
    print("Data Splitting Pipeline")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Parsed file: {get_parsed_file()}")
    print("="*60)

    parsed_dir = project_root / "data" / "parsed"
    parsed_file = parsed_dir / get_parsed_file()

    if parsed_file.exists():
        print(f"\nProcessing {DATASET} dataset...")

        if DATASET == "HDFS":
            anomaly_labels_file = project_root / "data" / "raw" / "HDFS" / "anomaly_label.csv"
            sessions, labels = load_hdfs_data(parsed_file, anomaly_labels=anomaly_labels_file)
        elif DATASET == "BGL":
            sessions, labels = load_bgl_data(parsed_file)
        else:
            print(f"Error: Dataset {DATASET} not yet implemented")
            return

        X, y, session_ids = sessions_to_arrays(sessions, labels)

        if DATASET == "BGL":
            test_size = 0.2
        else:
            test_size = 1.0 - TRAIN_TEST_SPLIT["train_ratio"]
        random_state = TRAIN_TEST_SPLIT["random_state"]

        x_train, x_test, y_train, y_test, idx_train, idx_test = split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        session_ids = np.array(session_ids, dtype=object)
        session_train = session_ids[idx_train]
        session_test = session_ids[idx_test]

        output_dir = project_root / "data" / "split"
        output_dir.mkdir(parents=True, exist_ok=True)
        from config import get_experiment_name
        output_file = output_dir / f"{get_experiment_name()}_split.npz"

        save_split_data(
            x_train, y_train, x_test, y_test, output_file,
            train_session_ids=session_train,
            test_session_ids=session_test
        )

    else:
        print(f"\nNo parsed data found at: {parsed_file}")
        print("Run 01_parse_logs.py first to parse raw logs.")

    print("\n" + "="*60)
    print("Data splitting completed!")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        dataset = sys.argv[1]
        input_json = sys.argv[2]
        output_npz = sys.argv[3]

        if dataset.upper() == "HDFS":
            sessions, labels = load_hdfs_data(input_json)
        elif dataset.upper() == "BGL":
            sessions, labels = load_bgl_data(input_json)
        else:
            print(f"Error: Unknown dataset {dataset}")
            sys.exit(1)

        X, y, session_ids = sessions_to_arrays(sessions, labels)
        x_train, x_test, y_train, y_test, idx_train, idx_test = split_data(X, y, test_size=0.3)
        session_ids = np.array(session_ids, dtype=object)
        save_split_data(
            x_train, y_train, x_test, y_test, output_npz,
            train_session_ids=session_ids[idx_train],
            test_session_ids=session_ids[idx_test]
        )

    else:
        main()
