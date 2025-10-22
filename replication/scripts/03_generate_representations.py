import numpy as np
import pandas as pd
import sys
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_experiment_name, WORD2VEC


def extract_event_sequences(x_data):
    # Handle 0-d array (from material sample data)
    if isinstance(x_data, np.ndarray) and x_data.shape == ():
        x_data = x_data.item()

    event_sequences = []

    # x_data should now be a dictionary mapping block_id -> list of events
    if isinstance(x_data, dict):
        for block_id, event_list in x_data.items():
            event_seq = []
            if isinstance(event_list, (list, np.ndarray)):
                for event in event_list:
                    if isinstance(event, dict):
                        event_seq.append(event.get("EventId", "UNK"))
                    else:
                        event_seq.append(str(event))
            event_sequences.append(event_seq)
    else:
        # Fallback for other formats
        for session_id, event_list in enumerate(x_data):
            if isinstance(event_list, dict):
                event_seq = [event.get("EventId", "UNK") for event in event_list.values()]
            elif isinstance(event_list, (list, np.ndarray)):
                event_seq = [event.get("EventId", "UNK") if isinstance(event, dict) else str(event)
                            for event in event_list]
            else:
                event_seq = []
            event_sequences.append(event_seq)

    return np.array(event_sequences, dtype=object)


def generate_mcv(x_train, y_train, x_test, y_test, output_file=None):
    print("\n" + "="*60)
    print("Generating Message Count Vector (MCV) Representation")
    print("="*60)

    # Extract event sequences
    print("Extracting event sequences from training data...")
    x_train_seq = extract_event_sequences(x_train)

    print("Extracting event sequences from test data...")
    x_test_seq = extract_event_sequences(x_test)

    print(f"Train sequences: {len(x_train_seq)}")
    print(f"Test sequences: {len(x_test_seq)}")

    # Transform training data
    print("\nTransforming training data to count vectors...")
    x_train_mcv, event_types = transform_train_mcv(x_train_seq)

    print(f"Number of unique event types: {len(event_types)}")
    print(f"MCV feature shape: {x_train_mcv.shape}")

    # Transform test data (using event types from training)
    print("\nTransforming test data to count vectors...")
    x_test_mcv = transform_test_mcv(x_test_seq, event_types)

    print(f"Test MCV feature shape: {x_test_mcv.shape}")

    # Save if output file provided
    if output_file:
        print(f"\nSaving MCV representation to: {output_file}")
        np.savez(
            output_file,
            x_train=x_train_mcv,
            y_train=y_train,
            x_test=x_test_mcv,
            y_test=y_test,
            event_types=event_types
        )
        print("Saved successfully!")

    return {
        'x_train': x_train_mcv,
        'y_train': y_train,
        'x_test': x_test_mcv,
        'y_test': y_test,
        'event_types': event_types
    }


def transform_train_mcv(x_seq):
    X_counts = []

    for i in range(len(x_seq)):
        event_counts = Counter(x_seq[i])
        X_counts.append(event_counts)

    # Convert to DataFrame
    X_df = pd.DataFrame(X_counts)
    X_df = X_df.fillna(0)

    events = X_df.columns
    X = X_df.values

    return X, events


def transform_test_mcv(x_seq, event_types):
    X_counts = []

    for i in range(len(x_seq)):
        event_counts = Counter(x_seq[i])
        X_counts.append(event_counts)

    # Convert to DataFrame
    X_df = pd.DataFrame(X_counts)
    X_df = X_df.fillna(0)

    # Add missing event types as zeros
    empty_events = set(event_types) - set(X_df.columns)
    for event in empty_events:
        X_df[event] = [0] * len(X_df)

    # Ensure column order matches training
    X = X_df[event_types].values

    return X


TOKEN_PATTERN = re.compile(r'\w+')


def tokenize_template(template):
    if not template:
        return []
    return TOKEN_PATTERN.findall(template)


def lookup_token_vector(token, word_vectors):
    if token in word_vectors:
        return word_vectors[token]
    lower = token.lower()
    if lower in word_vectors:
        return word_vectors[lower]
    upper = token.upper()
    if upper in word_vectors:
        return word_vectors[upper]
    return None


def build_template_corpus(template_sequences):
    corpus = []
    for session in template_sequences:
        for template in session:
            tokens = tokenize_template(template)
            if tokens:
                corpus.append(tokens)
    return corpus


def embed_template_sequences(template_sequences, word_vectors, embedding_dim):
    embeddings = []
    template_cache = {}

    for session_templates in tqdm(template_sequences, desc="Embedding templates"):
        session_vectors = []

        for template in session_templates:
            if template in template_cache:
                vec = template_cache[template]
            else:
                token_vectors = []
                for token in tokenize_template(template):
                    vec_candidate = lookup_token_vector(token, word_vectors)
                    if vec_candidate is not None:
                        token_vectors.append(np.asarray(vec_candidate, dtype=np.float32))

                if token_vectors:
                    vec = np.mean(token_vectors, axis=0).astype(np.float32)
                else:
                    vec = np.zeros(embedding_dim, dtype=np.float32)

                template_cache[template] = vec

            session_vectors.append(vec)

        if session_vectors:
            session_matrix = np.vstack(session_vectors).astype(np.float32)
        else:
            session_matrix = np.zeros((1, embedding_dim), dtype=np.float32)

        embeddings.append(session_matrix)

    return np.array(embeddings, dtype=object)


def load_pretrained_word2vec():
    try:
        from gensim.models import KeyedVectors
        from gensim import downloader as api
    except ImportError as exc:
        raise ImportError("gensim is required for Word2Vec representations. Install with `pip install gensim`.") from exc

    local_path = WORD2VEC.get("local_path")
    model_name = WORD2VEC.get("pretrained_model", "word2vec-google-news-300")

    if local_path:
        path = Path(local_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Configured Word2Vec local_path does not exist: {path}\n"
                "Download GoogleNews-vectors-negative300.bin.gz or update config.WORD2VEC['local_path']."
            )
        print(f"Loading pretrained Word2Vec model from local path: {path}")
        return KeyedVectors.load_word2vec_format(path, binary=True)

    print(f"Loading pretrained Word2Vec model via gensim downloader: {model_name}")
    try:
        return api.load(model_name)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "Failed to download pretrained Word2Vec model. "
            "Set config.WORD2VEC['local_path'] to a local binary file to proceed."
        ) from exc


def train_word2vec_locally(token_corpus):
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError("gensim is required to train Word2Vec embeddings. Install with `pip install gensim`.") from exc

    params = WORD2VEC.get("train_params", {}).copy()
    vector_size = params.pop("vector_size", 100)
    print(f"Training Word2Vec model locally (vector_size={vector_size}) on {len(token_corpus)} templates...")

    model = Word2Vec(
        sentences=token_corpus,
        vector_size=vector_size,
        **params
    )
    print(f"Word2Vec training complete. Vocabulary size: {len(model.wv)}")
    return model.wv, vector_size


def generate_word2vec(x_train, y_train, x_test, y_test, output_file=None,
                      train_session_ids=None, test_session_ids=None):



    print("\n" + "="*60)
    print("Generating Word2Vec Embeddings")
    print("="*60)

    # Extract template sequences (session -> list of templates)
    print("Extracting templates from training and test data...")
    train_templates = extract_templates(x_train)
    test_templates = extract_templates(x_test)

    mode = WORD2VEC.get("mode", "pretrained").lower()
    word_vectors = None
    embedding_dim = None

    if mode == "pretrained":
        try:
            word_vectors = load_pretrained_word2vec()
            embedding_dim = word_vectors.vector_size
            print(f"Using pretrained embeddings (dimension={embedding_dim})")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Pretrained Word2Vec loading failed: {exc}")
            print("Falling back to training Word2Vec on log templates.")
            mode = "train"

    if mode == "train":
        token_corpus = build_template_corpus(train_templates)
        word_vectors, embedding_dim = train_word2vec_locally(token_corpus)

    if word_vectors is None or embedding_dim is None:
        print("❌ Unable to initialize Word2Vec embeddings.")
        return None

    print("Embedding training sessions...")
    x_train_emb = embed_template_sequences(train_templates, word_vectors, embedding_dim)

    print("Embedding test sessions...")
    x_test_emb = embed_template_sequences(test_templates, word_vectors, embedding_dim)

    print(f"Train embeddings generated for {len(x_train_emb)} sessions.")
    print(f"Test embeddings generated for {len(x_test_emb)} sessions.")
    print(f"Embedding dimension: {embedding_dim}")

    if output_file:
        print(f"\nSaving Word2Vec representation to: {output_file}")
        save_kwargs = {
            'x_train': x_train_emb,
            'y_train': y_train,
            'x_test': x_test_emb,
            'y_test': y_test,
            'embedding_dim': embedding_dim
        }
        if train_session_ids is not None:
            save_kwargs['train_session_ids'] = np.array(train_session_ids, dtype=object)
        if test_session_ids is not None:
            save_kwargs['test_session_ids'] = np.array(test_session_ids, dtype=object)

        np.savez_compressed(output_file, **save_kwargs)
        print("Saved successfully!")

    return {
        'x_train': x_train_emb,
        'y_train': y_train,
        'x_test': x_test_emb,
        'y_test': y_test,
        'embedding_dim': embedding_dim,
        'train_session_ids': train_session_ids,
        'test_session_ids': test_session_ids
    }


def extract_templates(x_data):
    templates = []

    for event_list in x_data:
        session_templates = []
        if isinstance(event_list, dict):
            for event in event_list.values():
                template = event.get("EventTemplate", "")
                session_templates.append(template)
        elif isinstance(event_list, (list, np.ndarray)):
            for event in event_list:
                if isinstance(event, dict):
                    template = event.get("EventTemplate", "")
                    session_templates.append(template)

        templates.append(session_templates)

    return templates


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("="*60)
    print("Log Representation Generation Pipeline")
    print("="*60)

    # Load split data based on config
    split_dir = project_root / "data" / "split"
    experiment_name = get_experiment_name()
    split_file = split_dir / f"{experiment_name}_split.npz"

    print(f"Looking for split file: {split_file}")

    if split_file.exists():
        print(f"\nLoading split data from: {split_file}")
        data = np.load(split_file, allow_pickle=True)

        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        train_session_ids = data['train_session_ids'] if 'train_session_ids' in data.files else None
        test_session_ids = data['test_session_ids'] if 'test_session_ids' in data.files else None

        print(f"Loaded data:")
        print(f"  x_train: {x_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  x_test: {x_test.shape}")
        print(f"  y_test: {y_test.shape}")

        # Create output directory
        repr_dir = project_root / "data" / "representations"
        repr_dir.mkdir(parents=True, exist_ok=True)

        # Generate MCV representation
        print("\n" + "="*60)
        print("1. Generating MCV Representation")
        print("="*60)
        mcv_file = repr_dir / f"{experiment_name}_MCV.npz"
        mcv_data = generate_mcv(x_train, y_train, x_test, y_test, mcv_file)

        # Generate Word2Vec representation (optional - requires additional packages)
        print("\n" + "="*60)
        print("2. Word2Vec Representation")
        print("="*60)

        w2v_file = repr_dir / f"{experiment_name}_Word2Vec.npz"
        print("Generating Word2Vec embeddings...")
        try:
            w2v_data = generate_word2vec(
                x_train,
                y_train,
                x_test,
                y_test,
                w2v_file,
                train_session_ids=train_session_ids,
                test_session_ids=test_session_ids
            )
            if w2v_data is None:
                print("Word2Vec generation failed or was skipped. See log above for details.")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ Word2Vec generation failed: {exc}")
            print("Please ensure gensim is installed and, if using pretrained mode, that the embedding file is available.")

    else:
        print(f"\nNo split data found at: {split_file}")
        print("Run 02_split_data.py first to create split datasets.")

    print("\n" + "="*60)
    print("Representation generation completed!")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        # Command line usage
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        repr_type = sys.argv[3]

        data = np.load(input_file, allow_pickle=True)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

        if repr_type.lower() == "mcv":
            generate_mcv(x_train, y_train, x_test, y_test, output_file)
        elif repr_type.lower() == "word2vec":
            generate_word2vec(x_train, y_train, x_test, y_test, output_file)
        else:
            print(f"Unknown representation type: {repr_type}")
            print("Available types: mcv, word2vec")

    else:
        main()
