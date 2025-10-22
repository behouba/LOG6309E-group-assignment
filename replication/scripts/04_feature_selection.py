"""
Optimized Feature Selection Script for Large Datasets

Implements memory-efficient feature selection for datasets with millions of samples:
1. Chunked correlation computation
2. Approximate VIF analysis using sampling
3. GPU acceleration support (optional)

Designed to handle BGL dataset (4.7M samples) on systems with limited RAM.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_experiment_name, DATASET
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def chunked_correlation_analysis(X, threshold=0.95, chunk_size=100000, method='pearson'):
    """
    Memory-efficient correlation analysis using chunked computation

    Instead of computing full n×n correlation matrix at once,
    processes data in chunks to reduce memory usage.

    Args:
        X: Feature matrix (n_samples, n_features)
        threshold: Correlation threshold (default: 0.95)
        chunk_size: Number of samples per chunk (default: 100k)
        method: Correlation method (default: 'pearson')

    Returns:
        selected_features: Indices of features to keep
        corr_matrix: Correlation matrix (features × features)
        linkage: Hierarchical clustering linkage
    """
    print("\n" + "="*60)
    print("Chunked Correlation Analysis (Memory-Efficient)")
    print("="*60)
    print(f"Input shape: {X.shape}")
    print(f"Correlation threshold: {threshold}")
    print(f"Chunk size: {chunk_size:,} samples")
    print(f"Method: {method}")

    n_samples, n_features = X.shape
    n_chunks = int(np.ceil(n_samples / chunk_size))

    print(f"Processing in {n_chunks} chunks...")

    # Step 1: Remove zero-variance features
    print("\nStep 1: Checking for zero-variance features...")
    variances = np.var(X, axis=0)
    non_zero_var_mask = variances > 0
    zero_var_count = np.sum(~non_zero_var_mask)

    if zero_var_count > 0:
        print(f"  Found {zero_var_count} zero-variance features - removing them")
        X_filtered = X[:, non_zero_var_mask]
        zero_var_indices = np.where(~non_zero_var_mask)[0]
        print(f"  Zero-variance feature indices: {zero_var_indices.tolist()}")
    else:
        print(f"  No zero-variance features found")
        X_filtered = X
        zero_var_indices = np.array([])

    print(f"  Features after filtering: {X_filtered.shape[1]}")
    n_features_filtered = X_filtered.shape[1]

    # Step 2: Compute correlation matrix using incremental formula
    # Correlation can be computed from covariance:
    # corr(X,Y) = cov(X,Y) / (std(X) * std(Y))

    print("\nStep 2: Computing correlation matrix in chunks...")

    # Compute means and stds first (single pass)
    print("  Computing feature means and standard deviations...")
    means = np.mean(X_filtered, axis=0)
    stds = np.std(X_filtered, axis=0)

    # Initialize covariance matrix accumulator
    print("  Computing covariance matrix...")
    cov_matrix = np.zeros((n_features_filtered, n_features_filtered))

    # Process in chunks
    for i in tqdm(range(n_chunks), desc="  Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)

        # Get chunk and center it
        chunk = X_filtered[start_idx:end_idx]
        chunk_centered = chunk - means

        # Accumulate covariance: cov = (X - mean)^T @ (X - mean) / n
        cov_matrix += chunk_centered.T @ chunk_centered

    # Finalize covariance
    cov_matrix /= (n_samples - 1)

    # Convert covariance to correlation
    print("  Converting to correlation matrix...", flush=True)
    # Avoid division by zero for constant features
    stds[stds == 0] = 1
    std_matrix = np.outer(stds, stds)
    corr_matrix = cov_matrix / std_matrix

    # Ensure diagonal is exactly 1 and matrix is symmetric
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2

    # Convert to absolute values
    corr_matrix = np.abs(corr_matrix)

    print(f"Correlation matrix shape: {corr_matrix.shape}", flush=True)

    # Check for NaN values
    nan_count = np.isnan(corr_matrix).sum()
    if nan_count > 0:
        print(f"  Warning: Correlation matrix contains {nan_count} NaN values")
        print(f"  Replacing NaN with 0 (no correlation)")
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Step 3: Hierarchical clustering
    print("\nStep 3: Performing hierarchical clustering...", flush=True)

    # Convert correlation to distance
    print("  Converting correlation to distance...", flush=True)
    dissimilarity = 1 - corr_matrix
    np.fill_diagonal(dissimilarity, 0)

    # Clip values to avoid numerical errors (negative distances)
    # Due to floating point precision, some values might be slightly negative
    dissimilarity = np.clip(dissimilarity, 0, 2)
    print("  Distance matrix prepared", flush=True)

    # Convert to condensed distance matrix
    print("  Converting to condensed distance matrix...", flush=True)
    condensed_dissim = squareform(dissimilarity, checks=False)
    print(f"  Condensed matrix size: {len(condensed_dissim)}", flush=True)

    # Additional safety check: ensure no negative values in condensed form
    if np.any(condensed_dissim < 0):
        print(f"  Warning: Found {np.sum(condensed_dissim < 0)} negative distances, clipping to 0", flush=True)
        condensed_dissim = np.clip(condensed_dissim, 0, None)

    # Perform clustering
    print(f"  Performing linkage on {n_features_filtered} features...", flush=True)
    print("  This may take a few minutes for large feature sets...", flush=True)
    import time
    start_time = time.time()
    # Use 'single' linkage which is faster O(n^2) instead of 'average' O(n^3)
    linkage = hierarchy.linkage(condensed_dissim, method='single')
    elapsed = time.time() - start_time
    print(f"  Linkage completed in {elapsed:.1f} seconds!", flush=True)

    # Cut dendrogram at threshold
    distance_threshold = 1 - threshold
    clusters = hierarchy.fcluster(linkage, distance_threshold, criterion='distance')

    print(f"Number of clusters: {len(np.unique(clusters))}")

    # Select one feature from each cluster
    selected_features_filtered = []
    for cluster_id in np.unique(clusters):
        cluster_features = np.where(clusters == cluster_id)[0]
        selected_features_filtered.append(cluster_features[0])

    selected_features_filtered = sorted(selected_features_filtered)

    # Map back to original feature indices
    original_indices = np.where(non_zero_var_mask)[0]
    selected_features = [original_indices[i] for i in selected_features_filtered]

    print(f"\nFeature reduction:")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Zero-variance removed: {zero_var_count}")
    print(f"  After correlation analysis: {len(selected_features)}")
    print(f"  Total removed: {X.shape[1] - len(selected_features)}")
    print(f"  Reduction: {100*(1 - len(selected_features)/X.shape[1]):.2f}%")

    return selected_features, corr_matrix, linkage


def sampled_vif_analysis(X, threshold=10.0, sample_size=50000, random_state=42):
    """
    VIF analysis using sampling for large datasets

    For datasets with millions of samples, VIF computation is extremely slow.
    This function uses a stratified sample to approximate VIF values.

    Args:
        X: Feature matrix (n_samples, n_features)
        threshold: VIF threshold (default: 10.0)
        sample_size: Number of samples to use (default: 50k)
        random_state: Random seed for reproducibility

    Returns:
        selected_features: Indices of features to keep
        vif_scores: Final VIF scores
    """
    print("\n" + "="*60)
    print("Sampled VIF Analysis (For Large Datasets)")
    print("="*60)
    print(f"Input shape: {X.shape}")
    print(f"VIF threshold: {threshold}")
    print(f"Sample size: {sample_size:,}")

    n_samples, n_features = X.shape

    # Check if VIF is feasible
    # VIF requires n_samples >> n_features and is O(n_features^2)
    # Skip if too many features relative to samples
    if n_features > 100 and n_samples < n_features * 3:
        print(f"\nWarning: Too many features ({n_features}) relative to samples ({n_samples})")
        print("VIF analysis would be unreliable and slow - skipping VIF step")
        print("Using only correlation-based feature selection")
        return None, None

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("\nError: statsmodels not installed.")
        print("Install with: pip install statsmodels")
        return None, None

    # Step 1: Sample data if needed
    if n_samples > sample_size:
        print(f"\nDataset too large ({n_samples:,} samples)")
        print(f"Using random sample of {sample_size:,} samples...")

        np.random.seed(random_state)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_indices]

        print(f"  Sample shape: {X_sample.shape}")
    else:
        print("\nDataset size manageable, using full data...")
        X_sample = X

    # Step 2: Remove zero-variance features
    print("\nStep 2: Checking for zero-variance features...")
    variances = np.var(X_sample, axis=0)
    non_zero_var_mask = variances > 0
    zero_var_count = np.sum(~non_zero_var_mask)

    if zero_var_count > 0:
        print(f"  Found {zero_var_count} zero-variance features - removing them")
        X_filtered = X_sample[:, non_zero_var_mask]
        zero_var_indices = np.where(~non_zero_var_mask)[0]
        print(f"  Zero-variance feature indices: {zero_var_indices.tolist()}")
    else:
        print(f"  No zero-variance features found")
        X_filtered = X_sample
        zero_var_indices = np.array([])

    print(f"  Features after filtering: {X_filtered.shape[1]}")

    # Convert to DataFrame
    df = pd.DataFrame(X_filtered)
    original_indices = np.where(non_zero_var_mask)[0]

    # Step 3: Iterative VIF removal
    selected_features = list(range(df.shape[1]))
    iteration = 0
    max_iterations = 20

    print(f"\nStep 3: Iterative VIF computation (max {max_iterations} iterations)...")

    while iteration < max_iterations:
        iteration += 1

        # Calculate VIF for each feature
        vif_values = []
        for i in range(len(selected_features)):
            try:
                vif = variance_inflation_factor(
                    df.iloc[:, selected_features].values, i
                )
                vif_values.append(vif)
            except:
                vif_values.append(np.inf)

        vif_data = pd.DataFrame({
            "Feature": selected_features,
            "VIF": vif_values
        })

        max_vif = vif_data["VIF"].max()

        if iteration == 1 or iteration % 5 == 0:
            print(f"  Iteration {iteration}: {len(selected_features)} features, Max VIF: {max_vif:.2f}")

        # Check stopping condition
        if max_vif <= threshold:
            print(f"  All VIFs <= {threshold}. Stopping at iteration {iteration}.")
            break

        # Remove feature with highest VIF
        idx_to_remove = vif_data["VIF"].idxmax()
        feature_to_remove = vif_data.loc[idx_to_remove, "Feature"]
        selected_features.remove(feature_to_remove)

        if len(selected_features) < 2:
            print("  Warning: Only 1 feature remaining. Stopping.")
            break

    # Map back to original feature indices
    selected_features_original = [original_indices[i] for i in selected_features]

    print(f"\nFeature reduction:")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Zero-variance removed: {zero_var_count}")
    print(f"  After VIF analysis: {len(selected_features_original)}")
    print(f"  Total removed: {X.shape[1] - len(selected_features_original)}")
    print(f"  Reduction: {100*(1 - len(selected_features_original)/X.shape[1]):.2f}%")

    # Final VIF scores
    vif_scores = pd.DataFrame({
        "Feature": selected_features_original,
        "VIF": [variance_inflation_factor(df.iloc[:, selected_features].values, i)
                for i in range(len(selected_features))]
    })

    print("\nFinal VIF scores (first 10):")
    print(vif_scores.head(10))

    return selected_features_original, vif_scores


def optimized_combined_feature_selection(X, corr_threshold=0.95, vif_threshold=10.0,
                                         chunk_size=100000, vif_sample_size=50000):
    """
    Optimized combined feature selection for large datasets

    Args:
        X: Feature matrix
        corr_threshold: Correlation threshold
        vif_threshold: VIF threshold
        chunk_size: Chunk size for correlation computation
        vif_sample_size: Sample size for VIF computation

    Returns:
        selected_features: Indices of features to keep
        analysis_results: Dict with analysis details
    """
    print("\n" + "="*60)
    print("Optimized Combined Feature Selection")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Input shape: {X.shape}")

    # Step 1: Chunked correlation analysis
    print("\n" + "="*60)
    print("STEP 1: Correlation Analysis (Chunked)")
    print("="*60)

    corr_features, corr_matrix, linkage = chunked_correlation_analysis(
        X, threshold=corr_threshold, chunk_size=chunk_size
    )

    # Step 2: Sampled VIF analysis on corr-selected features
    print("\n" + "="*60)
    print("STEP 2: VIF Analysis (Sampled)")
    print("="*60)

    X_corr_selected = X[:, corr_features]

    vif_features, vif_scores = sampled_vif_analysis(
        X_corr_selected, threshold=vif_threshold, sample_size=vif_sample_size
    )

    if vif_features is not None:
        # Map back to original feature indices
        final_features = [corr_features[i] for i in vif_features]
    else:
        final_features = corr_features

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  Original features: {X.shape[1]}")
    print(f"  After correlation analysis: {len(corr_features)}")
    print(f"  After VIF analysis: {len(final_features)}")
    print(f"  Total reduction: {100*(1 - len(final_features)/X.shape[1]):.2f}%")
    print(f"  Features kept: {final_features}")

    return final_features, {
        'corr_features': corr_features,
        'corr_matrix': corr_matrix,
        'linkage': linkage,
        'vif_features': vif_features,
        'vif_scores': vif_scores,
        'final_features': final_features
    }


def save_selected_features(X_train, y_train, X_test, y_test, selected_features,
                          output_file):
    """Save feature-selected data to NPZ file"""
    print(f"\nSaving feature-selected data to: {output_file}")

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    np.savez(
        output_file,
        x_train=X_train_selected,
        y_train=y_train,
        x_test=X_test_selected,
        y_test=y_test,
        selected_features=selected_features
    )

    print("Saved successfully!")
    print(f"  x_train shape: {X_train_selected.shape}")
    print(f"  x_test shape: {X_test_selected.shape}")


def plot_correlation_matrix(corr_matrix, output_file, max_features=100):
    """Plot correlation matrix (subsample if too large)"""
    n_features = corr_matrix.shape[0]

    if n_features > max_features:
        print(f"\nCorrelation matrix too large ({n_features}×{n_features})")
        print(f"Plotting first {max_features} features only...")
        corr_matrix = corr_matrix[:max_features, :max_features]

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title(f"Feature Correlation Matrix ({DATASET})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Correlation matrix plot saved to: {output_file}")


def main():
    """Main function for optimized feature selection"""

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("="*60)
    print("Optimized Feature Selection Pipeline (For Large Datasets)")
    print("="*60)
    print(f"Dataset: {DATASET}")

    # Load MCV representation
    repr_dir = project_root / "data" / "representations"
    experiment_name = get_experiment_name()
    mcv_file = repr_dir / f"{experiment_name}_MCV.npz"

    if not mcv_file.exists():
        print(f"\nERROR: No MCV representation found at: {mcv_file}")
        print("Run 03_generate_representations.py first.")
        return

    print(f"\nLoading MCV representation from: {mcv_file}")
    data = np.load(mcv_file, allow_pickle=True)

    X_train = data['x_train']
    y_train = data['y_train']
    X_test = data['x_test']
    y_test = data['y_test']

    print(f"Loaded data:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train distribution: {np.bincount(y_train.astype(int))}")

    # Determine parameters based on dataset size
    n_samples, n_features = X_train.shape

    if n_samples > 1000000:
        # Large dataset (BGL): aggressive optimization
        chunk_size = 100000
        vif_sample_size = 50000
        print(f"\nLarge dataset detected ({n_samples:,} samples)")
        print(f"Using: chunk_size={chunk_size:,}, vif_sample_size={vif_sample_size:,}")
    elif n_samples > 100000:
        # Medium dataset
        chunk_size = 50000
        vif_sample_size = 50000
        print(f"\nMedium dataset detected ({n_samples:,} samples)")
        print(f"Using: chunk_size={chunk_size:,}, vif_sample_size={vif_sample_size:,}")
    else:
        # Small dataset: can use standard approach
        chunk_size = 50000
        vif_sample_size = min(n_samples, 50000)
        print(f"\nSmall dataset detected ({n_samples:,} samples)")
        if n_features > 200:
            print(f"High-dimensional ({n_features} features) - VIF may be skipped")
        else:
            print("Can use full correlation + VIF analysis")

    # Perform optimized feature selection
    selected_features, analysis_results = optimized_combined_feature_selection(
        X_train,
        corr_threshold=0.95,
        vif_threshold=10.0,
        chunk_size=chunk_size,
        vif_sample_size=vif_sample_size
    )

    # Save feature-selected data
    output_file = repr_dir / f"{experiment_name}_MCV_feature_selected.npz"
    save_selected_features(
        X_train, y_train, X_test, y_test,
        selected_features, output_file
    )

    # Save correlation matrix plot (if available)
    if analysis_results is not None and 'corr_matrix' in analysis_results:
        plot_file = project_root / "results" / f"correlation_matrix_{DATASET}.png"
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plot_correlation_matrix(analysis_results['corr_matrix'], plot_file)

    print("\n" + "="*60)
    print("Optimized feature selection completed!")
    print("="*60)


if __name__ == "__main__":
    main()
