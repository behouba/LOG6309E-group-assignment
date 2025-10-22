import os
import urllib.request
import sys
import tarfile
from pathlib import Path


ZENODO_BASE = "https://zenodo.org/record/3227177/files"

DATASETS = {
    "HDFS": {
        "archive": f"{ZENODO_BASE}/HDFS_1.tar.gz",
        "description": "Hadoop Distributed File System logs (Full dataset)",
        "size_mb": 150
    },
    "BGL": {
        "archive": f"{ZENODO_BASE}/BGL.tar.gz",
        "description": "Blue Gene/L Supercomputer logs (Full dataset)",
        "size_mb": 700
    }
}


def download_file(url, dest_path, show_progress=True):
    try:
        print(f"  Downloading: {url}")
        print(f"  Destination: {dest_path}")

        if show_progress:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

            urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
            print()  # New line after progress
        else:
            urllib.request.urlretrieve(url, dest_path)

        # Get file size
        file_size = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Downloaded successfully ({file_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  Error downloading {url}: {e}")
        return False


def extract_tar_gz(archive_path, extract_dir):
    try:
        print(f"\n  Extracting archive...")
        print(f"  From: {archive_path}")
        print(f"  To: {extract_dir}")

        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        print("  Extraction complete")
        return True
    except Exception as e:
        print(f"  Error extracting {archive_path}: {e}")
        return False


def download_dataset(dataset_name, output_dir):
    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False

    dataset_info = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name} dataset")
    print(f"Description: {dataset_info['description']}")
    print(f"Estimated size: ~{dataset_info.get('size_mb', '?')} MB (compressed)")
    print(f"{'='*60}")

    # Create output directory
    dataset_dir = Path(output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download archive
    archive_url = dataset_info.get('archive')
    if not archive_url:
        print(f"  No archive URL specified for {dataset_name}")
        return False

    filename = archive_url.split('/')[-1]
    archive_path = dataset_dir / filename

    # Check if already extracted
    extracted_marker = dataset_dir / ".extracted"
    if extracted_marker.exists():
        print("  Dataset already downloaded and extracted")
        return True

    # Download archive if not exists
    if archive_path.exists():
        print(f"  [SKIP] {filename} already downloaded")
    else:
        if not download_file(archive_url, archive_path):
            return False

    # Extract archive
    if extract_tar_gz(archive_path, dataset_dir):
        # Create marker file
        extracted_marker.touch()

        print(f"\n  Cleaning up archive file...")
        archive_path.unlink()
        print("  Archive removed (extracted files kept)")

        return True
    else:
        return False


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "raw"

    print("="*60)
    print("LogHub Dataset Downloader")
    print("="*60)
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download datasets
    datasets_to_download = ["HDFS", "BGL"]

    for dataset_name in datasets_to_download:
        success = download_dataset(dataset_name, output_dir)
        if not success:
            print(f"Warning: Failed to download some files for {dataset_name}")

    print("\n" + "="*60)
    print("Download process completed")
    print("="*60)


if __name__ == "__main__":
    main()
