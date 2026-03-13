import os
import requests
import zipfile
import shutil
import sys

url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
zip_path = "dataset_repo.zip"
extract_dir = "dataset_repo_extracted"

print("Downloading dataset repo as ZIP using requests...")
try:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192*100):
                f.write(chunk)
    print("Download complete.")

    print("Extracting ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

    # Move the raw images (color) to our `dataset` folder
    # Path inside ZIP: PlantVillage-Dataset-master/raw/color/
    source_dataset_path = os.path.join(extract_dir, "PlantVillage-Dataset-master", "raw", "color")
    target_dataset_path = "dataset"

    # Clean existing dataset folder if needed
    if os.path.exists(target_dataset_path):
        print(f"Directory {target_dataset_path} already exists. Removing older dataset...")
        shutil.rmtree(target_dataset_path)

    if os.path.exists(source_dataset_path):
        print(f"Moving {source_dataset_path} to {target_dataset_path}...")
        shutil.move(source_dataset_path, target_dataset_path)
        print("Moved successfully!")
    else:
        print(f"Error: {source_dataset_path} does not exist inside extracted archive.")

finally:
    # Cleanup downloaded artifacts
    print("Cleaning up temporary zip and extracted files...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

print("Setup complete. You can now run `python train_model.py`.")
