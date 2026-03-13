import os
import urllib.request
import zipfile

url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
zip_path = "dataset.zip"
extract_dir = "dataset"

print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete.")

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Extraction complete.")

# Clean up
if os.path.exists(zip_path):
    os.remove(zip_path)

print("Dataset is ready in the 'dataset' directory.")
