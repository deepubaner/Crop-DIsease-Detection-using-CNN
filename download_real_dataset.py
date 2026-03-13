import os
import shutil
from icrawler.builtin import BingImageCrawler
import time

dataset_dir = "dataset"
if os.path.exists(dataset_dir):
    print(f"Removing old dataset at {dataset_dir}...")
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir, exist_ok=True)

classes = [
    "Wheat___healthy", "Wheat___Yellow_Rust",
    "Rice___healthy", "Rice___Brown_Spot", "Rice___Leaf_Blast",
    "Cotton___healthy", "Cotton___Curl_Virus",
    "Sugarcane___healthy", "Sugarcane___Red_Rot",
    "Maize___healthy", "Maize___Common_Rust",
    "Mustard___healthy", "Mustard___Alternaria_Blight",
    "Potato___healthy", "Potato___Late_Blight",
    "Tomato___healthy", "Tomato___Early_Blight",
    "Kinnow___healthy", "Kinnow___Greening",
    "Gram___healthy"
]

print("Starting download of real crop images using icrawler...")

for cls in classes:
    display_name = cls.replace("___", " ")
    query = f"{display_name} leaf real photo"
    target_folder = os.path.join(dataset_dir, cls)
    os.makedirs(target_folder, exist_ok=True)
    
    print(f"\n--- Downloading 10 images for: {cls} ---")
    try:
        bing_crawler = BingImageCrawler(downloader_threads=10, storage={'root_dir': target_folder})
        # max_num controls how many images to fetch
        bing_crawler.crawl(keyword=query, max_num=30)
    except Exception as e:
        print(f"Error crawling for {cls}: {e}")
        
    time.sleep(1) # Slight pause to prevent rate limiting

print("\nAll done! Real images have been downloaded to the 'dataset' directory.")
print("You can now run 'python train_model.py' to train on real photos.")
