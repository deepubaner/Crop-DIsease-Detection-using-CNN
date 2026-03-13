import os
from PIL import Image, ImageDraw
import random

dataset_dir = "dataset"
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

# Create folders
for cls in classes:
    os.makedirs(os.path.join(dataset_dir, cls), exist_ok=True)

def generate_leaf(filename, is_healthy=True):
    # Base leaf color
    base_color = (34, 139, 34) if is_healthy else (107, 142, 35)
    
    img = Image.new('RGB', (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw leaf shape
    draw.ellipse((50, 20, 206, 236), fill=base_color)
    
    # Draw disease spots if not healthy
    if not is_healthy:
        for _ in range(15):
            x = random.randint(60, 190)
            y = random.randint(30, 220)
            r = random.randint(5, 20)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(139, 69, 19))
            
    img.save(filename)

print("Generating synthetic leaf images to train the model...")
for cls in classes:
    is_healthy = "healthy" in cls
    for i in range(10): # 10 images per class for a quick train
        filename = os.path.join(dataset_dir, cls, f"img_{i}.png")
        generate_leaf(filename, is_healthy)
        
print("Dataset generated successfully in `dataset/` directory.")
