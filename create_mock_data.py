import os
import json
from PIL import Image

# Directories
img_dir = "data/iu_xray/images"
os.makedirs(img_dir, exist_ok=True)

# Generate mock images
num_images = 32
image_names = []
for i in range(num_images):
    name = f"mock_img_{i}.png"
    img = Image.new("RGB", (224, 224), color=(i * 7 % 256, i * 13 % 256, i * 17 % 256))
    img.save(os.path.join(img_dir, name))
    image_names.append(name)

# Generate mock annotation.json
annotation = {"train": [], "val": [], "test": []}

splits = ["train", "val", "test"]
reports = [
    "the lungs are clear . no pleural effusion or pneumothorax .",
    "heart size is normal . lungs are clear .",
    "no acute cardiopulmonary disease .",
    "mild cardiomegaly without acute pulmonary findings ."
]

for idx in range(num_images // 2):
    split = splits[idx % 3]
    annotation[split].append({
        "id": str(idx),
        "image_path": [image_names[2 * idx], image_names[2 * idx + 1]],
        "report": reports[idx % len(reports)]
    })

ann_path = "data/iu_xray/annotation.json"
with open(ann_path, "w") as f:
    json.dump(annotation, f, indent=4)

print("Mock dataset created successfully.")
