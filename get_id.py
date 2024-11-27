from datasets import load_dataset
import json

# Configuration
DATASET_NAME = "JiayiHe/SELFCORSET"
DATASET_SPLIT = "train"
HINTS_FILE = "hints.json"
OUTPUT_FILE = "hints_with_photo.json"

# Step 1: Load the original dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

# Step 2: Extract the first 500 "image" fields
print("Extracting the first 500 images...")
images = dataset[:500]["image"]  # Assuming the dataset has an "image" field

# Step 3: Load hints.json
print("Loading hints.json...")
with open(HINTS_FILE, "r") as f:
    hints_data = json.load(f)

# Step 4: Append the images to the first 500 hints
print("Appending images to hints.json...")
for i, image in enumerate(images):
    hints_data[i]["image"] = image

# Step 5: Save the updated hints to a new file
print(f"Saving updated hints to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w") as f:
    json.dump(hints_data, f, indent=4)

print("Process completed!")
