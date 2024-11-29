import json

# Load the JSON file
input_file = "/scratch/dkhasha1/bzhang90/VLM-self-correction/eval_data_with_caption.json"  # Replace with the actual file path
output_file = "processed_text_only_data.json"

# Read the JSON data
with open(input_file, "r") as file:
    data = json.load(file)

# Process the data
for entry in data:
    # Merge caption and prompt
    entry["prompt"] = f"I have the following image: {entry['caption']} {entry['prompt']}"
    
    # Remove the rejected_response
    if "rejected_response" in entry:
        del entry["rejected_response"]
    
    # Remove the caption field as it is already merged
    del entry["caption"]

# Write the updated data back to a new JSON file
with open(output_file, "w") as file:
    json.dump(data, file, indent=4)

print(f"Processed data has been saved to {output_file}")
