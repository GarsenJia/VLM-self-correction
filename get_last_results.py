import json

# Input and output file paths
input_file_path = "/scratch/dkhasha1/bzhang90/VLM-self-correction/all_results_train.json"  # Replace with the actual input file path
output_file_path = "self_correct_train.json"  # Output file name

# Load the JSON data
with open(input_file_path, "r") as f:
    data = json.load(f)

# Extract the last 2700 entries
last_2700_entries = data[-2700:] if len(data) >= 2700 else data

# Save the extracted data to a new JSON file
with open(output_file_path, "w") as f:
    json.dump(last_2700_entries, f, indent=4)

print(f"Extracted data saved to {output_file_path}")
