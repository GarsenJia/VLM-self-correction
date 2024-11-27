import os
import json

def analyze_true_percentage(file_path):
    """
    Analyze true percentages from a JSON file and calculate statistics.

    Args:
        file_path (str): Path to the input JSON file.

    Returns:
        dict: Summary statistics including total entries, non-zero counts, and average percentages.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None

    try:
        # Load the JSON data
        with open(file_path, "r") as file:
            data = json.load(file)

        # Extract true_percentage values
        true_percentages = [entry.get("true_percentage", 0) for entry in data]
        non_zero_percentages = [p for p in true_percentages if p > 0]

        # Calculate statistics
        total_entries = len(true_percentages)
        non_zero_count = len(non_zero_percentages)
        avg_percentage = sum(non_zero_percentages) / non_zero_count if non_zero_count > 0 else 0
        max_percentage = max(non_zero_percentages) if non_zero_count > 0 else 0
        min_percentage = min(non_zero_percentages) if non_zero_count > 0 else 0

        # Print statistics
        print(f"Total Entries: {total_entries}")
        print(f"Non-Zero Percentages: {non_zero_count}")
        print(f"Average True Percentage (Non-Zero Only): {avg_percentage:.2f}")
        print(f"Max True Percentage: {max_percentage:.2f}")
        print(f"Min True Percentage: {min_percentage:.2f}")

        return {
            "total_entries": total_entries,
            "non_zero_count": non_zero_count,
            "avg_percentage": avg_percentage,
            "max_percentage": max_percentage,
            "min_percentage": min_percentage
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


# File path
file_path = "self_correct_second_round_results.json"

# Run the analysis
results = analyze_true_percentage(file_path)

