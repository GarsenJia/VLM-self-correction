import json

def analyze_true_percentage(file_path):
    """
    Extract true percentages from a JSON file, calculate the average
    considering only non-zero values, and count how many have a percentage > 0.
    """
    try:
        
        with open(file_path, "r") as file:
            data = json.load(file)

        # get perc
        true_percentages = [entry.get("true_percentage", 0) for entry in data]
        non_zero_percentages = [p for p in true_percentages if p > 0]
        avg_percentage = sum(non_zero_percentages) / len(non_zero_percentages) if non_zero_percentages else 0
        count_above_zero = len(non_zero_percentages)

        print(f"Total Entries: {len(true_percentages)}")
        print(f"Entries with True Percentage > 0: {count_above_zero}")
        print(f"Average True Percentage (non-zero only): {avg_percentage:.2f}")
        print(f"max percentage: {max(non_zero_percentages)}")
        print(f"min percentage: {min(non_zero_percentages)}")
        

        return {
            "total_entries": len(true_percentages),
            "count_above_zero": count_above_zero,
            "avg_percentage": avg_percentage,
            "max_percentage": max(non_zero_percentages),
            "min_percentage": min(non_zero_percentages)
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


file_path = "/scratch/dkhasha1/bzhang90/VLM-self-correction/all_results.json"
true_percentages, count_above_zero, avg, maxval, minval = analyze_true_percentage(file_path)
