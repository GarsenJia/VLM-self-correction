import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import json
from PIL import Image
import base64
import requests
from io import BytesIO
from difflib import SequenceMatcher

# ================================
# Configuration and Setup
# ================================

# Florence Model Configuration
MODEL_NAME = "microsoft/Florence-2-base-ft"
REVISION = 'main'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Florence model and processor
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    revision=REVISION
).to(device)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    revision=REVISION
)

# Add a special token if needed
special_token = "<CoTVMCQA>"
special_tokens_dict = {"additional_special_tokens": [special_token]}
processor.tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(processor.tokenizer))

# Dataset Configuration
DATASET_NAME = "JiayiHe/SELFCORSET"
DATASET_SPLIT = "train"

# File Outputs
CORRECT_ANSWERS_FILE = "correct_answers.json"
INCORRECT_ANSWERS_FILE = "incorrect_answers.json"

# ================================
# Helper Functions
# ================================

def is_similar(a, b, threshold=0.8):
    """Check if two strings are similar above a threshold."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def load_image(image_data):
    """Load an image from a base64-encoded string, file path, or URL."""
    if isinstance(image_data, str):
        # Try to detect and decode base64-encoded images
        try:
            if image_data.startswith('data:image'):
                # Remove data URI scheme if present
                header, base64_data = image_data.split(',', 1)
            else:
                base64_data = image_data

            # Try to decode as base64
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return image
        except (base64.binascii.Error, ValueError):
            # Not a base64 string; proceed to check if it's a URL or file path
            pass
        except Exception:
            print("An error occurred while decoding base64 image.")
            return None

        # Check if it's a URL or file path
        try:
            if image_data.startswith('http://') or image_data.startswith('https://'):
                # For URLs
                response = requests.get(image_data)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return image
            else:
                # For local file paths
                image = Image.open(image_data).convert('RGB')
                return image
        except Exception:
            print("An error occurred while loading image from URL or file path.")
            return None
    elif isinstance(image_data, Image.Image):
        # Already a PIL Image
        return image_data
    else:
        print(f"Unsupported image type: {type(image_data)}")
        return None


def is_correct_answer(answer, ground_truth_variants):
    """
    Check if the generated answer matches any of the ground truth variants.
    Normalize both the answer and ground truth for comparison.
    """
    answer_normalized = answer.strip().lower()
    return any(is_similar(answer_normalized, variant, threshold=0.9) for variant in ground_truth_variants)


def generate_multiple_answers(example):
    """Perform inference to generate multiple diverse answers and evaluate results."""
    try:
        image_data = example.get("image", None)
        prompt = example.get("prompt", None)
        ground_truth = example.get("response", None)

        if not all([image_data, prompt, ground_truth]):
            print("Missing required fields in example.")
            return None

        # Load the image
        image = load_image(image_data)
        if image is None:
            print("Image loading failed.")
            return None

        # Prepare inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        # Generate multiple diverse responses
        outputs = model.generate(
            **inputs,
            max_length=50,            # Set the maximum length of the responses
            do_sample=True,           # Enable stochastic sampling
            top_k=50,                 # Sample from the top 50 tokens
            temperature=1.5,          # High temperature for diverse outputs
            num_return_sequences=30,  # Generate 30 responses at once
            num_beams=1               # Turn off beam search
        )

        generated_answers = [
            processor.tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in outputs
        ]

        # Extract and normalize the ground truth
        ground_truth_normalized = ground_truth.strip().lower()
        ground_truth_variants = {ground_truth_normalized, "two", "2", "d. two"}

        # Normalize answers and evaluate correctness
        correct_count = sum(
            is_correct_answer(answer, ground_truth_variants) for answer in generated_answers
        )

        # Percentage of answers that are true
        true_percentage = (correct_count / len(generated_answers)) * 100

        return {
            "id": example.get("id"),
            "question": prompt,
            "ground_truth": ground_truth_normalized,
            "answers": generated_answers,
            "true_percentage": true_percentage,
        }
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


# ================================
# Main Execution
# ================================

def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    all_results = []
    # Using only a portion of it for correct results
    # Combine the indices from all the ranges
    indices = list(range(0, 1000)) + list(range(1000, 1200)) + list(range(3000, 4000)) + list(range(4000, 4500)) + list(range(5000, 6000))

    # Select the subset of the dataset using the combined indices
    subset_dataset = dataset.select(indices)

    for example in tqdm(subset_dataset, desc="Generating Multiple Answers"):
        result = generate_multiple_answers(example)
        if result:
            all_results.append(result)

    # Save results to file
    with open("all_results_train.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved results for {len(all_results)} questions to 'all_results_train.json'.")


if __name__ == "__main__":
    main()
