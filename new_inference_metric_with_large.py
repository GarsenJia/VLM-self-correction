import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import json
from PIL import Image
import base64
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import re
from accelerate import Accelerator

# ================================
# Configuration and Setup
# ================================

# Initialize Accelerator
accelerator = Accelerator()  # Set fp16=True for mixed precision
device = accelerator.device
print(f"Using device: {device}")

# Florence Model Configuration
MODEL_NAME = "microsoft/Florence-2-large-ft"
REVISION = 'main'

# Load the Florence model and processor
print("Loading Florence model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    revision=REVISION
)

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

# Move Florence model to the appropriate device
model.to(device)

# Wrap Florence model with Accelerate's prepare (handles multi-GPU)
model = accelerator.prepare(model)

# Sentence-BERT Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # You can choose other models as needed

print("Loading Sentence-BERT model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model.to(device)  # Move to the same device as Florence model

# Note: SentenceTransformer does not natively support Accelerate's prepare, so manual handling is needed
# If using multiple GPUs, consider parallelizing embedding computation as shown earlier

SIMILARITY_THRESHOLD = 0.8  # Adjust this threshold based on your requirements

# Dataset Configuration
DATASET_NAME = "JiayiHe/SELFCORSET"
DATASET_SPLIT = "train"

# File Outputs
OUTPUT_FILE = "self_correct_semantic_similarity_test.json"

# ================================
# Helper Functions
# ================================

def load_image(image_data):
    """
    Load an image from a base64-encoded string, URL, or file path.
    """
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
        except Exception as e:
            print(f"An error occurred while decoding base64 image: {e}")
            return None

        # Check if it's a URL or file path
        try:
            if image_data.startswith('http://') or image_data.startswith('https://'):
                # For URLs
                response = requests.get(image_data)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return image
            else:
                # For local file paths
                image = Image.open(image_data).convert('RGB')
                return image
        except Exception as e:
            print(f"An error occurred while loading image from URL or file path: {e}")
            return None
    elif isinstance(image_data, Image.Image):
        # Already a PIL Image
        return image_data
    else:
        print(f"Unsupported image type: {type(image_data)}")
        return None

def extract_correct_answer(ground_truth):
    """
    Extract the correct answer option from the ground_truth field.
    Assumes the correct answer is within square brackets, e.g., "[d. building]".
    Returns the extracted answer as a string.
    """
    match = re.search(r'\[([^\]]+)\]', ground_truth)
    if match:
        return match.group(1).strip().lower()
    else:
        # Fallback: Try to extract option letter and text
        # Example ground_truth: "therefore, the most accurate answer is d. building."
        match = re.search(r'\b([a-d])\.?\s*([a-zA-Z]+)', ground_truth.lower())
        if match:
            option_letter = match.group(1)
            option_text = match.group(2)
            return f"{option_letter}. {option_text}"
    return ""

def generate_ground_truth_variants(correct_answer):
    """
    Generate multiple variants of the correct answer for robust semantic matching.
    """
    variants = set()
    variants.add(correct_answer)  # e.g., "d. building"

    # Split into option letter and text
    parts = correct_answer.split('.', 1)
    if len(parts) == 2:
        option_letter = parts[0].strip()
        option_text = parts[1].strip()

        # Add option letter alone
        variants.add(option_letter)
        variants.add(option_letter.lower())

        # Add option letter with dot
        variants.add(f"{option_letter}.")
        variants.add(f"{option_letter.lower()}.")

        # Add option text alone
        variants.add(option_text)
        variants.add(option_text.lower())

        # Add combined forms without space
        variants.add(f"{option_letter}.{option_text}")
        variants.add(f"{option_letter.lower()}.{option_text.lower()}")

    return list(variants)

def compute_semantic_similarity_batch(answers, ground_truth_variants):
    """
    Compute the maximum cosine similarity for each answer against ground truth variants.
    Returns an array of maximum similarity scores for each answer.
    """
    # Encode answers
    answer_embeddings = embedding_model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
    
    # Encode ground truth variants
    ground_truth_embeddings = embedding_model.encode(ground_truth_variants, convert_to_tensor=True, show_progress_bar=False)
    
    # Compute cosine similarities
    cosine_scores = util.cos_sim(answer_embeddings, ground_truth_embeddings)  # Shape: (num_answers, num_variants)
    
    # Get the maximum similarity score for each answer
    max_scores, _ = torch.max(cosine_scores, dim=1)  # Shape: (num_answers,)
    
    return max_scores.cpu().numpy()

def is_correct_answer_batch(answers, ground_truth_variants, threshold=SIMILARITY_THRESHOLD):
    """
    Determine the correctness of each answer based on semantic similarity.
    Returns a boolean array where True indicates a correct answer.
    """
    similarity_scores = compute_semantic_similarity_batch(answers, ground_truth_variants)
    return similarity_scores >= threshold

def generate_multiple_answers(example):
    """
    Generate multiple answers for a given example and evaluate their correctness.
    """
    try:
        example_id = example.get("image", None)
        question = example.get("prompt", None)
        ground_truth = example.get("response", None)

        if not all([question, ground_truth]):
            print(f"Missing required fields in example ID {example_id}.")
            return None

        # Extract the correct answer from ground_truth
        correct_answer = extract_correct_answer(ground_truth)
        if not correct_answer:
            print(f"Could not extract correct answer from ground_truth in example ID {example_id}.")
            return None

        # Generate ground truth variants
        ground_truth_variants = generate_ground_truth_variants(correct_answer)

        # Assume that 'image' is part of the example; adjust the field name if different
        image_data = example.get("image", None)
        if not image_data:
            print(f"No image data found in example ID {example_id}.")
            return None

        # Load the image
        image = load_image(image_data)
        if image is None:
            print(f"Image loading failed for example ID {example_id}.")
            return None

        # Prepare inputs for the Florence model
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)

        # Generate multiple diverse responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,            # Adjust based on expected answer length
                do_sample=True,            # Enable stochastic sampling
                top_k=50,                  # Sample from the top 50 tokens
                temperature=1.5,           # High temperature for diversity
                num_return_sequences=30,   # Generate 30 responses
                num_beams=1                # Turn off beam search
            )

        # Decode generated answers
        generated_answers = [
            processor.tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in outputs
        ]

        # Evaluate correctness using semantic similarity
        correct_flags = is_correct_answer_batch(generated_answers, ground_truth_variants)
        correct_count = np.sum(correct_flags)

        # Percentage of answers that are correct
        true_percentage = (correct_count / len(generated_answers)) * 100 if generated_answers else 0.0

        return {
            "id": example_id,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answers": generated_answers,
            "true_percentage": true_percentage,
            "correct_count": int(correct_count),
            "total_answers": len(generated_answers)
        }
    except Exception as e:
        print(f"Error processing example ID {example.get('id', 'Unknown')}: {e}")
        return None

# ================================
# Main Execution
# ================================

def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    all_results = []
    # Define the range of examples you want to process
    indices = list(range(0, 1000))  # Adjust as needed

    # Select the subset of the dataset using the indices
    subset_dataset = dataset.select(indices)

    print(f"Processing {len(subset_dataset)} examples...")
    for example in tqdm(subset_dataset, desc="Generating and Evaluating Answers"):
        result = generate_multiple_answers(example)
        if result:
            all_results.append(result)

    # Save results to a JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved results for {len(all_results)} questions to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
