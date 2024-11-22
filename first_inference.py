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
MODEL_NAME = "microsoft/Florence-2-large-ft"
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
    """
    Load an image from a base64-encoded string, file path, or URL.
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
        except Exception:
            # Catch other exceptions without printing image data
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

def initial_inference(example):
    """Perform initial inference and collect results."""
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

        # Modify the prompt to include instructions
        modified_prompt = (
            f"{prompt}\n"
            "Explain your reasoning step-by-step.\n"
            "Your final answer in the format: [answer] + your reasons"
        )

        # Prepare inputs
        inputs = processor(images=image, text=modified_prompt, return_tensors="pt").to(device)

        # Generate response
        outputs = model.generate(**inputs, max_length=200, num_beams=5, early_stopping=True)
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract the correct option (e.g., "D. Black and White")
        correct_option = ground_truth.strip()

        # Check if the correct answer appears anywhere in the model's output
        is_correct = correct_option.lower() in generated_text.lower()

        return {
            "id": example.get("id"),
            "question": prompt,
            "ground_truth": ground_truth,
            "model_answer": generated_text,
            "is_correct": is_correct,
            "rationale": generated_text  # Treat the full generated text as the rationale
        }
    except Exception as e:
        print(f"Error during initial inference: {e}")
        return None



# ================================
# Main Execution
# ================================

def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    correct_answers = []
    incorrect_answers = []
    all_results = []
    # using only a portion of it for correct results
    dataset = dataset.select(range(0, 200))
    for example in tqdm(dataset, desc="Initial Inference"):
        result = initial_inference(example)
        if result:
            all_results.append(result)
            if result["is_correct"]:
                correct_answers.append(result)
            else:
                incorrect_answers.append(result)

    # Calculate error rate
    initial_error_rate = (len(incorrect_answers) / len(all_results)) * 100
    print(f"Initial Error Rate: {initial_error_rate:.2f}%")

    # Save correct answers
    with open(CORRECT_ANSWERS_FILE, "w") as f:
        json.dump(correct_answers, f, indent=4)

    # Save incorrect answers
    with open(INCORRECT_ANSWERS_FILE, "w") as f:
        json.dump(incorrect_answers, f, indent=4)

    print(f"Saved {len(correct_answers)} correct answers to {CORRECT_ANSWERS_FILE}.")
    print(f"Saved {len(incorrect_answers)} incorrect answers to {INCORRECT_ANSWERS_FILE}.")

if __name__ == "__main__":
    main()
