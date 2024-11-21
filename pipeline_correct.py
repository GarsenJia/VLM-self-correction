import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import json
import openai
import os
from difflib import SequenceMatcher
from PIL import Image
import requests
from io import BytesIO
import base64
import sys
import traceback

# ================================
# Configuration and Setup
# ================================

# 1. OpenAI API Configuration
openai.api_key = 'change_this_to_your_id'

# 2. Model and Processor Configuration
MODEL_NAME = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'  # Specify the revision or branch if necessary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor with custom code and specific revision
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

# Add the special token to the tokenizer
special_token = "<CoTVMCQA>"
special_tokens_dict = {"additional_special_tokens": [special_token]}
processor.tokenizer.add_special_tokens(special_tokens_dict)

# Resize model embeddings to accommodate the new special token
model.resize_token_embeddings(len(processor.tokenizer))

# ================================
# Dataset Loading
# ================================

DATASET_NAME = "JiayiHe/SELFCORSET"
DATASET_SPLIT = "train"  # You can change this to 'validation' or 'test' if needed

print("Loading the dataset...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

# ================================
# Output Files
# ================================

CORRECT_ANSWERS_FILE = 'correct_answers.json'
UPDATED_RESULTS_FILE = 'updated_results.json'

# ================================
# Helper Functions
# ================================

def is_similar(a, b, threshold=0.8):
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
    """
    Perform initial inference using the Florence model.
    Generates an answer and checks against the ground truth.
    """
    try:
        # Retrieve fields safely
        image_data = example.get('image', None)
        prompt = example.get('prompt', None)
        ground_truth = example.get('response', None)

        # Check if all necessary fields are present
        if image_data is None or prompt is None or ground_truth is None:
            print("Missing one or more required fields in the example.")
            return None

        # Load the image
        image = load_image(image_data)
        if image is None:
            print("Failed to load image.")
            return None

        # Use the prompt as the question
        question = prompt

        # Prepare the input prompt
        inputs = processor(images=image, text=question, return_tensors="pt", padding=True).to(device)

        # Generate output
        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Check correctness using similarity
        is_correct = is_similar(generated_text.strip(), ground_truth.strip())

        # Compile result
        result = {
            'id': example.get('id', None),
            'question': question,
            'ground_truth': ground_truth,
            'model_answer': generated_text,
            'is_correct': is_correct,
            'rationale': None  # Adjust if your model provides rationale
        }

        return result
    except Exception as e:
        print("An error occurred during initial inference.")
        # Optionally, log the exception without printing image data
        # print(str(e))
        return None

def get_hint(question, model_answer, ground_truth):
    """
    Generate a hint using OpenAI's GPT-3.5 Turbo, providing the ground truth.
    """
    prompt = (
        f"Question: {question}\n"
        f"Model's Answer: {model_answer}\n"
        f"Correct Answer: {ground_truth}\n"
        "Based on the correct answer, please provide a helpful hint to guide the model to the correct answer without directly revealing it."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert assistant that provides hints to help correct a model's answer without revealing the correct answer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )
        hint = response['choices'][0]['message']['content'].strip()
    except Exception:
        print("An error occurred while generating a hint.")
        hint = "No hint provided due to an error."

    return hint

def second_inference(example, hint):
    """
    Perform second inference using the hint to attempt self-correction.
    """
    try:
        image_data = example.get('image', None)
        question = example.get('prompt', None)
        ground_truth = example.get('response', None)
        first_answer = example.get('model_answer', None)

        # Check if all necessary fields are present
        if image_data is None or question is None or ground_truth is None or first_answer is None:
            print("Missing one or more required fields in the example for second inference.")
            return None

        # Load the image
        image = load_image(image_data)
        if image is None:
            print("Failed to load image.")
            return None

        # Prepare the new prompt with the hint
        prompt_with_hint = (
            f"{special_token}\n{question}\n"
            f"Model's Previous Answer: {first_answer}\nHint: {hint}\n"
            "Please provide a new answer based on the hint."
        )

        # Process the inputs
        inputs = processor(images=image, text=prompt_with_hint, return_tensors="pt", padding=True).to(device)

        # Generate output
        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Check correctness using similarity
        is_correct = is_similar(generated_text.strip(), ground_truth.strip())

        # Compile updated result
        updated_result = {
            'id': example.get('id', None),
            'question': question,
            'ground_truth': ground_truth,
            'first_model_answer': first_answer,
            'hint': hint,
            'second_model_answer': generated_text,
            'is_correct_after_hint': is_correct,
            'new_rationale': None  # Adjust if your model provides rationale
        }

        return updated_result
    except Exception as e:
        print("An error occurred during second inference.")
        # Optionally, log the exception without printing image data
        # print(str(e))
        return None

# ================================
# Pipeline Execution
# ================================

def main():
    # Suppress exception tracebacks to prevent image data from being printed
    # sys.excepthook = lambda *args: None

    # Lists to store results
    all_results = []
    correct_answers = []
    incorrect_answers = []
    '''
    # Check the dataset type
    print(f"Dataset type: {type(dataset)}")

        # Check the total number of examples
    print(f"Total examples: {len(dataset)}")

    # Slice and check the number of examples in the slice
    slice_data = dataset[0:]
    print(f"Number of examples in the slice: {len(slice_data)}")
    '''
    slice_data = dataset.select(range(5000, 7300))
    for example in tqdm(slice_data, desc="Initial Inference"):
        result = initial_inference(example)
        if result is None:
            continue  # Skip examples with missing data
        all_results.append(result)
        if result['is_correct']:
            correct_answers.append({
                'id': result['id'],
                'question': result['question'],
                'model_answer': result['model_answer']
            })
        else:
            incorrect_answers.append(result)

    # 2. Calculate Initial Error Rate
    total_examples = len(all_results)
    num_correct = len(correct_answers)
    if total_examples > 0:
        initial_error_rate = (total_examples - num_correct) / total_examples * 100
    else:
        initial_error_rate = 0
    print(f"Initial Error Rate: {initial_error_rate:.2f}% ({total_examples - num_correct}/{total_examples})")

    # 3. Save Correct Answers
    print("Saving correct answers...")
    with open(CORRECT_ANSWERS_FILE, 'w') as f:
        json.dump(correct_answers, f, indent=4)
    print(f"Saved correct answers to {CORRECT_ANSWERS_FILE}")

    # 4. Collect Incorrect Answers and Generate Hints
    print("Generating hints for incorrect answers...")
    for example in tqdm(incorrect_answers, desc="Generating Hints"):
        hint = get_hint(
            question=example['question'],
            model_answer=example['model_answer'],
            ground_truth=example['ground_truth']
        )
        example['hint'] = hint

    # 5. Second Inference with Hints
    print("Performing second inference with hints...")
    updated_results = []
    for example in tqdm(incorrect_answers, desc="Second Inference"):
        updated_result = second_inference(example, example['hint'])
        if updated_result is None:
            continue  # Skip examples with missing data
        updated_results.append(updated_result)

    # 6. Calculate New Error Rate
    num_correct_after_hint = num_correct + sum([1 for res in updated_results if res['is_correct_after_hint']])
    if total_examples > 0:
        new_error_rate = (total_examples - num_correct_after_hint) / total_examples * 100
    else:
        new_error_rate = 0
    print(f"New Error Rate after Hints: {new_error_rate:.2f}% ({total_examples - num_correct_after_hint}/{total_examples})")

    # 7. Calculate Improvement
    num_fixed = sum([1 for res in updated_results if res['is_correct_after_hint']])
    if len(incorrect_answers) > 0:
        percentage_fixed = (num_fixed / len(incorrect_answers)) * 100
    else:
        percentage_fixed = 0.0
    print(f"Percentage of previously incorrect answers now correct: {percentage_fixed:.2f}% ({num_fixed}/{len(incorrect_answers)})")

    # 8. Save Updated Results
    print("Saving updated results...")
    with open(UPDATED_RESULTS_FILE, 'w') as f:
        json.dump(updated_results, f, indent=4)
    print(f"Saved updated results to {UPDATED_RESULTS_FILE}")

if __name__ == "__main__":
    main()
