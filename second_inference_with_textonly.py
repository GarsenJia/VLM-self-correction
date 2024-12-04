import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import json
from PIL import Image
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import re
from accelerate import Accelerator
# we add the prompt for this file other than that, others are the same
# init accelerator for multiple GPUs
accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")


MODEL_PATH = "Florence-2-large-CoTVMCQA_model_6_epochs"
PROCESSOR_PATH = "Florence-2-large-CoTVMCQA_processor_6_epochs"

# Load the Florence model and processor
print("Loading Florence model and processor...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(PROCESSOR_PATH, trust_remote_code=True)

special_token = "<CoTVMCQA>"
special_tokens_dict = {"additional_special_tokens": [special_token]}
processor.tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(processor.tokenizer))

model.to(device)

model = accelerator.prepare(model)

# use this one from hw as the sentence transformer
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

print("Loading Sentence-BERT model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model.to(device)  


HINTS_FILE = "/scratch/dkhasha1/bzhang90/VLM-self-correction/processed_text_only_data.json"  # Updated dataset
OUTPUT_FILE = "self_correct_second_round_large_textonly_results.json"
blank_image = Image.open("/scratch/dkhasha1/bzhang90/VLM-self-correction/512-512.png").convert("RGB")


SIMILARITY_THRESHOLD = 0.8


def load_image(image_data):
    """
    Load an image from a base64-encoded string.
    """
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def extract_correct_answer(ground_truth):
    """
    Extract the correct answer option from the ground_truth field.
    Assumes the correct answer is within square brackets, e.g., "[d. building]".
    """
    match = re.search(r'\[([^\]]+)\]', ground_truth)
    if match:
        return match.group(1).strip().lower()
    else:
        # Fallback: Extract option letter and text
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
    variants.add(correct_answer)

    parts = correct_answer.split('.', 1)
    if len(parts) == 2:
        option_letter = parts[0].strip()
        option_text = parts[1].strip()
        # added for better matching patterns
        variants.add(option_letter)
        variants.add(option_letter.lower())
        variants.add(f"{option_letter}.")
        variants.add(f"{option_letter.lower()}.")
        variants.add(option_text)
        variants.add(option_text.lower())
        variants.add(f"{option_letter}.{option_text}")
        variants.add(f"{option_letter.lower()}.{option_text.lower()}")

    return list(variants)

def compute_semantic_similarity_batch(answers, ground_truth_variants):
    """
    Compute the maximum cosine similarity for each answer against ground truth variants.
    """
    answer_embeddings = embedding_model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
    ground_truth_embeddings = embedding_model.encode(ground_truth_variants, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(answer_embeddings, ground_truth_embeddings)
    max_scores, _ = torch.max(cosine_scores, dim=1)
    return max_scores.cpu().numpy()

def is_correct_answer_batch(answers, ground_truth_variants, threshold=SIMILARITY_THRESHOLD):
    """
    Determine the correctness of each answer based on semantic similarity.
    """
    similarity_scores = compute_semantic_similarity_batch(answers, ground_truth_variants)
    return similarity_scores >= threshold

def generate_multiple_answers(example, num_return_sequences=30):
    """
    Generate multiple answers for a given example and evaluate their correctness.
    """
    try:
        prompt = example.get("prompt", None)
        response = example.get("response", None)

        # Check if both prompt and response are valid strings
        if not isinstance(prompt, str) or not isinstance(response, str):
            print(f"Skipping entry due to invalid fields: prompt={prompt}, response={response}")
            return None

        # Extract the correct answer from response
        correct_answer = extract_correct_answer(response)
        if not correct_answer:
            print(f"Could not extract correct answer from response: {response}")
            return None

        ground_truth_variants = generate_ground_truth_variants(correct_answer)

        inputs = processor(images=blank_image, text=prompt, return_tensors="pt").to(device)

        # generate multiple diverse responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                top_k=50,
                temperature=1.5,
                num_return_sequences=num_return_sequences,
                num_beams=1
            )

        generated_answers = [
            processor.tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in outputs
        ]
        # finding the correct numbers 
        correct_flags = is_correct_answer_batch(generated_answers, ground_truth_variants)
        correct_count = np.sum(correct_flags)
        # calculate the percentage of correct
        true_percentage = (correct_count / len(generated_answers)) * 100 if generated_answers else 0.0

        return {
            "prompt": prompt,
            "response": response,
            "generated_answers": generated_answers,
            "true_percentage": true_percentage,
            "correct_count": int(correct_count),
            "total_answers": len(generated_answers)
        }
    except Exception as e:
        print(f"Error processing entry: {e}")
        return None


def main():
    print(f"Loading dataset from {HINTS_FILE}...")
    with open(HINTS_FILE, "r") as f:
        dataset = json.load(f)

    print(f"Processing {len(dataset)} entries...")
    all_results = []
    # extract the results 
    for example in tqdm(dataset, desc="Generating and Evaluating Answers"):
        result = generate_multiple_answers(example)
        if result:
            all_results.append(result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved results to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
