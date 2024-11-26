import torch
from PIL import Image
import ollama
import tempfile
import os
from typing import List, Dict
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import json
import base64
import io


class CoTVMCQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example['prompt']
        answer = example['response']
        image = self.decode_base64_to_image(example['image'])
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image

    @staticmethod
    def decode_base64_to_image(base64_string):
        decoded_string = io.BytesIO(base64.b64decode(base64_string))
        img = Image.open(decoded_string)
        return img

# GPT generated TODO: edit prompts!!
def format_question_for_llava(question: str) -> str:
    """Format the question to ensure proper output structure."""
    formatted_prompt = f"""{question}

    Please analyze the image and provide:
    1. Step-by-step reasoning for how to arrive at the answer
    2. Your final answer between two ## marks
    
    For example, your response should be structured like:
    Step 1: [First observation]
    Step 2: [Second observation]
    Step 3: [Third observation]
    ...
    Therefore, based on this reasoning...
    
    ## [A/B/C/D] ##
    
    Remember to:
    - Explain your thinking process clearly
    - Support your answer with visual evidence from the image
    - Put your final answer between ## marks
    """
    return formatted_prompt
class ImageQuestionProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_single_image(self, question: str, image: Image.Image) -> Dict:
        """Process a single image-question pair using Ollama's LLaVA model."""
        try:
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image_path = tmp.name
                image.save(image_path, format='JPEG', quality=95)
                print(f"Saved temporary image to: {image_path}")

                message = {
                    'role': 'user',
                    'content': question,
                    'images': [image_path]
                }

                print("Sending request to Ollama...")
                response = ollama.chat(
                    model="llava",
                    messages=[message]
                )

                answer = response['message']['content']
                print(f"Received response from Ollama: {answer}")

                return {
                    "question": question,
                    "answer": answer
                }

        except Exception as e:
            print(f"Error processing question: {question}\nError: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}"
            }
        finally:
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed temporary file: {image_path}")

    def process_dataset(self, dataset_name: str = "JiayiHe/SELFCORSET", batch_size: int = 4):
        """Process entire dataset and generate hints"""
        print(f"Loading dataset: {dataset_name}")

        # Load and prepare dataset
        working_indices = list(range(1000, 1200)) + list(range(3000, 4000)) + \
                          list(range(4000, 4500)) + list(range(5000, 6000))
        data = load_dataset(dataset_name, split="train")
        working_data = data.select(working_indices)
        initial_fine_tune_data = working_data.train_test_split(test_size=0.3)
        train_data = initial_fine_tune_data['train']

        # Create dataset instance
        dataset = CoTVMCQADataset(train_data)

        # Create dataloader
        def collate_fn(batch):
            questions, answers, images = zip(*batch)
            return list(questions), list(answers), list(images)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Process batches and collect results
        all_results = []
        for batch_idx, (questions, answers, images) in enumerate(dataloader):
            print(f"\nProcessing batch {batch_idx + 1}/{len(dataloader)}")

            for q, a, img in zip(questions, answers, images):
                result = self.process_single_image(q, img)
                result['original_answer'] = a
                all_results.append(result)

        return all_results

    def save_results(self, results: List[Dict], output_file: str = "hints.json"):
        """Save results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")


def main():
    processor = ImageQuestionProcessor()

    # Choose processing mode
    mode = input("Choose mode (1: Single Image, 2: Dataset Processing): ")

    if mode == "1":
        # Single image processing
        image_path = input("Enter image path (or press Enter for default C:\\python\\llava_test.png): ")
        if not image_path:
            image_path = r"C:\python\llava_test.png"

        question = input("Enter question (or press Enter for default): ")
        if not question:
            question = "What can you see in this image?"

        print(f"\nProcessing single image from: {image_path}")
        try:
            image = Image.open(image_path)
            result = processor.process_single_image(question, image)
            print("\nResults:")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
        except Exception as e:
            print(f"Error processing image: {e}")

    elif mode == "2":
        # Dataset processing
        print("\nProcessing dataset...")
        batch_size = int(input("Enter batch size (default 4): ") or 4)
        results = processor.process_dataset(batch_size=batch_size)

        # Save results
        output_file = input("Enter output file name (default: hints.json): ") or "hints.json"
        processor.save_results(results, output_file)

    else:
        print("Invalid mode selected")


if __name__ == "__main__":
    main()