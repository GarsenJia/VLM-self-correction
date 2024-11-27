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


def format_question_for_llava(question: str) -> str:
    """Format the question to ensure proper hint generation without answers."""
    formatted_prompt = f"""You are a helpful visual reasoning assistant.

Please analyze the image and generate helpful hints to answer the question provided. **Do not** include or reveal the direct answer. Your role is to guide others toward arriving at the correct answer through insightful hints.

**Question:**

{question}

**Response Structure:**

- **Hint 1:** [First observation or clue based on the image]
- **Hint 2:** [Second observation or clue building upon the first and leading closer to the answer]

**Guidelines:**

- Be clear and concise in your explanations.
- Use relevant visual evidence from the image to support each hint.
- Ensure each hint logically leads to the next, gradually guiding toward the answer *without* stating it.
- Avoid unnecessary details that do not contribute to understanding the image or question.

"""
    return formatted_prompt



class ImageQuestionProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_single_image(self, question: str, image: Image.Image) -> Dict:
        """Process a single image-question pair using Ollama's LLaVA model."""
        try:
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

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
        """Process first 10 questions from the dataset with modified output format"""
        print(f"Loading dataset: {dataset_name}")

        # Load dataset and take first 10 items
        data = load_dataset(dataset_name, split="train")
        working_data = data.select(range(500))  # Only select first 500 items

        # Create dataset instance
        dataset = CoTVMCQADataset(working_data)

        # extract hint: only takes less than 2 hints. in case of parsing failure, put the whole answer and don't throw error
        def extract_hints(response: str) -> str:
            """Extract hints from response, return full response if parsing fails."""
            try:
                hints = []
                for line in response.split('\n'):
                    if ('Hint 1:' in line or '**Hint 1:**' in line or
                            'Hint 2:' in line or '**Hint 2:**' in line):
                        hint = line.replace('**', '').strip()
                        hints.append(hint)

                if not hints:
                    for line in response.split('\n'):
                        if line.strip().startswith(('Step 1:', '- Step 1:', 'Step 2:', '- Step 2:')):
                            hints.append(line.strip())

                hints = hints[:2]
                return '\n\n'.join(hints) if hints else response
            except Exception as e:
                print(f"Warning: Error parsing hints: {e}. Using full response instead.")
                return response

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
                formatted_q = format_question_for_llava(q)  # Format the question
                llava_result = self.process_single_image(formatted_q, img)

                parsed_hints = extract_hints(llava_result['answer'])
                # Format the result with combined question and hint
                result = {
                    "question": f"{q}\n\nYou have the following hint:\n{parsed_hints}",
                    "answer": a
                }

                all_results.append(result)
                print(f"Processed {len(all_results)}/10 questions")

        return all_results

    def save_results(self, results: List[Dict], output_file: str = "hints_10.json"):
        """Save results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")


def main():
    processor = ImageQuestionProcessor()
    print("\nProcessing first 10 questions from dataset...")
    results = processor.process_dataset(batch_size=2)  # Smaller batch size for testing
    processor.save_results(results, "hints.json")


if __name__ == "__main__":
    main()