import torch
from PIL import Image
import ollama
import tempfile
import os
from typing import List, Dict


class ImageQuestionProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_single_image(self, question: str, image: Image.Image) -> Dict:
        """Process a single image-question pair using Ollama's LLaVA model."""
        try:
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                # Paste using alpha channel as mask
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image_path = tmp.name
                image.save(image_path, format='JPEG', quality=95)
                print(f"Saved temporary image to: {image_path}")

                # Prepare message for Ollama
                message = {
                    'role': 'user',
                    'content': question,
                    'images': [image_path]
                }

                print("Sending request to Ollama...")
                # Call Ollama's chat API
                response = ollama.chat(
                    model="llava",
                    messages=[message]
                )

                # Extract response
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
            # Cleanup temporary file
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed temporary file: {image_path}")


def main():
    # Initialize processor
    processor = ImageQuestionProcessor()

    # Fix the file path format
    image_path = r"C:\python\llava_test.png"  # Use raw string to handle Windows paths
    print(f"Opening image from: {image_path}")

    # Open and process image
    try:
        image = Image.open(image_path)
        question = "What can you see in this image?"

        result = processor.process_single_image(question, image)
        print("\nResults:")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"Error opening image: {e}")


if __name__ == "__main__":
    main()