import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import base64
from tqdm import tqdm
import os

# Helper function to decode base64 to image
def decode_base64_to_image(base64_string):
    decoded_string = io.BytesIO(base64.b64decode(base64_string))
    img = Image.open(decoded_string)
    return img

# Dataset class
class CoTVMCQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<CoTVMCQA>" + example['prompt']
        answer = example['response']
        image = decode_base64_to_image(example['image'])
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image

# Collate function
def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

# Training function
def train_model(train_loader, val_loader, model, processor, epochs=6, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
            ).input_ids.to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, answers = batch
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
                ).input_ids.to(device)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch + 1 + 2}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

# Main function
def main():
    # Set device
    global device, processor  # Used in the `collate_fn`
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select working indices
    working_indices = list(range(1000, 1200)) + list(range(3000, 4000)) + list(range(4000, 4500)) + list(range(5000, 6000))

    # Load and split data
    data = load_dataset("JiayiHe/SELFCORSET", split="train")
    working_data = data.select(working_indices)
    # initial_fine_tune_data = working_data.train_test_split(test_size=0.3)
    florence_fine_tune_train = working_data # all data except 0-1000
    test_indices = list(range(0, 1000))
    test_data = data.select(test_indices)
    florence_fine_tune_valid = test_data

    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6').to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')

    # Create datasets and DataLoaders
    train_dataset = CoTVMCQADataset(florence_fine_tune_train)
    val_dataset = CoTVMCQADataset(florence_fine_tune_valid)
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)

    # Train the model
    train_model(train_loader, val_loader, model, processor, epochs=6)

    # Save final model and processor
    model.save_pretrained("Florence-2-CoTVMCQA_model_6_epochs")
    processor.save_pretrained("Florence-2-CoTVMCQA_processor_6_epochs")

if __name__ == "__main__":
    main()
