
import os
import random
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import torch
from transformers import CLIPTokenizer

# Configuración
DATASET_ID = "gigant/oldbookillustrations"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
IMAGE_SIZE = 512
NUM_SAMPLES = 200
SEED = 42

def prepare_dataset():
    print(f"Cargando dataset {DATASET_ID}...")
    # Cargar dataset (split train)
    dataset = load_dataset(DATASET_ID, split="train")
    
    # Seleccionar subconjunto aleatorio si es necesario, o los primeros N
    # Para reproducibilidad usamos shuffle con seed
    if len(dataset) > NUM_SAMPLES:
        dataset = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))
    
    print(f"Dataset reducido a {len(dataset)} muestras.")
    return dataset

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE), # O RandomCrop
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def preprocess_train(examples, tokenizer):
    images = [image.convert("RGB") for image in examples["1600px"]]
    captions = examples["info_alt"]
    
    train_transforms = get_train_transforms()
    pixel_values = [train_transforms(image) for image in images]
    
    # Tokenizar
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    
    return {
        "pixel_values": pixel_values,
        "input_ids": inputs.input_ids,
    }

if __name__ == "__main__":
    # Test simple
    try:
        dataset = prepare_dataset()
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        
        # Test de procesamiento de un batch
        processed = preprocess_train(dataset[:2], tokenizer)
        print("Shape de pixel_values:", torch.stack(processed["pixel_values"]).shape)
        print("Shape de input_ids:", processed["input_ids"].shape)
        print("Data Prep completado exitosamente.")
    except Exception as e:
        print(f"Error en data_prep: {e}")
