
import argparse
import math
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, PNDMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from data_prep import prepare_dataset, preprocess_train

# Configuración por defecto
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
OUTPUT_DIR = "sd-oldbook-model"
TRAIN_BATCH_SIZE = 1 # Para asegurar que corra en CPU/Low RAM, usuario puede subirlo a 6
NUM_EPOCHS = 2
gradient_accumulation_steps = 4 
LEARNING_RATE = 1e-5

def parse_args():
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning de Stable Diffusion")
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Inicializar Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    set_seed(42)

    if accelerator.is_main_process:
        print(f"Iniciando entrenamiento en {accelerator.device}")
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. Cargar Modelos y Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

    # 2. Congelar VAE y Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Mover a device (Accelerator lo maneja, pero modelos congelados mejor moverlos manual si no se pasan a prepare)
    # Sin embargo, accelerate.prepare maneja el modelo optimizado (unet). 
    # VAE y Text Encoder no se entrenan, los movemos a device manualmente para inferencia en bucle
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # 3. Preparar Dataset
    dataset = prepare_dataset()
    
    def collate_fn(examples):
        # Re-estructurar lista de diccionarios a diccionario de listas
        # Nuestra funcion preprocess_train ya fue aplicada? No, la aplicamos con transform
        # Pero `datasets` map function es mejor. O lo hacemos al vuelo en el dataloader
        # Aqui simplificamos: aplicamos preprocess a todo el dataset primero o usamos set_transform
        return torch.stack([ex["pixel_values"] for ex in examples]), torch.stack([ex["input_ids"] for ex in examples])

    # Aplicar transformaciones on-the-fly
    with accelerator.main_process_first():
        # Usamos with_transform para aplicar el preprocesamiento
        def transform_wrapper(examples):
            return preprocess_train(examples, tokenizer)
        
        train_dataset = dataset.with_transform(transform_wrapper)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None # Default collate sirve si devuelve dicts de tensores?
    )
    # Ajuste: preprocess_train devuelve dict de listas. el collate default de torch puede fallar con dicts si no estan bien estructurados
    # Vamos a usar un collate custom simple
    def custom_collate(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate
    )

    # 4. Optimizador
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # 5. Prepare con Accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # 6. Loop de Entrenamiento
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    total_steps = args.epochs * num_update_steps_per_epoch

    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convertir imagenes a latentes
                # Latents shape: (batch_size, 4, 64, 64)
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample random timesteps
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get config prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backprop
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                train_loss += loss.item()
                logs = {"loss": loss.item()}
                progress_bar.set_postfix(**logs)
        
        print(f"Epoch {epoch} finalizada. Loss promedio: {train_loss / len(train_dataloader)}")

    # 7. Guardar Modelo
    if accelerator.is_main_process:
        pipeline = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler") 
        # Cuidado: para guardar un pipeline usable, necesitamos cargar todos los componentes en el pipeline final
        # Pero aqui solo hemos entrenado la UNet.
        # Guardaremos la UNet independientemente para cargarla luego en un pipeline base.
        print("Guardando UNet entrenada...")
        unwrap_unet = accelerator.unwrap_model(unet)
        unwrap_unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        print("Entrenamiento completado y modelo guardado.")

if __name__ == "__main__":
    main()
