
import argparse
import torch
from diffusers import StableDiffusionPipeline
import os

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
DEFAULT_OUTPUT_DIR = "sd-oldbook-model/unet"

def parse_args():
    parser = argparse.ArgumentParser(description="Script de Inferencia para Modelo Fine-tuneado")
    parser.add_argument("--unet_path", type=str, default=DEFAULT_OUTPUT_DIR, help="Ruta a la carpeta de la UNet entrenada")
    parser.add_argument("--prompt", type=str, default="a flying cat over a city", help="Prompt para generar imagen")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--output_image", type=str, default="generated_image.png")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    print(f"Cargando pipeline base: {MODEL_NAME}")
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    
    if os.path.exists(args.unet_path):
        print(f"Cargando UNet fine-tuneada desde {args.unet_path}...")
        # Cargar UNet personalizada
        try:
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=torch.float32)
            pipeline.unet = unet
        except Exception as e:
            print(f"Error cargando UNet: {e}")
            return
    else:
        print(f"Advertencia: No se encontró path {args.unet_path}, usando modelo base.")

    pipeline.to(device)

    print(f"Generando imagen para: '{args.prompt}'")
    images = pipeline(args.prompt, num_inference_steps=50, num_images_per_prompt=args.num_images).images

    for i, img in enumerate(images):
        save_path = f"{args.output_image.replace('.png', '')}_{i}.png"
        img.save(save_path)
        print(f"Imagen guardada en: {save_path}")

if __name__ == "__main__":
    main()
