# Práctica Final: Fine-Tuning de Stable Diffusion

Proyecto de Fine-Tuning del modelo Stable Diffusion 1.4 para generar imágenes con estilo de "Ilustraciones de Libros Antiguos".

## Estructura del Proyecto

*   `src/data_prep.py`: Script para cargar y preprocesar el dataset `gigant/oldbookillustrations`.
*   `src/train.py`: Script principal de entrenamiento (Fine-Tuning de la UNet).
*   `src/inference.py`: Script para probar el modelo generativo.

## Requisitos

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Instrucciones de Uso

### 1. Entrenamiento

Para iniciar el entrenamiento (se recomienda hacerlo con GPU, o ajustar batch size para CPU):

```bash
# Ejecutar desde la raíz del proyecto
python src/train.py --batch_size 1 --epochs 2
```

Opciones:
*   `--batch_size`: Tamaño del lote (default 1).
*   `--epochs`: Número de épocas (default 2).
*   `--output_dir`: Carpeta donde se guardará el modelo.

### 2. Generación de Imágenes

Una vez entrenado (se guardará en `sd-oldbook-model/unet`), generar imágenes:

```bash
python src/inference.py --prompt "a beautiful landscape with a castle" --output_image "resultado.png"
```

## Notas

*   El entrenamiento completo puede tardar varias horas en CPU.
*   Se utiliza un subconjunto de 200 imágenes del dataset original para la demostración.
