"""
Script Modular de Entrenamiento para Detección de Emociones Faciales con YOLO11
Curso: Inteligencia Artificial - UPeU 2026
"""

import argparse
import shutil
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

def entrenar_emociones(epochs=5, batch=8, imgsz=416, fraction=0.2, experiment_name="facial_emotions_exp"):
    print("=" * 65)
    print("  ENTRENAMIENTO YOLO11 - RECONOCIMIENTO DE EMOCIONES FACIALES")
    print("=" * 65)

    # 1. Diagnóstico de Hardware
    cuda_disponible = torch.cuda.is_available()
    device = 0 if cuda_disponible else "cpu"
    print(f"PyTorch:         {torch.__version__}")
    print(f"CUDA disponible: {cuda_disponible}")
    print(f"Dispositivo:     {device} ({torch.cuda.get_device_name(0) if cuda_disponible else 'AMD Radeon / CPU'})")

    # 2. Archivo de Configuración de Datos
    data_yaml = BASE_DIR / "config" / "data.yaml"
    if not data_yaml.exists():
        data_yaml = BASE_DIR / "face emotions.v1i.yolo26" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"[ERROR] No se encontró data.yaml en {data_yaml}")

    print(f"Configuración:   {data_yaml}")

    # 3. Cargar Arquitectura Base YOLO11 Nano
    modelo_base = "yolo11n.pt"
    print(f"Modelo Base:     {modelo_base}")
    model = YOLO(modelo_base)

    # 4. Parámetros de Entrenamiento
    print("\nParámetros:")
    print(f"  - Épocas:            {epochs}")
    print(f"  - Tamaño de Lote:    {batch}")
    print(f"  - Resolución Img:    {imgsz}")
    print(f"  - Fracción Dataset:  {fraction * 100:.1f}%")
    print(f"  - Proyecto Destino:  {BASE_DIR / 'runs'}")
    print("=" * 65 + "\n")

    # 5. Ejecución del Entrenamiento
    runs_dir = BASE_DIR / "runs"
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        fraction=fraction,
        device=device,
        project=str(runs_dir),
        name=experiment_name,
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("[ÉXITO] Entrenamiento finalizado correctamente.")
    print("=" * 65)

    # 6. Copia de Pesos a la Carpeta models/
    exp_dir = runs_dir / experiment_name
    best_weights = exp_dir / "weights" / "best.pt"
    last_weights = exp_dir / "weights" / "last.pt"
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
        shutil.copy2(best_weights, models_dir / "best.pt")
        print(f"[OK] Pesos óptimos respaldados en: {models_dir / 'best.pt'}")
    if last_weights.exists():
        shutil.copy2(last_weights, models_dir / "last.pt")
        print(f"[OK] Últimos pesos respaldados en: {models_dir / 'last.pt'}")

    # 7. Validación Final y Resumen de Métricas
    print("\n[INFO] Ejecutando validación formal...")
    metricas = model.val(data=str(data_yaml))
    print("\n--- RESUMEN DE RENDIMIENTO (MÉTRICAS) ---")
    print(f"mAP@50 (Global):    {metricas.box.map50:.4f}")
    print(f"mAP@50-95 (Global): {metricas.box.map:.4f}")

    return metricas

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenador de Emociones YOLO11")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño de lote")
    parser.add_argument("--imgsz", type=int, default=416, help="Tamaño de imagen")
    parser.add_argument("--fraction", type=float, default=0.20, help="Fracción del dataset (0.0 a 1.0)")
    parser.add_argument("--name", type=str, default="facial_emotions_exp", help="Nombre del experimento")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    entrenar_emociones(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        experiment_name=args.name
    )
