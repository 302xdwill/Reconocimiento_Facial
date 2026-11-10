"""
Evaluación de Desempeño e Inferencia sobre Imágenes de Prueba
Curso: Inteligencia Artificial - UPeU 2026
"""

import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils import obtener_color_emocion, traducir_emocion

def evaluar_modelo(weights_path=None):
    if weights_path is None:
        posibles = [
            BASE_DIR / "models" / "best.pt",
            BASE_DIR / "runs" / "facial_emotions_exp" / "weights" / "best.pt",
            BASE_DIR / "runs" / "test_quick_train" / "weights" / "best.pt"
        ]
        for p in posibles:
            if p.exists():
                weights_path = p
                break

    if weights_path is None or not Path(weights_path).exists():
        print("[ERROR] No se encontraron pesos entrenados para evaluar.")
        return

    print("=" * 65)
    print(f"  EVALUACIÓN DEL MODELO: {weights_path}")
    print("=" * 65)

    model = YOLO(str(weights_path))

    # 1. Validación Formal con Dataset
    data_yaml = BASE_DIR / "config" / "data.yaml"
    if data_yaml.exists():
        print("\n[INFO] Calculando métricas en el conjunto de validación...")
        val_res = model.val(data=str(data_yaml), verbose=False)
        print(f"  -> mAP@50:    {val_res.box.map50:.4f}")
        print(f"  -> mAP@50-95: {val_res.box.map:.4f}")

    # 2. Inferencia Visual en Muestras de Test
    test_dir = BASE_DIR / "face emotions.v1i.yolo26" / "test" / "images"
    if not test_dir.exists():
        test_dir = BASE_DIR / "samples"

    out_dir = BASE_DIR / "runs" / "eval_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_imgs = list(test_dir.glob("*.jpg"))[:5]
    print(f"\n[INFO] Generando inferencias en {len(test_imgs)} imágenes de prueba...")

    for idx, img_path in enumerate(test_imgs):
        img = cv2.imread(str(img_path))
        results = model.predict(img, conf=0.25, verbose=False)

        annotated = img.copy()
        emociones_detectadas = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                raw_name = model.names.get(cls_id, f"Cls_{cls_id}")
                nombre_es = traducir_emocion(raw_name)
                emociones_detectadas.append(f"{nombre_es} ({conf*100:.0f}%)")

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = obtener_color_emocion(raw_name)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"{nombre_es} {conf*100:.0f}%",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        save_name = out_dir / f"prediccion_{idx+1}_{img_path.stem[:12]}.jpg"
        cv2.imwrite(str(save_name), annotated)
        print(f"  [Foto {idx+1}] {save_name.name} -> Detecciones: {emociones_detectadas}")

    print("\n[OK] Evaluaciones guardadas en:", out_dir)

if __name__ == "__main__":
    modelo = sys.argv[1] if len(sys.argv) > 1 else None
    evaluar_modelo(weights_path=modelo)
