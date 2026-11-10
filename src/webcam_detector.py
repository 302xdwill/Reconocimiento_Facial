"""
Detector de Emociones Faciales en Tiempo Real con Cámara Web y YOLO
Curso: Inteligencia Artificial - UPeU 2026
"""

import sys
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

# Ajuste de rutas para importar utils independientemente de dónde se ejecute
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils import abrir_camara, dibujar_hud, obtener_color_emocion, traducir_emocion

def buscar_mejor_modelo():
    """Busca automáticamente los pesos entrenados más recientes."""
    posibles = [
        BASE_DIR / "runs" / "facial_emotions_exp" / "weights" / "best.pt",
        BASE_DIR / "runs" / "facial_emotions_exp" / "weights" / "last.pt",
        BASE_DIR / "models" / "best.pt",
        BASE_DIR / "models" / "last.pt",
        BASE_DIR / "runs" / "test_quick_train" / "weights" / "best.pt",
        Path("yolo11n.pt")
    ]
    for p in posibles:
        if p.exists():
            return str(p)
    return "yolo11n.pt"

class EmotionWebcamDetector:
    def __init__(self, model_path=None, cam_index=0, conf_threshold=0.30):
        self.model_path = model_path or buscar_mejor_modelo()
        self.cam_index = cam_index
        self.conf_threshold = conf_threshold
        self.show_hud = True
        self.screenshot_counter = 1

        print("=" * 65)
        print("  RECONOCIMIENTO FACIAL DE EMOCIONES EN VIVO - UPeU AI")
        print("=" * 65)
        print(f"[INFO] Cargando modelo: {self.model_path}")
        self.model = YOLO(self.model_path)
        print(f"[OK] Modelo cargado. Clases: {list(self.model.names.values())}")

    def iniciar(self):
        cap = abrir_camara(self.cam_index)
        if cap is None:
            print("[ERROR] No se pudo acceder a ninguna cámara web.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = "Reconocimiento Facial de Emociones - UPeU"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "-" * 65)
        print("CONTROLES ACTIVOS EN LA VENTANA:")
        print("  [q] / [ESC] : Salir del programa")
        print("  [s]         : Guardar captura de pantalla con las detecciones")
        print("  [+] / [-]   : Aumentar o disminuir el umbral de confianza")
        print("  [h]         : Ocultar o mostrar el banner HUD superior")
        print("-" * 65 + "\n")

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
                prev_time = curr_time

                # Inferencia con YOLO
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    verbose=False,
                    device='cpu'
                )

                # Procesar cajas y etiquetas personalizadas
                detecciones_por_emocion = {}
                annotated_frame = frame.copy()

                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        raw_name = self.model.names.get(cls_id, f"Clase_{cls_id}")
                        nombre_es = traducir_emocion(raw_name)

                        # Acumular conteo
                        detecciones_por_emocion[raw_name] = detecciones_por_emocion.get(raw_name, 0) + 1

                        # Coordenadas
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = obtener_color_emocion(raw_name)

                        # Dibujar caja con bordes redondeados/estilizados
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        # Etiqueta de emoción con fondo
                        label = f"{nombre_es}: {conf*100:.0f}%"
                        (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - h_txt - 10), (x1 + w_txt + 8, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # Dibujar HUD
                if self.show_hud:
                    nombre_corto = Path(self.model_path).stem
                    annotated_frame = dibujar_hud(
                        annotated_frame,
                        fps=fps,
                        conf_threshold=self.conf_threshold,
                        detecciones=detecciones_por_emocion,
                        nombre_modelo=nombre_corto
                    )

                cv2.imshow(window_name, annotated_frame)

                # Gestión de teclas
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    print("[INFO] Cerrando aplicación de visión...")
                    break
                elif key == ord('s'):
                    save_path = BASE_DIR / f"captura_emocion_{self.screenshot_counter}.jpg"
                    cv2.imwrite(str(save_path), annotated_frame)
                    print(f"[OK] Captura guardada: {save_path.name}")
                    self.screenshot_counter += 1
                elif key == ord('+'):
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"[INFO] Umbral aumentado a: {self.conf_threshold:.2f}")
                elif key == ord('-'):
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"[INFO] Umbral reducido a: {self.conf_threshold:.2f}")
                elif key == ord('h'):
                    self.show_hud = not self.show_hud

        except KeyboardInterrupt:
            print("\n[INFO] Interrupción por teclado.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Recursos de cámara liberados correctamente.")

def main():
    modelo = sys.argv[1] if len(sys.argv) > 1 else None
    detector = EmotionWebcamDetector(model_path=modelo)
    detector.iniciar()

if __name__ == "__main__":
    main()
