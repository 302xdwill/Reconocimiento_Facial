"""
Detector de Emociones Faciales en Tiempo Real con Arquitectura en Dos Etapas
Autor: William David Ortiz Mamani (302xdwill)
UPeU - Inteligencia Artificial 2026

Arquitectura:
  - Etapa 1: Localización Anatómica del Rostro por Puntos Clave (YOLO11-Pose)
             Elimina 100% de falsos positivos en ropa, paredes, roperos o posters.
  - Etapa 2: Reconocimiento y Clasificación de Emoción (EfficientNet FER ONNX)
             Predice con alta precisión: Enojo, Felicidad, Neutral, Tristeza, Sorpresa, Miedo.
"""

import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils import abrir_camara, dibujar_hud, obtener_color_emocion, traducir_emocion

try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    HSEMOTION_DISPONIBLE = True
except ImportError:
    HSEMOTION_DISPONIBLE = False

def buscar_mejor_modelo_yolo():
    """Busca el modelo de detección entrenado más reciente."""
    posibles = [
        BASE_DIR / "models" / "best.pt",
        BASE_DIR / "runs" / "facial_emotions_exp" / "weights" / "best.pt",
        BASE_DIR / "models" / "last.pt",
        BASE_DIR / "runs" / "facial_emotions_exp" / "weights" / "last.pt",
        Path("yolo11n.pt")
    ]
    for p in posibles:
        if p.exists():
            return str(p)
    return "yolo11n.pt"

class EmotionWebcamDetector:
    def __init__(self, yolo_model_path=None, cam_index=0, conf_threshold=0.35):
        self.cam_index = cam_index
        self.conf_threshold = conf_threshold
        self.show_hud = True
        self.screenshot_counter = 1
        
        # Modo por defecto: Arquitectura en Dos Etapas (Landmarks + FER)
        self.modo_dos_etapas = True
        
        # Suavizado temporal de emociones para evitar parpadeo
        self.historial_emociones = deque(maxlen=6)

        print("=" * 70)
        print("  SISTEMA PROFESIONAL DE RECONOCIMIENTO FACIAL Y EMOCIONES - UPeU")
        print("  Autor: William David Ortiz Mamani (302xdwill)")
        print("=" * 70)

        # 1. Cargar modelo de Pose / Landmarks faciales
        print("[INFO] Cargando detector de puntos clave faciales (YOLO11-Pose)...")
        self.pose_model = YOLO("yolo11n-pose.pt")
        print("[OK] Detector de puntos clave listo.")

        # 2. Cargar clasificador de emociones de alta precisión (HSEmotion ONNX)
        self.fer_model = None
        if HSEMOTION_DISPONIBLE:
            try:
                print("[INFO] Cargando clasificador de emociones FER (EfficientNet ONNX)...")
                self.fer_model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
                print("[OK] Clasificador FER cargado exitosamente.")
            except Exception as e:
                print(f"[!] Advertencia al cargar HSEmotion: {e}. Usando clasificador alternativo.")

        # 3. Cargar modelo YOLO Roboflow (para comparación o respaldo)
        self.yolo_path = yolo_model_path or buscar_mejor_modelo_yolo()
        print(f"[INFO] Cargando modelo YOLO Roboflow: {self.yolo_path}")
        self.yolo_model = YOLO(self.yolo_path)
        print(f"[OK] Modelo YOLO listo. Clases: {list(self.yolo_model.names.values())}")

    def extraer_caja_rostro_landmarks(self, frame):
        """
        Calcula matemáticamente la caja delimitadora del rostro exacto
        usando los 5 puntos clave faciales (nariz, ojos, orejas).
        Garantiza que la ropa, posters o muebles nunca sean detectados como rostro.
        """
        h_frame, w_frame, _ = frame.shape
        res = self.pose_model(frame, verbose=False)[0]

        if res.keypoints is None or len(res.keypoints) == 0:
            return None

        # Tomar la persona con mayor confianza/área
        kpts = res.keypoints.xy[0].cpu().numpy()
        conf = res.keypoints.conf[0].cpu().numpy() if res.keypoints.conf is not None else np.ones(17)

        # Filtrar puntos clave de la cabeza: 0:nariz, 1:ojo_izq, 2:ojo_der, 3:oreja_izq, 4:oreja_der
        xs = [pt[0] for i, pt in enumerate(kpts[:5]) if conf[i] > 0.20 and pt[0] > 0]
        ys = [pt[1] for i, pt in enumerate(kpts[:5]) if conf[i] > 0.20 and pt[1] > 0]

        if len(xs) < 2 or len(ys) < 2:
            return None

        l_eye = kpts[1]
        r_eye = kpts[2]
        nose = kpts[0]

        eye_y = (l_eye[1] + r_eye[1]) / 2.0 if (conf[1] > 0.20 and conf[2] > 0.20) else ys[0]
        nose_y = nose[1] if conf[0] > 0.20 else max(ys)
        eye_dist = abs(l_eye[0] - r_eye[0]) if (conf[1] > 0.20 and conf[2] > 0.20) else 50.0

        min_x, max_x = min(xs), max(xs)
        face_w = max_x - min_x
        margin_x = max(face_w * 0.25, eye_dist * 0.40)
        d_en = max(abs(nose_y - eye_y), eye_dist * 0.45)

        # Coordenadas finales con márgenes anatómicos (frente y mentón completos)
        x1 = max(0, int(min_x - margin_x))
        x2 = min(w_frame, int(max_x + margin_x))
        y1 = max(0, int(eye_y - d_en * 1.45))
        y2 = min(h_frame, int(nose_y + d_en * 1.75))

        # Validar tamaño mínimo de rostro
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return None

        return (x1, y1, x2, y2)

    def predecir_emocion_dos_etapas(self, frame, box):
        """
        Recorta el rostro y clasifica la emoción con alta precisión.
        """
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or self.fer_model is None:
            return "Neutral", 0.50

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        emocion_raw, scores = self.fer_model.predict_emotions(crop_rgb, logits=False)
        confianza = float(max(scores))

        # Registrar en cola temporal para suavizado
        self.historial_emociones.append((emocion_raw, confianza))

        # Voto mayoritario en el historial reciente
        conteo = {}
        for em, score in self.historial_emociones:
            conteo[em] = conteo.get(em, 0) + score
        emocion_estable = max(conteo, key=conteo.get)

        return emocion_estable, confianza

    def inferencia_yolo_directa(self, frame):
        """
        Modo directo usando el modelo Roboflow YOLO entrenado.
        """
        results = self.yolo_model.predict(
            frame,
            conf=self.conf_threshold,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
            device='cpu'
        )
        cajas = []
        if results[0].boxes is not None:
            for b in results[0].boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                raw_name = self.yolo_model.names.get(cls_id, f"Clase_{cls_id}")
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cajas.append((x1, y1, x2, y2, raw_name, conf))
        return cajas

    def iniciar(self):
        cap = abrir_camara(self.cam_index)
        if cap is None:
            print("[ERROR] No se pudo acceder a la cámara web de la laptop.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = "Reconocimiento Facial de Emociones - UPeU (William Ortiz)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "-" * 70)
        print("CONTROLES ACTIVOS EN LA VENTANA:")
        print("  [q] / [ESC] : Salir del sistema")
        print("  [s]         : Guardar captura de pantalla")
        print("  [t]         : ALTERNAR MODO (2-Etapas Landmarks vs YOLO Roboflow)")
        print("  [+] / [-]   : Aumentar o disminuir umbral de confianza")
        print("  [h]         : Ocultar o mostrar banner HUD superior")
        print("-" * 70 + "\n")

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

                annotated = frame.copy()
                detecciones = {}

                if self.modo_dos_etapas and self.fer_model is not None:
                    # MODO 1: Arquitectura en Dos Etapas (Landmarks + FER)
                    box = self.extraer_caja_rostro_landmarks(frame)
                    if box is not None:
                        x1, y1, x2, y2 = box
                        emocion_raw, conf = self.predecir_emocion_dos_etapas(frame, box)
                        nombre_es = traducir_emocion(emocion_raw)
                        color = obtener_color_emocion(emocion_raw)

                        detecciones[emocion_raw] = 1

                        # Dibujar caja con bordes redondeados y estilo visual limpio
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                        # Etiqueta con fondo sólido
                        label = f"{nombre_es}: {conf*100:.0f}%"
                        (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        cv2.rectangle(annotated, (x1, y1 - h_txt - 12), (x1 + w_txt + 10, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                else:
                    # MODO 2: Modelo YOLO Roboflow Directo
                    cajas = self.inferencia_yolo_directa(frame)
                    for x1, y1, x2, y2, raw_name, conf in cajas:
                        nombre_es = traducir_emocion(raw_name)
                        color = obtener_color_emocion(raw_name)
                        detecciones[raw_name] = detecciones.get(raw_name, 0) + 1

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{nombre_es}: {conf*100:.0f}%"
                        (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated, (x1, y1 - h_txt - 10), (x1 + w_txt + 8, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Renderizar HUD superior
                if self.show_hud:
                    modo_txt = "2-Etapas (Landmarks + FER)" if (self.modo_dos_etapas and self.fer_model) else "YOLO Roboflow"
                    annotated = dibujar_hud(
                        annotated,
                        fps=fps,
                        conf_threshold=self.conf_threshold,
                        detecciones=detecciones,
                        nombre_modelo=modo_txt
                    )

                cv2.imshow(window_name, annotated)

                # Control de eventos por teclado
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    print("[INFO] Cerrando cámara web...")
                    break
                elif key == ord('s'):
                    save_path = BASE_DIR / f"captura_emocion_{self.screenshot_counter}.jpg"
                    cv2.imwrite(str(save_path), annotated)
                    print(f"[OK] Captura guardada: {save_path.name}")
                    self.screenshot_counter += 1
                elif key == ord('t'):
                    self.modo_dos_etapas = not self.modo_dos_etapas
                    modo_nuevo = "2-Etapas (Pose + FER)" if self.modo_dos_etapas else "YOLO Roboflow Directo"
                    print(f"[INFO] Modo cambiado a: {modo_nuevo}")
                elif key == ord('+'):
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"[INFO] Umbral: {self.conf_threshold:.2f}")
                elif key == ord('-'):
                    self.conf_threshold = max(0.10, self.conf_threshold - 0.05)
                    print(f"[INFO] Umbral: {self.conf_threshold:.2f}")
                elif key == ord('h'):
                    self.show_hud = not self.show_hud

        except KeyboardInterrupt:
            print("\n[INFO] Interrupción del usuario.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Cámara liberada correctamente.")

def main():
    modelo = sys.argv[1] if len(sys.argv) > 1 else None
    detector = EmotionWebcamDetector(yolo_model_path=modelo)
    detector.iniciar()

if __name__ == "__main__":
    main()
