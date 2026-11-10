"""
Utilidades y Helpers para Visión Artificial y Procesamiento de Emociones
"""

import cv2
import numpy as np

# Mapeo de colores BGR para cada emoción
EMOTION_COLORS = {
    "happy":    (50, 205, 50),    # Verde esmeralda (Felicidad)
    "anger":    (40, 40, 240),    # Rojo brillante (Enojo)
    "surprise": (0, 215, 255),    # Amarillo dorado (Sorpresa)
    "sad":      (235, 145, 30),   # Azul cobalto (Tristeza)
    "fear":     (180, 50, 180),   # Púrpura (Miedo)
    "neutral":  (200, 200, 200)   # Gris platino (Neutral)
}

# Traducción pedagógica al español
EMOTION_ES = {
    "anger": "Enojo",
    "fear": "Miedo",
    "happy": "Felicidad",
    "neutral": "Neutral",
    "sad": "Tristeza",
    "surprise": "Sorpresa"
}

def obtener_color_emocion(nombre_emocion):
    """Retorna el color BGR para la emoción especificada."""
    return EMOTION_COLORS.get(nombre_emocion.lower(), (0, 255, 0))

def traducir_emocion(nombre_emocion):
    """Traduce el nombre de la emoción a español."""
    return EMOTION_ES.get(nombre_emocion.lower(), nombre_emocion)

def abrir_camara(indice_primario=0, indices_alternativos=(1, 2)):
    """
    Intenta abrir la cámara web en el índice primario o busca en los alternativos.
    """
    cap = cv2.VideoCapture(indice_primario)
    if cap.isOpened():
        print(f"[OK] Cámara web conectada exitosamente en índice {indice_primario}.")
        return cap

    print(f"[!] No se pudo abrir la cámara en el índice {indice_primario}. Probando alternativos...")
    for alt_idx in indices_alternativos:
        cap = cv2.VideoCapture(alt_idx)
        if cap.isOpened():
            print(f"[OK] Cámara web conectada en índice alternativo {alt_idx}.")
            return cap

    return None

def dibujar_hud(frame, fps, conf_threshold, detecciones, nombre_modelo):
    """
    Superpone un banner HUD estilizado con métricas en tiempo real.
    """
    h, w, _ = frame.shape
    
    # Crear overlay semi-transparente en la parte superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 85), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Borde exterior del HUD
    cv2.rectangle(frame, (10, 10), (w - 10, 85), (70, 70, 80), 1)

    # Texto línea 1: Estado del Sistema
    fps_color = (0, 255, 120) if fps >= 10 else (0, 200, 255)
    info_line1 = f"FPS: {fps:.1f}  |  Confianza Min: {conf_threshold:.2f}  |  Modelo: {nombre_modelo}"
    cv2.putText(frame, info_line1, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2)

    # Resumen de Emociones detectadas
    if detecciones:
        resumen_texto = "  |  ".join([f"{traducir_emocion(k)}: {v}" for k, v in detecciones.items()])
        color_texto = (0, 255, 255)
    else:
        resumen_texto = "Sin rostros / emociones detectadas actualmente"
        color_texto = (160, 160, 160)

    cv2.putText(frame, resumen_texto, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_texto, 2)

    return frame
