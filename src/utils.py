"""
Utilidades y Helpers para Visión Artificial y Procesamiento de Emociones
Autor: William David Ortiz Mamani (302xdwill)
"""

import cv2
import numpy as np

# Mapeo de colores BGR para cada emoción
EMOTION_COLORS = {
    # Clases Roboflow y estándar
    "happy":       (50, 205, 50),    # Verde esmeralda (Felicidad)
    "happiness":   (50, 205, 50),    # Verde esmeralda (Felicidad)
    "anger":       (40, 40, 240),    # Rojo brillante (Enojo)
    "disgust":     (30, 70, 220),    # Rojo anaranjado (Disgusto/Enojo)
    "surprise":    (0, 215, 255),    # Amarillo dorado (Sorpresa)
    "sad":         (235, 145, 30),   # Azul cobalto (Tristeza)
    "sadness":     (235, 145, 30),   # Azul cobalto (Tristeza)
    "fear":        (180, 50, 180),   # Púrpura (Miedo)
    "neutral":     (200, 200, 200),  # Gris platino (Neutral)
    "contempt":    (150, 100, 210)   # Magenta apagado (Desprecio)
}

# Traducción pedagógica al español
EMOTION_ES = {
    "anger":       "Enojo",
    "disgust":     "Enojo / Molestia",
    "fear":        "Miedo",
    "happy":       "Felicidad",
    "happiness":   "Felicidad",
    "neutral":     "Neutral",
    "sad":         "Tristeza",
    "sadness":     "Tristeza",
    "surprise":    "Sorpresa",
    "contempt":    "Desprecio"
}

def obtener_color_emocion(nombre_emocion):
    """Retorna el color BGR para la emoción especificada."""
    return EMOTION_COLORS.get(str(nombre_emocion).lower(), (0, 255, 0))

def traducir_emocion(nombre_emocion):
    """Traduce el nombre de la emoción a español."""
    return EMOTION_ES.get(str(nombre_emocion).lower(), str(nombre_emocion).capitalize())

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
    cv2.rectangle(overlay, (10, 10), (w - 10, 95), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Borde exterior del HUD
    cv2.rectangle(frame, (10, 10), (w - 10, 95), (70, 70, 80), 1)

    # Texto línea 1: Estado del Sistema
    fps_color = (0, 255, 120) if fps >= 10 else (0, 200, 255)
    info_line1 = f"FPS: {fps:.1f}  |  Conf: {conf_threshold:.2f}  |  Pipeline: {nombre_modelo}"
    cv2.putText(frame, info_line1, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 2)

    # Resumen de Emociones detectadas
    if detecciones:
        resumen_texto = "  |  ".join([f"{traducir_emocion(k)}: {v}" for k, v in detecciones.items()])
        color_texto = (0, 255, 255)
    else:
        resumen_texto = "Esperando deteccion de rostro..."
        color_texto = (160, 160, 160)

    cv2.putText(frame, resumen_texto, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_texto, 2)

    # Línea 3: Guía rápida de teclas
    guia_teclas = "[q]: Salir | [s]: Captura | [t]: Alternar Modo 2-Etapas/YOLO | [+]/[-]: Confianza"
    cv2.putText(frame, guia_teclas, (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    return frame
