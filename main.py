"""
Punto de Entrada Principal - Sistema de Reconocimiento Facial de Emociones
Curso: Inteligencia Artificial - UPeU 2026
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.train import entrenar_emociones
from src.evaluate import evaluar_modelo
from src.webcam_detector import EmotionWebcamDetector

def mostrar_menu():
    print("\n" + "=" * 65)
    print("    SISTEMA DE RECONOCIMIENTO FACIAL DE EMOCIONES (YOLO11)")
    print("                 Universidad Peruana Unión (UPeU)")
    print("=" * 65)
    print("  [1] Iniciar Cámara Web en Tiempo Real (Reconocimiento en Vivo)")
    print("  [2] Entrenar Modelo de Emociones Faciales")
    print("  [3] Evaluar Modelo y Generar Predicciones sobre Fotos de Test")
    print("  [4] Salir")
    print("=" * 65)
    
    opcion = input("\nSeleccione una opción [1-4]: ").strip()
    return opcion

def main():
    if len(sys.argv) > 1:
        comando = sys.argv[1].lower()
        if comando == "webcam":
            detector = EmotionWebcamDetector()
            detector.iniciar()
        elif comando == "train":
            entrenar_emociones(epochs=5, batch=8, fraction=0.25)
        elif comando == "eval":
            evaluar_modelo()
        else:
            print(f"[!] Comando desconocido '{comando}'. Opciones válidas: webcam, train, eval")
        return

    # Menú Interactivo si no se pasaron argumentos
    while True:
        opc = mostrar_menu()
        if opc == "1":
            detector = EmotionWebcamDetector()
            detector.iniciar()
        elif opc == "2":
            print("\nIniciando entrenamiento del modelo...")
            epocas_input = input("Ingrese número de épocas [por defecto 5]: ").strip()
            epocas = int(epocas_input) if epocas_input.isdigit() else 5
            entrenar_emociones(epochs=epocas, batch=8, fraction=0.25)
        elif opc == "3":
            evaluar_modelo()
        elif opc == "4":
            print("\n¡Hasta luego!")
            break
        else:
            print("[!] Opción inválida. Por favor elija un número del 1 al 4.")

if __name__ == "__main__":
    main()
