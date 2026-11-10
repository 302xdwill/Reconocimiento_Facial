# Reconocimiento_Facial

## Sistema de Reconocimiento Facial y Clasificación de Emociones en Tiempo Real (YOLO11)

**Curso:** Inteligencia Artificial  
**Institución:** Universidad Peruana Unión (UPeU)  
**Autor:** David (302xdwill)  
**Versión:** 1.0.0  

---

## 📌 Descripción del Proyecto

Este proyecto implementa una solución integral y optimizada de **Visión Artificial y Aprendizaje Profundo (Deep Learning)** para la detección y clasificación en tiempo real de **6 emociones faciales humanas** utilizando la arquitectura de última generación **YOLO11** (Ultralytics) y **OpenCV**.

El sistema procesa la señal de video de la cámara web integrada de la laptop a alta velocidad, detectando los rostros y clasificando su estado emocional instantáneamente con cuadros delimitadores coloreados según la emoción y un panel frontal (HUD) interactivo.

---

## 🎭 Clases y Emociones Reconocidas

El modelo ha sido entrenado para identificar con precisión 6 estados emocionales:

| ID | Emoción (Etiqueta) | Traducción al Español | Color Visual en Pantalla |
|:---:|:---|:---|:---:|
| **0** | `anger` | Ira / Enojo | 🔴 Rojo brillante |
| **1** | `fear` | Miedo | 🟣 Púrpura |
| **2** | `happy` | Felicidad | 🟢 Verde esmeralda |
| **3** | `neutral` | Neutralidad | ⚪ Gris / Platino |
| **4** | `sad` | Tristeza | 🔵 Azul cobalto |
| **5** | `surprise` | Sorpresa | 🟡 Amarillo dorado |

---

## 📂 Arquitectura y Estructura del Proyecto

El proyecto está diseñado bajo un enfoque modular, limpio y escalable:

```
Renococimiento Facial/
├── config/
│   └── data.yaml                   # Configuración del dataset, rutas y clases
├── src/
│   ├── __init__.py                 # Paquete de Python
│   ├── utils.py                    # Utilidades de cámara, colores BGR y HUD
│   ├── webcam_detector.py          # Módulo de visión artificial en vivo con webcam
│   ├── train.py                    # Script modular de entrenamiento con argumentos
│   └── evaluate.py                 # Script de evaluación de métricas e inferencia
├── models/
│   ├── best.pt                     # Pesos con mejor desempeño mAP guardados
│   └── last.pt                     # Pesos de la última época de entrenamiento
├── samples/                        # Fotos de muestra para pruebas rápidas
├── runs/                           # Registro de experimentos, gráficas y matrices
│   └── facial_emotions_exp/        # Resultados del entrenamiento de emociones
│       ├── results.png             # Curvas de convergencia y pérdidas
│       ├── confusion_matrix.png    # Matriz de confusión entre emociones
│       └── weights/                # Pesos generados durante el entrenamiento
├── face emotions.v1i.yolo26/       # Dataset etiquetado (1,455 imágenes)
├── main.py                         # Punto de entrada principal con menú interactivo
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Exclusión de archivos masivos y temporales
└── README.md                       # Documentación técnica
```

---

## ⚙️ Instalación y Configuración del Entorno

1. **Abrir la terminal PowerShell en la carpeta del proyecto:**
   ```powershell
   cd "D:\Documentos\Trabajos upeu\Trabajos upeu\C8\IA\Renococimiento Facial"
   ```

2. **Activar el entorno virtual de Python:**
   ```powershell
   ..\env\Scripts\activate
   ```

3. **Verificar dependencias instaladas:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🚀 Guía de Uso Rápido

### Opción A: Menú Interactivo (Recomendado)
Ejecuta simplemente el script maestro:
```powershell
python main.py
```
Aparecerá un menú con opciones numéricas para iniciar la cámara web, entrenar o evaluar el modelo.

---

### Opción B: Ejecución Directa por Comandos

#### 1. Iniciar la Cámara Web en Tiempo Real:
```powershell
python main.py webcam
# O directamente:
python src/webcam_detector.py
```

#### Controles de Teclado en Vivo:
* **`q`** o **`ESC`**: Finalizar y cerrar la ventana de video de forma segura.
* **`s`**: Tomar una captura de pantalla instantánea con las emociones trazadas (`captura_emocion_X.jpg`).
* **`+`** / **`-`**: Aumentar o disminuir dinámicamente el umbral de confianza en vivo.
* **`h`**: Ocultar o mostrar el panel HUD superior.

#### 2. Entrenar el Modelo:
```powershell
python main.py train
# O con parámetros personalizados:
python src/train.py --epochs 10 --batch 8 --fraction 0.30
```

#### 3. Evaluar el Desempeño:
```powershell
python main.py eval
# O directamente:
python src/evaluate.py
```
Generará predicciones visuales en la carpeta `runs/eval_predictions/` mostrando las caras detectadas con sus emociones y porcentajes de certeza.

---

## 💻 Hardware y Diagnóstico Técnico

* **Hardware Detectado:** Procesador móvil **AMD Ryzen 5 4500U con gráficos integrados AMD Radeon(TM) Graphics**.
* **Compatibilidad CUDA:** Al tratarse de gráficos AMD, **CUDA** (tecnología propietaria de NVIDIA) no está soportada.
* **Optimización Aplicada:** Se utiliza **PyTorch optimizado para CPU** (`torch-2.15.0+cpu`) acoplado a **YOLO11 Nano (`yolo11n.pt`)**, el cual requiere únicamente 2.6 millones de parámetros. Esto garantiza una inferencia fluida y en tiempo real (~130 a 170 ms por fotograma en CPU).

---
**Universidad Peruana Unión (UPeU) - 2026**
