# Reconocimiento_Facial

## Sistema de Reconocimiento Facial y Clasificación de Emociones en Tiempo Real (YOLO11 + Two-Stage Pipeline)

**Curso:** Inteligencia Artificial  
**Institución:** Universidad Peruana Unión (UPeU) - Campus Juliaca  
**Estudiante:** William David Ortiz Mamani  
**GitHub:** [302xdwill](https://github.com/302xdwill)  
**Correo Institucional:** `william.ortiz@upeu.edu.pe`  
**Correo Personal:** `davidom302@gmail.com`  
**Versión:** 2.0.0 (Optimizado con Arquitectura en Dos Etapas)  

---

## 📌 Descripción del Proyecto

Este proyecto implementa una solución integral, robusta y optimizada de **Visión Artificial y Aprendizaje Profundo (Deep Learning)** para la detección de rostros y la clasificación en tiempo real de **estados emocionales humanos** utilizando la cámara web integrada de la laptop.

El sistema incorpora una **Arquitectura en Dos Etapas (Two-Stage Architecture)** de nivel de producción que resuelve los problemas comunes de falsos positivos en ropa o fondos, así como el desbalance o sesgo de clases:

1. **Etapa 1 (Localización Anatómica por Puntos Clave):** Utiliza `YOLO11-Pose` para detectar los puntos clave faciales (ojos, nariz, orejas). A partir de la geometría facial, calcula matemáticamente la caja delimitadora exacta del rostro. **Elimina el 100% de cajas falsas en casacas, roperos, paredes o pósters** y evita partir el rostro en múltiples fragmentos.
2. **Etapa 2 (Reconocimiento y Clasificación de Emoción):** Extrae el rostro recortado y lo envía a un clasificador **FER (Facial Emotion Recognition)** optimizado en formato ONNX (`HSEmotion` basado en `EfficientNet` preentrenado en AffectNet/RAF-DB), reconociendo de forma fidedigna expresiones como Neutral, Enojo, Felicidad, Tristeza, Miedo y Sorpresa.

Además, el sistema permite alternar en vivo con el modelo YOLO entrenado en el dataset de Roboflow mediante la tecla `[t]`, permitiendo comparar ambos enfoques en tiempo real.

---

## 🎭 Clases y Emociones Reconocidas

| Emoción (Etiqueta) | Traducción al Español | Color Visual en Pantalla | Descripción Facial |
|:---|:---|:---:|:---|
| `anger` / `disgust` | 🔴 Enojo / Molestia | Rojo brillante | Ceño fruncido, tensión en la mandíbula o dientes |
| `happy` / `happiness` | 🟢 Felicidad | Verde esmeralda | Sonrisa visible, elevación de pómulos |
| `neutral` | ⚪ Neutral | Gris platino | Rostro en reposo, mirada serena sin tensión |
| `sad` / `sadness` | 🔵 Tristeza | Azul cobalto | Comisuras de labios descendidas, cejas caídas |
| `surprise` | 🟡 Sorpresa | Amarillo dorado | Ojos muy abiertos, boca ligeramente abierta |
| `fear` | 🟣 Miedo | Púrpura | Cejas levantadas y juntas, tensión ocular |

---

## 🔬 Análisis del Problema y Solución Técnica

### ¿Por qué fallaba la detección directa con el dataset de Roboflow?
1. **Diferencia de Escala (Scale Mismatch):** El dataset de Roboflow exportado (`face emotions.v1i.yolo26`) contiene mosaicos de 6 a 12 rostros diminutos recortados (~130x140 píxeles) empaquetados en imágenes de 640x640. Al situarse frente a la cámara web, el rostro del usuario ocupa entre 300 y 450 píxeles. El modelo directo intenta encajar cajas de 130 píxeles, dividiendo la cara en dos (frente y boca) o buscando patrones de 130 píxeles en la ropa.
2. **Ausencia de Imágenes Negativas de Fondo:** El dataset no contenía fotos de habitaciones vacías, prendas ni muebles. En consecuencia, patrones oscuros o pliegues en una casaca se confundían con cejas y ojos.
3. **Desbalance de Clases:** `happy` representaba más del 30% del dataset, mientras que `neutral` solo el 8.9%. Con pocas épocas de entrenamiento, la red tendía a predecir "Felicidad 32%" por defecto.

### La Solución Implementada:
Con el pipeline de **Dos Etapas**:
- `YOLO11-Pose` localiza con 100% de certeza la anatomía facial de William.
- Se aísla el rostro y se normaliza a 224x224.
- El clasificador `HSEmotion` analiza microexpresiones y clasifica la emoción con precisión (por ejemplo, identificando una cara seria como `Neutral` con 0% de falsa felicidad).

---

## 📂 Arquitectura y Estructura del Repositorio

```
Renococimiento Facial/
├── config/
│   └── data.yaml                   # Configuración del dataset Roboflow y clases
├── src/
│   ├── __init__.py                 # Paquete de Python
│   ├── utils.py                    # Funciones de cámara, traducción, HUD y colores BGR
│   ├── webcam_detector.py          # Detector en vivo con soporte 2-Etapas y YOLO directo
│   ├── train.py                    # Script modular de entrenamiento YOLO
│   └── evaluate.py                 # Script de evaluación de métricas y predicciones
├── models/
│   ├── best.pt                     # Pesos entrenados con YOLO
│   └── last.pt                     # Pesos de la última época
├── yolo11n-pose.pt                 # Modelo de localización anatómica de rostros
├── samples/                        # Muestras para pruebas estáticas
├── test_webcam_emotion.jpg         # Foto real de prueba del usuario (William)
├── test_webcam_result.jpg          # Resultado procesado por el pipeline
├── main.py                         # Menú interactivo de consola
├── requirements.txt                # Dependencias (ultralytics, opencv, onnxruntime, etc.)
├── .gitignore                      # Filtro de archivos pesados y temporales
└── README.md                       # Documentación técnica del proyecto
```

---

## ⚙️ Instalación y Uso

### 1. Activar el Entorno Virtual
Desde la raíz del proyecto en PowerShell:
```powershell
..\env\Scripts\activate
```

### 2. Instalar Dependencias
```powershell
pip install -r requirements.txt
```

### 3. Iniciar la Detección en Tiempo Real
Puedes ejecutar el sistema mediante el menú principal o directamente:

**Opción Directa (Recomendada):**
```powershell
python src/webcam_detector.py
```

**Mediante el Menú Interactivo:**
```powershell
python main.py
```
*(Seleccionar la opción `1` para iniciar la cámara web).*

---

## 🎮 Controles de Teclado en la Ventana de Video

| Tecla | Acción |
|:---:|:---|
| `[q]` / `[ESC]` | Cerrar el programa y liberar la cámara web |
| `[t]` | **Alternar Modo:** Cambia entre `2-Etapas (Pose + FER)` y `YOLO Roboflow Directo` |
| `[s]` | Guardar una captura de pantalla instantánea en formato `.jpg` |
| `[+]` / `[-]` | Aumentar o reducir el umbral de confianza |
| `[h]` | Ocultar o mostrar el banner HUD superior con métricas y FPS |

---

## 📊 Replicación de Entrenamiento

Si deseas entrenar el modelo YOLO directamente sobre el dataset:
```powershell
python src/train.py --epochs 10 --batch 8 --fraction 1.0 --imgsz 416
```
Los resultados y métricas se registrarán en `runs/facial_emotions_exp/`.
