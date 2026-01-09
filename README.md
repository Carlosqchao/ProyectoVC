# MazeBall
<img width="1024" height="877" alt="Caratula" src="https://github.com/user-attachments/assets/8ca9c06b-88a4-41ef-8a44-5b7f79fca7f4" />

### Descripción
- MazeBall es un juego de puzzles y plataformas desarrollado en Godot que incorpora control mediante reconocimiento de manos usando MediaPipe. El objetivo es resolver laberintos y desafíos de plataformas moviendo una bola y accionando mecanismos mediante entradas tradicionales (teclado/ratón) y gestos de la mano capturados por la webcam.

### Características principales
- Niveles tipo puzzle y plataformas con física de bola.
- Integración de seguimiento de manos (MediaPipe) para interacción alternativa basada en gestos.
- Código de juego en GDScript (Godot) y scripts en Python para el procesamiento de la webcam y MediaPipe.
- El sistema **detecta** varias poses de la mano en tiempo real usando MediaPipe y OpenCV: índice extendido (pointer), gesto de “rock” con índice y meñique levantados, gesto de “peace” o “victoria” con índice y medio, y una forma de “C” en la que índice y medio están curvados mientras anular y meñique permanecen recogidos. Para cada mano (izquierda o derecha), se analiza qué dedos están extendidos o doblados y se clasifica la mano en una de estas categorías, asociando también si la mano está invertida o no.

- Además de la clasificación de la pose, se dibujan cajas rotadas alrededor de los dedos relevantes y, para el gesto “rock”, se traza una línea entre la base del índice y la del meñique para remarcar la silueta. El sistema calcula y suaviza la posición, tamaño y ángulo de cada mano, generando datos estables (coordenadas, longitud, orientación y tipo de gesto) que ya se están utilizando como entrada gestual en Godot dentro de la aplicación interactiva.

## Gif gestos Mediapipe

![Gif1](https://github.com/user-attachments/assets/923602bf-c09e-4e94-9c2b-90c8948cf597)


### Requisitos
- Godot (revisa project.godot para la versión recomendada; puede ser Godot 3.x o 4.x según la configuración del proyecto).
- Python 3.8+ para la integración con MediaPipe.
- Dependencias Python: mediapipe, opencv-python, numpy.
- Webcam con permisos de acceso.
- Sistema operativo: Windows, macOS o Linux.

### Instalación rápida
1. Clona el repositorio:
   git clone https://github.com/Carlosqchao/ProyectoVC.git
   cd ProyectoVC

2. Preparar el entorno Python (opcional pero recomendado):
   python -m venv venv
   - En macOS/Linux: source venv/bin/activate
   - En Windows (PowerShell): .\venv\Scripts\Activate.ps1

3. Instalar dependencias Python:
   - Si existe requirements.txt:
     pip install -r requirements.txt
   - Si no existe, instala lo mínimo necesario:
     pip install mediapipe opencv-python numpy

4. Abrir el proyecto en Godot:
   - Abre Godot y selecciona la carpeta del proyecto (contiene project.godot) o
   - Desde línea de comandos (según versión): godot --path .

### Cómo ejecutar
1. Inicia el proceso de seguimiento de manos en Python. Ejecuta el script encargado del procesamiento de la webcam y envío de datos de mano (por ejemplo hand_tracking.py o el script presente en la carpeta scripts/). Ejemplo:
   python scripts/hand_tracking.py

   - Este proceso captura la webcam, ejecuta MediaPipe para detectar manos y expone las coordenadas/gestos para que Godot los consuma.

2. Ejecuta el juego en Godot desde el editor o usando la línea de comandos. Asegúrate de que el script Python de seguimiento esté activo antes de iniciar el juego para que la integración funcione correctamente.

### Controles y gestos (general)
- Controles tradicionales: teclado y ratón (configuración en Input Map del proyecto).
- Gestos con la mano: el proyecto utiliza posiciones y gestos detectados por MediaPipe para:
  - Mover o empujar la bola (posición o gesto de señalamiento).
  - Atraer o alejar la bola mediante gestos especiales
  - Activar interruptores mediante gesto de empuje o toque.

### Estructura sugerida del repositorio
- project.godot                — archivo del proyecto Godot
- /scenes                     — escenas de Godot (.tscn/.scn)
- /scripts_gd                 — scripts GDScript del juego
- /scripts_py                 — scripts Python para MediaPipe y comunicación
- /notebooks                  — Jupyter notebooks para experimentación (opcional)
- /assets                     — sprites, sonidos, modelos
- README.md

## Video Demo
[![Video MazeBall](https://img.youtube.com/vi/VrBp10FoIUI/maxresdefault.jpg)](https://youtu.be/VrBp10FoIUI)

### Créditos y contacto
- Autor/Repositorio: Carlosqchao y Juanelboi / ProyectoVC
- Para dudas, issues o propuestas de mejora, abre un issue en el repositorio.
