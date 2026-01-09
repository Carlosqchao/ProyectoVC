# MazeBall

Descripción
- MazeBall es un juego de puzzles y plataformas desarrollado en Godot que incorpora control mediante reconocimiento de manos usando MediaPipe. El objetivo es resolver laberintos y desafíos de plataformas moviendo una bola y accionando mecanismos mediante entradas tradicionales (teclado/ratón/joystick) y gestos de la mano capturados por la webcam.

Características principales
- Niveles tipo puzzle y plataformas con física de bola.
- Integración de seguimiento de manos (MediaPipe) para interacción alternativa basada en gestos.
- Código de juego en GDScript (Godot) y scripts en Python para el procesamiento de la webcam y MediaPipe.
- Notebooks (Jupyter) incluidos para experimentación o visualización de datos (si aplica).

Requisitos
- Godot (revisa project.godot para la versión recomendada; puede ser Godot 3.x o 4.x según la configuración del proyecto).
- Python 3.8+ para la integración con MediaPipe.
- Dependencias Python: mediapipe, opencv-python, numpy (o las listadas en requirements.txt si existe).
- Webcam con permisos de acceso.
- Sistema operativo: Windows, macOS o Linux.

Instalación rápida
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

Cómo ejecutar
1. Inicia el proceso de seguimiento de manos en Python. Ejecuta el script encargado del procesamiento de la webcam y envío de datos de mano (por ejemplo hand_tracking.py o el script presente en la carpeta scripts/). Ejemplo:
   python scripts/hand_tracking.py

   - Este proceso captura la webcam, ejecuta MediaPipe para detectar manos y expone las coordenadas/gestos para que Godot los consuma.
   - Revisa los scripts en /scripts o /scripts_py para detalles sobre el método de comunicación (sockets, UDP, WebSocket o stdin/stdout) y los parámetros de calibración.

2. Ejecuta el juego en Godot desde el editor o usando la línea de comandos. Asegúrate de que el script Python de seguimiento esté activo antes de iniciar el juego para que la integración funcione correctamente.

Controles y gestos (general)
- Controles tradicionales: teclado y ratón (configuración en Input Map del proyecto).
- Gestos con la mano: el proyecto utiliza posiciones y gestos detectados por MediaPipe para:
  - Mover o empujar la bola (posición o gesto de señalamiento).
  - Agarrar/soltar mediante gesto de pinza o cierre de mano.
  - Activar interruptores mediante gesto de empuje o toque.
- Consulta la implementación de entrada en los scripts de integración para ver el mapeo exacto entre las coordenadas/gestos y las acciones del juego.

Estructura sugerida del repositorio
- project.godot                — archivo del proyecto Godot
- /scenes                     — escenas de Godot (.tscn/.scn)
- /scripts_gd                 — scripts GDScript del juego
- /scripts_py                 — scripts Python para MediaPipe y comunicación
- /notebooks                  — Jupyter notebooks para experimentación (opcional)
- /assets                     — sprites, sonidos, modelos
- README.md

Consejos de desarrollo y depuración
- Latencia y rendimiento: el procesamiento de MediaPipe y la comunicación con Godot pueden introducir latencia. Ajusta la resolución de la cámara, la frecuencia de envío de datos y el pipeline de MediaPipe para optimizar el rendimiento.
- Calibración: añade opciones para calibrar la posición/escala de las manos respecto a la ventana del juego.
- Logs: habilita logs tanto en el script Python como en Godot para verificar la recepción de mensajes y las coordenadas/gestos transmitidos.
- Permisos: asegura que la webcam tenga permisos concedidos al ejecutar los scripts.

Contribuir
- Para contribuir abre un issue o crea un pull request con la mejora o corrección.
- Añade descripciones claras, pasos para reproducir y pruebas si aplican.
- Si propones cambios en la integración MediaPipe, documenta los parámetros y dependencias.

Licencia
- Añade un archivo LICENSE con la licencia que prefieras (por ejemplo MIT). Si no hay licencia en el repositorio, añádela antes de redistribuir.

Créditos y contacto
- Autor/Repositorio: Carlosqchao / ProyectoVC
- Para dudas, issues o propuestas de mejora, abre un issue en el repositorio.

Nota final
- Revisa los scripts incluidos en las carpetas del repositorio (especialmente los scripts Python encargados de MediaPipe y los scripts GDScript de entrada) para adaptar las instrucciones de ejecución y los nombres de archivos a la estructura real del proyecto.
