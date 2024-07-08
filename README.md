# Introducción a GPT-4o

GPT-4o ("o" de "omni") está diseñado para manejar una combinación de entradas de texto, audio y video, y puede generar salidas en formatos de texto, audio e imagen.

## Antecedentes

Antes de GPT-4o, los usuarios podían interactuar con ChatGPT utilizando el Modo de Voz, que operaba con tres modelos separados. GPT-4o integrará estas capacidades en un solo modelo entrenado en texto, visión y audio. Este enfoque unificado asegura que todas las entradas, ya sean textuales, visuales o auditivas, sean procesadas de manera cohesiva por la misma red neuronal.

## Capacidades Actuales de la API

Actualmente, la API solo admite entradas de texto e imagen, con salidas de texto, las mismas modalidades que `gpt-4-turbo`. Se introducirán pronto modalidades adicionales, incluido el audio. Esta guía te ayudará a comenzar a usar GPT-4o para la comprensión de texto, imagen y video.

---

## Empezando

### Instalar el SDK de OpenAI para Python

```bash
%pip install --upgrade openai --quiet
```

### Configurar el cliente de OpenAI y enviar una solicitud de prueba

Para configurar el cliente para nuestro uso, necesitamos crear una clave API para usar con nuestra solicitud. Omite estos pasos si ya tienes una clave API para uso.

Puedes obtener una clave API siguiendo estos pasos:

1. [Crear un nuevo proyecto](https://help.openai.com/en/articles/9186755-managing-your-work-in-the-api-platform-with-projects)
2. [Generar una clave API en tu proyecto](https://platform.openai.com/api-keys)
3. (RECOMENDADO, PERO NO REQUERIDO) [Configurar tu clave API para todos los proyectos como una variable de entorno](https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key)

Una vez que tengamos esto configurado, comencemos con una simple entrada de {texto} al modelo para nuestra primera solicitud. Usaremos tanto mensajes `system` como `user` para nuestra primera solicitud, y recibiremos una respuesta del rol `assistant`.

```python
from openai import OpenAI
import os

# Configurar la clave API y el nombre del modelo
MODEL = "gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<tu clave API de OpenAI si no está configurada como variable de entorno>"))

completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "Eres un asistente útil. ¡Ayúdame con mi tarea de matemáticas!"},
    {"role": "user", "content": "¡Hola! ¿Podrías resolver 2+2?"}
  ]
)

print("Assistant: " + completion.choices[0].message.content)
```

### Procesamiento de Imágenes

GPT-4o puede procesar imágenes directamente y tomar acciones inteligentes basadas en la imagen. Podemos proporcionar imágenes en dos formatos:

1. Codificado en Base64
2. URL

Primero, veamos la imagen que usaremos y luego intentemos enviar esta imagen tanto como Base64 como un enlace URL a la API.

#### Procesamiento de Imágenes en Base64

```python
from IPython.display import Image, display, Audio, Markdown
import base64

IMAGE_PATH = "data/triangle.png"

# Vista previa de la imagen para contexto
display(Image(IMAGE_PATH))

# Abrir el archivo de imagen y codificarlo como una cadena base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Eres un asistente útil que responde en Markdown. ¡Ayúdame con mi tarea de matemáticas!"},
        {"role": "user", "content": [
            {"type": "text", "text": "¿Cuál es el área del triángulo?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

#### Procesamiento de Imágenes por URL

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Eres un asistente útil que responde en Markdown. ¡Ayúdame con mi tarea de matemáticas!"},
        {"role": "user", "content": [
            {"type": "text", "text": "¿Cuál es el área del triángulo?"},
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"}}
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

### Procesamiento de Video

Aunque no es posible enviar un video directamente a la API, GPT-4o puede entender videos si muestrean fotogramas y luego los proporcionan como imágenes. Realiza mejor esta tarea que GPT-4 Turbo.

Dado que GPT-4o en la API aún no admite entrada de audio (a partir de mayo de 2024), usaremos una combinación de GPT-4o y Whisper para procesar tanto el audio como lo visual de un video proporcionado, y mostrar dos casos de uso:

1. Resumen
2. Preguntas y Respuestas

#### Configuración para el Procesamiento de Video

Usaremos dos paquetes de Python para el procesamiento de video - opencv-python y moviepy.

Estos requieren [ffmpeg](https://ffmpeg.org/about.html), así que asegúrate de instalar esto de antemano. Dependiendo de tu sistema operativo, es posible que necesites ejecutar `brew install ffmpeg` o `sudo apt install ffmpeg`.

```bash
%pip install opencv-python --quiet
%pip install moviepy --quiet
```

#### Procesar el Video en Dos Componentes: Fotogramas y Audio

```python
import cv2
from moviepy.editor import VideoFileClip
import time
import base64

# Usaremos el video de resumen del OpenAI DevDay. Puedes revisar el video aquí: https://www.youtube.com/watch?v=h02ti0Bl6zk
VIDEO_PATH = "data/keynote_recap.mp4"

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    # Bucle a través del video y extraer fotogramas a la tasa de muestreo especificada
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extraer audio del video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

# Extraer 1 fotograma por segundo. Puedes ajustar el parámetro `seconds_per_frame` para cambiar la tasa de muestreo
base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)
```

```python
## Mostrar los fotogramas y el audio para contexto
display_handle = display(None, display_id=True)
for img in base64Frames:
    display_handle.update(Image(data=base64.b64decode(img.encode("utf-8")), width=600))
    time.sleep(0.025)

Audio(audio_path)
```

### Ejemplo 1: Resumen

Ahora que tenemos tanto los fotogramas del video como el audio, realicemos algunas pruebas diferentes para generar un resumen del video y comparar los resultados de usar los modelos con diferentes modalidades. Deberíamos esperar ver que el resumen generado con contexto de entradas visuales y auditivas será el más preciso, ya que el modelo puede usar todo el contexto del video.

1. Resumen Visual
2. Resumen de Audio
3. Resumen Visual + Audio

#### Resumen Visual

El resumen visual se genera enviando al modelo solo los fotogramas del video. Con solo los fotogramas, es probable que el modelo capture los aspectos visuales, pero perderá cualquier detalle discutido por el orador.

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Estás generando un resumen de video. Proporciona un resumen del video. Responde en Markdown."},
        {"role": "user", "content": [
            "Estos son los fotogramas del video.",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
        ]}
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

#### Resumen de Audio

El resumen de audio se genera enviando al modelo la transcripción de audio. Con solo el audio, es probable que el modelo se incline hacia el contenido del audio y perderá el contexto proporcionado por las presentaciones y los visuales.

La entrada de `{audio}` para GPT-4o no está disponible actualmente, pero llegará pronto. Por ahora, usamos nuestro modelo existente `whisper-1` para procesar el audio.

```python
# Transcribir el audio
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(audio_path, "rb"),
)

## OPCIONAL: Descomenta la línea a continuación para imprimir la transcripción
#print("Transcript: ", transcription.text + "\n\n")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": """Estás generando un resumen de transcripción. Crea un resumen de la transcripción proporcionada. Responde en Markdown."""},
        {"role": "user", "content": [
            {"type": "text", "text": f"La transcripción de audio es: {transcription.text}"}
        ]}
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

#### Resumen Visual + Audio

El resumen Visual + Audio se genera enviando al modelo tanto lo visual como el audio del video a la vez. Al enviar ambos, se espera que el modelo resuma mejor ya que puede percibir todo el video a la vez.

```python
## Generar un resumen con visual y audio
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": """Estás generando un resumen de video. Crea un resumen del video y su transcripción proporcionados. Responde en Markdown."""},
        {"role": "user", "content": [
            "Estos son los fotogramas del video.",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
            {"type": "text", "text": f"La transcripción de audio es: {transcription.text}"}
        ]}
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

### Ejemplo 2: Preguntas y Respuestas

Para la P&R, usaremos el mismo concepto que antes para hacer preguntas sobre nuestro video procesado mientras realizamos las mismas 3 pruebas para demostrar el beneficio de combinar modalidades de entrada:

1. P&R Visual
2. P&R de Audio
3. P&R Visual + Audio

```python
QUESTION = "Pregunta: ¿Por qué Sam Altman puso un ejemplo sobre subir las ventanas y encender la radio?"
```

```python
qa_visual_response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Usa el video para responder la pregunta proporcionada. Responde en Markdown."},
        {"role": "user", "content": [
            "Estos son los fotogramas del video.",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
            QUESTION
        ]}
    ],
    temperature=0,
)

print("Visual QA:\n" + qa_visual_response.choices[0].message.content)
```

```python
qa_audio_response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": """Usa la transcripción para responder la pregunta proporcionada. Responde en Markdown."""},
        {"role": "user", "content": f"La transcripción de audio es: {transcription.text}. \n\n {QUESTION}"},
    ],
    temperature=0,
)

print("Audio QA:\n" + qa_audio_response.choices[0].message.content)
```

```python
qa_both_response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": """Usa el video y la transcripción para responder la pregunta proporcionada."""},
        {"role": "user", "content": [
            "Estos son los fotogramas del video.",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
            {"type": "text", "text": f"La transcripción de audio es: {transcription.text}"},
            QUESTION
        ]}
    ],
    temperature=0,
)

print("Both QA:\n" + qa_both_response.choices[0].message.content)
```

## Conclusión

Integrar muchas modalidades de entrada como audio, visual y textual, mejora significativamente el rendimiento del modelo en una amplia gama de tareas. Este enfoque multimodal permite una comprensión e interacción más completas, reflejando más de cerca cómo los humanos perciben y procesan la información.

Actualmente, GPT-4o en la API admite entradas de texto e imagen, con capacidades de audio próximamente.
