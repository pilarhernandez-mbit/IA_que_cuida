import os
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from base64 import b64decode
from pydub import AudioSegment
import tempfile

_RELEASE = True
_LOAD_LOCAL = False

# Definir el componente si se está ejecutando en desarrollo local o en un entorno de producción
if _LOAD_LOCAL and not _RELEASE:
    _component_func = components.declare_component(
        "audiorecorder",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("audiorecorder", path=build_dir)


def audiorecorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    pause_prompt="",
    custom_style={},
    start_style={},
    pause_style={},
    stop_style={},
    show_visualizer=True,
    key=None,
) -> AudioSegment:
    """
    Función para gestionar la grabación de audio con el componente personalizado.
    Devuelve un AudioSegment a partir del audio grabado.
    """
    base64_audio = _component_func(
        startPrompt=start_prompt,
        stopPrompt=stop_prompt,
        pausePrompt=pause_prompt,
        customStyle=custom_style,
        startStyle=start_style,
        pauseStyle=pause_style,
        stopStyle=stop_style,
        showVisualizer=show_visualizer,
        key=key,
        default=b"",
    )
    
    # Inicializamos un objeto vacío de AudioSegment
    audio_segment = AudioSegment.empty()

    # Si hay datos de audio (base64), los procesamos
    if len(base64_audio) > 0:
        # Convertimos de base64 a un formato que pydub pueda manejar (en este caso, cualquier formato que pydub soporte)
        audio_segment = AudioSegment.from_file(BytesIO(b64decode(base64_audio)))
    
    return audio_segment


# En la parte de Streamlit donde vamos a usar la grabación
if not _RELEASE:
    # Mostrar una subsección en Streamlit
    st.subheader("Prueba de grabación de audio")

    # Agregar un indicador de carga mientras el audio se graba
    with st.spinner("Preparando la grabación..."):
        # Usamos el componente para grabar el audio
        audio = audiorecorder(
            "🎙 Graba",
            "⏹ Para",
            "🛑 Pausa",
            custom_style={"color": "blue", "backgroundColor": "lightgrey"},
            start_style={"color": "pink", "backgroundColor": "red"},
            pause_style={"color": "green"},
            stop_style={"backgroundColor": "purple"},
            key="audio_1",
        )

    # Ver si se grabó audio
    if len(audio) > 0:
        # Creamos un archivo temporal para guardar el audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        
        # Exportamos el audio a este archivo temporal en formato MP3 (más compatible con Safari)
        audio.export(temp_file.name, format="mp3")
        
        # Guardamos la ruta en el estado de sesión de Streamlit para poder usarla después
        st.session_state.audio_path = temp_file.name
        st.write(f"Audio guardado en: {st.session_state.audio_path}")

        # Reproducir el archivo de audio usando Streamlit
        st.audio(st.session_state.audio_path)

        # Opcional: Mostrar algunas propiedades del audio
        st.write(f"Frame rate: {audio.frame_rate}")
        st.write(f"Duración: {audio.duration_seconds:.2f} segundos")
