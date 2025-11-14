
import streamlit as st
from audiorecorder import audiorecorder
import wave
import os
import tempfile
from datetime import datetime
from pydub import AudioSegment
import pandas as pd
from pysentimiento import create_analyzer
import altair as alt 
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import pyttsx3
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import pyttsx3
import whisper
import torch
from datetime import datetime
from core.intent_classifier import IntentClassifier
from core.response_templates import get_template_for_intent
from core.router import should_use_llm
from openai import OpenAI
from google.cloud import texttospeech
from playsound import playsound
import sys
from pathlib import Path
import re
import html
import emoji
import html


# Integración Módulos 2-3


BASE = Path(__file__).resolve().parent #Path.cwd()
CORE_DIR   = BASE / 'core'
MODELS_DIR = BASE / 'models'
DATA_DIR   = BASE / 'scripts' / 'data'
LOG_CSV    = DATA_DIR / 'logs_interaccion.csv'
sys.path.append(str(CORE_DIR))


# Clasificador de intención (Módulo 2)
intent_clf = IntentClassifier(
    model_path      = MODELS_DIR / 'intent_clf.joblib',
    vectorizer_path = MODELS_DIR / 'vectorizer.joblib',
    label_enc_path  = MODELS_DIR / 'label_encoder.joblib'
)


if intent_clf.clf is None or intent_clf.vec is None or intent_clf.le is None:
    print("Model, vectorizer or label encoder not found, training from scratch...")
    resultado_train = intent_clf.train_intent_classifier_safe(train_csv_path=DATA_DIR / 'intents_train.csv')
    print("✅ Entrenamiento completado. Reporte:\n", resultado_train["report"])

    print("clf cargado:", intent_clf.clf is not None)
    print("vectorizador cargado:", intent_clf.vec is not None)
print("label encoder cargado:", intent_clf.le is not None)





os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./iaquecuida-1aeab7c2627c.json"
client = OpenAI(api_key="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")


#Mini App
st.set_page_config(page_title="IA que cuida 🧓👵💖🩺", layout="wide")
st.title("IA que cuida 🧓👵💖🩺")

# Datos usuario
nombre_usuario = st.text_input("Nombre del usuario") or "Paciente"
nivel_dependencia = st.selectbox(
    "Nivel de dependencia",
    ["Dependiente", "Semi-dependiente", "Autónomo"]
) or "Semi-dependiente"

st.text("🎤 Graba tu mensaje de voz")

# Iniciar variables
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "interacciones" not in st.session_state:
    st.session_state.interacciones = []

# ------------------------------
# CONFIGURAR TTS Y WHISPER
# ------------------------------

#os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"  # Ajusta según tu ruta

#  Cargar el modelo Whisper (solo una vez)
whisper_model = whisper.load_model("base")

# Configuración de voz (TTS)
engine = pyttsx3.init()
engine.setProperty('volume', 1.0)  # Ajusta el volumen
engine.setProperty('rate', 150)  # Velocidad de habla
voices = engine.getProperty('voices')

voz_seleccionada = False
for voice in voices:
    if 'spanish' in str(voice.languages).lower():
        engine.setProperty('voice', voice.id)
        voz_seleccionada = True
        break

if not voz_seleccionada:
    engine.setProperty('voice', voices[0].id)  # Usa la primera voz disponible



# ------------------------------
# FUNCIONES audio a voz
# ------------------------------


def transcribir_audio(audio_path):
    """Transcribe el audio a texto usando Whisper."""
    result = whisper_model.transcribe(audio_path, language="es")
    return result["text"].strip()

def audio_to_text(audio_path):
    """Convierte audio en texto usando Whisper (alternativa a Google Speech)."""
    return transcribir_audio(audio_path)
        

# ========================
# FUNCIONES Mod 2 y 4
# ========================

def hablar(
    texto,
    voz='female',       # 'female' o 'male'
    velocidad=150,      # velocidad del habla (por defecto 200)
    volumen=1.0,        # rango 0.0 a 1.0
    idioma=None,        # opcional: seleccionar voz por idioma
    mostrar_voces=False # para listar voces disponibles
):
    """
    Convierte texto en voz con parámetros personalizables.
    Compatible con pyttsx3 (funciona sin conexión).
    """

    engine = pyttsx3.init()

    # Mostrar todas las voces instaladas 
    if mostrar_voces:
        voices = engine.getProperty('voices')
        #for i, v in enumerate(voices):
           # print(f"[{i}] {v.name} ({v.languages}) - {v.id}")
        #return  # solo muestra voces y sale

    # Configurar voz (male/female o idioma)
    voices = engine.getProperty('voices')
    selected_voice = None

    if idioma:
        # Buscar una voz que contenga el idioma (ej: 'es' para español)
        for v in voices:
            if idioma.lower() in ''.join(v.languages).lower():
                selected_voice = v
                break

    if selected_voice is None:
        # Fallback a male/female
        if voz == 'male':
            selected_voice = voices[0]
        else:
            selected_voice = voices[1]

    engine.setProperty('voice', selected_voice.id)

    # Configurar velocidad y volumen
    engine.setProperty('rate', velocidad)
    engine.setProperty('volume', volumen)

    # Decir el texto
    engine.say(texto)
    engine.runAndWait()



def generar_audio(texto, nombre_archivo):
    # Crear cliente
    client_options = {"api_endpoint": "eu-texttospeech.googleapis.com:443"}  # ejemplo Europa
    client = texttospeech.TextToSpeechClient(client_options=client_options)

    # Texto a convertir
    texto = texto.strip()
    if texto.startswith("<speak>") and texto.endswith("</speak>"):
        # Es SSML
        synthesis_input = texttospeech.SynthesisInput(ssml=texto)
    else:
        # Es texto plano
        synthesis_input = texttospeech.SynthesisInput(text=texto)
        
    # Selección de voz
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-ES",  # español de España
         name="es-ES-Wavenet-F",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Configuración de audio
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # Generar audio
    response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
    )
   
     # Guardar archivo
    with open(nombre_archivo, "wb") as out:
        out.write(response.audio_content)
    
    print(f"Audio generado: {nombre_archivo}")
    # Reproducir automáticamente
    playsound(nombre_archivo)





def reescribir_respuesta(base_texto, nombre_usuario):
    prompt = base_texto
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    texto_llm = respuesta.choices[0].message.content
    print(f"Texto generado reescribir_respuesta {texto_llm}")
    return texto_llm



def texto_a_ssml(texto):
    # Detectar emojis en el texto
    emojis_en_texto = [e['emoji'] for e in emoji.emoji_list(texto)]

    # Reemplazar los emojis para SSML 
    for e in emojis_en_texto:
        texto = texto.replace(e, f"<say-as interpret-as='characters'>{e}</say-as>")

    # Envolver en <speak>
    ssml = f"<speak>{texto}</speak>"
    return ssml



def ssml_a_texto_natural(ssml):
    """
    Convierte un texto en SSML a lenguaje natural legible
    y elimina emojis.
    """
    # 1. Quitar etiquetas SSML
    texto = re.sub(r'<.*?>', '', ssml)
    
    # 2. Convertir entidades HTML (&amp;, &quot;, etc.)
    texto = html.unescape(texto)

    # 3. Quitar emojis (patrón unicode completo)
    emoji_pattern = re.compile(
        "["                   
        "\U0001F600-\U0001F64F"  # Emoticonos
        "\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # Transporte y mapas
        "\U0001F1E0-\U0001F1FF"  # Banderas
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    texto = emoji_pattern.sub("", texto)

    # 4. Limpiar espacios repetidos
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto


def generar_respuestaEnd(usuario, mensaje, llm_callable):
    print(f"Plantilla {llm_callable}")
      #  generar_audio(respuesta,"fichero_user.wav")
    text_convert = texto_a_ssml(mensaje)
    respuesta =  f"<speak>{text_convert}.</speak>"
    print(respuesta)
    generar_audio(respuesta,"fichero_user.wav")
    
    return respuesta


def append_log_csv(row: dict, csv_path=LOG_CSV):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    
    # Convertir fecha_hora a string estándar si es datetime
    if isinstance(row.get("fecha_hora"), datetime):
        row["fecha_hora"] = row["fecha_hora"].strftime("%Y-%m-%d %H:%M:%S")
    
    # Guardar CSV de forma consistente
    pd.DataFrame([row]).to_csv(
        csv_path,
        mode='a',
        header=header,
        index=False,
        encoding='utf-8',
        sep=',',          # fuerza coma como separador
        quoting=1         
    )



def generar_respuesta(nombre_paciente: str, texto_base: str, llm_callable=None) -> str:
    return texto_base if llm_callable is None else llm_callable(texto_base)




def procesar_texto_esther(
    texto_inicial_escrito: str,
    nombre_paciente: str,
    nivel_dependencia: str,
    llm_provider=None
) -> dict:
    """
    1) Detecta intención ('recordatorio', 'ayuda', 'rechazo', 'social', 'abierta')
    2) Obtiene score (confianza)
    3) Aplica reglas de decisión (plantilla vs LLM)
    4) Genera texto_base y texto_natural (con la firma acordada)
    """
    # --- Comprensión (Módulo 2) ---
    if hasattr(intent_clf, "predict_intent"):
        resultado = intent_clf.predict_intent(texto_inicial_escrito)
        intent = resultado['intent']
        score = resultado['score']

    else:
        intent = intent_clf.predict_intent(texto_inicial_escrito)
        score  = getattr(intent_clf, "predict_proba_max", lambda x: 1.0)(texto_inicial_escrito)

    emotion  = getattr(intent_clf, 'predict_emotion',  lambda x: 'neutral')(texto_inicial_escrito)
    entities = getattr(intent_clf, 'extract_entities', lambda x: {})(texto_inicial_escrito)

    # --- Decisión (Módulo 3) ---
    usar_llm = should_use_llm(texto_inicial_escrito, intent=intent, score=score, emotion=emotion)
    decision = 'llm' if usar_llm else 'plantilla'

    # --- texto_base + texto_natural ---
    if decision == 'plantilla':
        texto_base = get_template_for_intent(
            intent=intent,
            perfil_usuario=nivel_dependencia,   # 'Autónomo' | 'Semi-dependiente' | 'Dependiente'
            nombre=nombre_paciente,
            entities=entities,
            emotion=emotion
        )
        texto_natural = generar_respuesta(nombre_paciente, texto_base, llm_callable=None)
        fuente, llm_usado = 'plantilla', ''
    else:
        # Prompt base para LLM (breve y seguro)
        texto_base = (
            f"Paciente: {nombre_paciente}\n"
            f"Dependencia: {nivel_dependencia}\n"
            f"Intención: {intent} (score={score:.2f})\n"
            f"Emoción: {emotion}\n"
            f"Entidades: {entities}\n"
            f"Entrada: {texto_inicial_escrito}\n\n"
            f"Responde en español, 1–2 frases, claro, amable y seguro para persona mayor."
        )
        textoInter = reescribir_respuesta(texto_base, nombre_usuario)
        texto_natural = generar_respuesta(nombre_paciente, textoInter, llm_callable=None)
        fuente = 'llm' if llm_provider else 'plantilla'
        llm_usado = getattr(llm_provider, '__name__', '') if llm_provider else ''

    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'nombre_paciente': nombre_paciente,
        'perfil_usuario': nivel_dependencia,
        'texto_inicial_escrito': texto_inicial_escrito,
        'intent': intent,
        'score': round(float(score), 3),
        'emotion': emotion,
        'entities': entities,
        'decision': decision,
        'texto_base': texto_base,
        'texto_natural': texto_natural,
        'fuente': fuente,
        'llm': llm_usado
    }


# Analizador emociones 
analyzer = create_analyzer(task="emotion", lang="es")


# Grabar el audio
audio = audiorecorder("🎙 Grabar", "⏹ Parar")

# Verifica si el audio tiene contenido y exporta
if len(audio) > 0 and st.session_state.audio_path is None:
    # Usar tempfile para crear un archivo temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_file.name, format="wav")  # Exportar a un archivo WAV
    st.session_state.audio_path = temp_file.name  # Guardar la ruta en la sesión


# Reproductor de autido
if st.session_state.audio_path:
    st.audio(st.session_state.audio_path)

    # Botón enviar
    if st.button("📤 Enviar audio"):
        # Analizar emoción 

        texto_transcrito = transcribir_audio(st.session_state.audio_path)
        if texto_transcrito:
            st.write("Texto transcrito del audio:", texto_transcrito)
            texto_inicial_escrito = texto_transcrito
        audio_text = f"Audio de {nombre_usuario}"  # Whisper

        resultado_emocion = analyzer.predict(texto_inicial_escrito)
        emocion = resultado_emocion.output
        score_emocion = round(resultado_emocion.probas[resultado_emocion.output], 2)


        # Guardar interacción en la sesión
        nueva_fila = {
            "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "usuario": nombre_usuario,
            "nivel_dependencia": nivel_dependencia,
            "confianza": score_emocion,
            "transcripcion": texto_inicial_escrito
        }


        print(f"TExtoINicial{texto_inicial_escrito}")
        # === 🔹 LLAMADA A LOS MÓDULOS 2–3 ===
        resultado_test = procesar_texto_esther(
            texto_inicial_escrito=texto_inicial_escrito,
            nombre_paciente=nombre_usuario,
            nivel_dependencia=nivel_dependencia,
            llm_provider=None   # o tu callable si quieres probar LLM
        )
        generar_respuestaEnd(nombre_usuario, resultado_test['texto_natural'], resultado_test['llm'])
        intent = resultado_test['intent']
        score = resultado_test['score']
        emotion = resultado_test['emotion']
        entities = resultado_test['entities']
        decision = resultado_test['decision']
        fuente = resultado_test['fuente']
        texto_base = resultado_test['texto_base']
        texto_natural = resultado_test['texto_natural']
        llm_usado = resultado_test['llm']
        print(f"🔎 Intent: {resultado_test['intent']} | score={resultado_test['score']:.2f} | decisión: {resultado_test['decision']} | fuente: {resultado_test['fuente']} || emotion: {emotion}")
        print("\n💬 texto_base:\n", resultado_test['texto_base'])
        print("\n🗨️ texto_natural:\n", resultado_test['texto_natural'])
        print("\n🗨️ texto iniciar:\n", texto_inicial_escrito)



        respuesta = client.chat.completions.create(
                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "user", "content": f"Traduce este texto al inglés y dame solo la traducción, sin explicaciones:\n{emocion}"}
        ]
                        )           
        emotion = respuesta.choices[0].message.content
        print(f"Respuestaaaaa: {respuesta.choices[0].message.content}")

         # Guardar interacción extendida
        nueva_fila.update({
            "intent": intent,
            "emotion": emotion,
            "score_intent": round(score, 3),
            "decision": decision,
            "texto_base": texto_base,
            "texto_natural": texto_natural
        })

        # Mostrar resultado en la interfaz
        st.success(f"✅ Audio procesado correctamente. Emoción: {emotion} | Intent: {intent} ({score:.2f}) | Decisión: {decision}")

       

        st.session_state.interacciones.append(nueva_fila)


        append_log_csv({
            "fecha_hora": nueva_fila["fecha_hora"],
            "usuario": nombre_usuario,
            "perfil_usuario": nivel_dependencia,
            "texto_inicial_escrito": texto_inicial_escrito,
            "intent": intent,
            "score": round(score,3),
            "emotion": emotion,
            "decision": decision,
            "texto_base": texto_base,
            "texto_natural": texto_natural,
            "fuente": fuente,
            "llm": llm_usado,
            "nivel_dependencia": nivel_dependencia
        })



        # Borrar archivo y limpiar estado
        if os.path.exists(st.session_state.audio_path):
            os.remove(st.session_state.audio_path)
        st.session_state.audio_path = None

    # Botón eliminar
    if st.button("🗑 Eliminar audio"):
        if os.path.exists(st.session_state.audio_path):
            os.remove(st.session_state.audio_path)
        st.session_state.audio_path = None


# Helpers
def _safe(val, default):
    return default if val in (None, "", "None") else val


# Dashboard de indicadores
st.header("📊 Dashboard de indicadores")

if st.session_state.interacciones:
    df = pd.read_csv(LOG_CSV, encoding='utf-8', sep=',', quotechar='"')

    # Métricas rápidas
    st.metric("Interacciones totales", len(df))
    st.metric("Última emoción detectada", df.iloc[-1]["emotion"])
    st.metric("Nivel de dependencia actual", df.iloc[-1]["nivel_dependencia"])

    #  Construcción de tabla completa
    st.subheader("📋 Interacciones registradas")

    # Convertir a datetime, ignorando errores
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

    # Verificar si hay conversiones fallidas
    if df['fecha_hora'].isna().any():
        st.warning("Algunas filas tienen fecha_hora inválida y se convertirán en NaT")

    # Extraer fecha y hora
    df['fecha'] = df['fecha_hora'].dt.date       # solo fecha
    df['hora']   = df['fecha_hora'].dt.hour      # solo hora

    
    st.dataframe(df)

   # 📊 Histograma de emociones
    st.subheader("📊 Distribución de emociones")
    col1, col2, col3 = st.columns([0.5, 5, 0.5])
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            data=df,
            x="emotion",
            order=df["emotion"].value_counts().index,
            palette="viridis",
            ax=ax
        )
        ax.set_xlabel("Emoción")
        ax.set_ylabel("Cantidad")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

   
    # ⏰ Interacciones por hora (heatmap)
    st.subheader("⏰ 🗣️ Interacciones por hora")
    col1, col2, col3 = st.columns([0.5, 5, 0.5])
    with col2:
        pivot = df.groupby(['hora', 'emotion']).size().unstack(fill_value=0)
        # Crear el gráfico
        fig2, ax2 = plt.subplots(figsize=(8,6))  # Ajustamos el tamaño del gráfico para que sea más legible
        sns.heatmap(pivot, cmap="Spectral", annot=True, fmt="d", ax=ax2, cbar_kws={'label': 'Número de interacciones'}, annot_kws={"size": 12}, linewidths=1, linecolor='white')
        
        # Ajustes estéticos
        ax2.set_xlabel("Emoción", fontsize=14, labelpad=10)
        ax2.set_ylabel("Hora del día", fontsize=14, labelpad=10)
        ax2.set_title("Interacciones por hora y emoción", fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()  
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig2, use_container_width=False)


    # 📈 Evolución emocional por día (area plot)
    st.subheader("📈 🧠 Evolución emocional por día")
    col1, col2, col3 = st.columns([0.5, 5, 0.5])
    with col2:
        evolucion = df.groupby(["fecha","emotion"]).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(8,4))
        evolucion.plot(kind="area", stacked=True, ax=ax3, alpha=0.6)
        ax3.set_ylabel("Número de interacciones")
        ax3.set_xlabel("Fecha")
        ax3.set_title("Evolución emocional diaria")
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=False)

    # 📈 Evolución intenciones por día (line plot)
    st.subheader("📈 📝👣 Evolución intenciones por día")
    col1, col2, col3 = st.columns([0.5, 5, 0.5])
    with col2:
        intenciones = df.groupby(["fecha",'intent']).size().unstack(fill_value=0)
        fig4, ax4 = plt.subplots(figsize=(8,4))
        intenciones.plot(kind="line", marker="o", ax=ax4)
        ax4.set_ylabel("Número de interacciones")
        ax4.set_xlabel("Fecha")
        ax4.set_title("Evolución de intenciones diarias")
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=False)



else:
    st.info("No hay interacciones registradas aún. Graba un audio para empezar.")

