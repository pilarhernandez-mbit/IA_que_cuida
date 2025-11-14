# 🧠 IA que Cuida  
_Asistente cognitivo domiciliario basado en Inteligencia Artificial_

## 📌 Introducción
**IA que Cuida** es un asistente cognitivo domiciliario diseñado para acompañar, asistir y monitorizar de forma no invasiva a personas mayores en su vida cotidiana. El sistema combina:

- Comprensión del **lenguaje natural**  
- Análisis de **emociones**  
- Monitorización de indicadores cognitivos, funcionales, sociales, nutricionales y clínicos  
- **Toma de decisiones personalizada**  
- Interacción accesible y empática

El objetivo es mejorar la calidad de vida, apoyar la continuidad asistencial y reducir el impacto de la soledad no deseada mediante una solución tecnológica avanzada pero cercana y de bajo coste operativo.

La hipótesis de trabajo es que un sistema capaz de interpretar lenguaje natural, detectar estados emocionales y adaptarse al perfil individual del usuario puede actuar como un puente eficaz entre la persona, su familia y los servicios sociosanitarios.

---

## 📁 Estructura del proyecto

```
IA_que_Cuida/
│
├── ia_que_cuida.py            # Script principal del asistente
├── packages.txt               # Dependencias del proyecto
├── setup.py                   # Configuración para empaquetado e instalación
│
├── audiorecorder/             # Librería integrada
│   ├── __init__.py
│   └── frontend/              # Archivos web compilados
│
├── core/                      # Código principal
│   ├── __init__.py
│   ├── intent_classifier.py
│   ├── response_templates.py
│   └── router.py
│
├── scripts/                   # Scripts de entrenamiento y datos
│   ├── data/
│   │   ├── fase3_resultados.csv
│   │   ├── fase4_validacion_intent.csv
│   │   ├── intencion_emocion_resultados.csv
│   │   ├── intents_train_expanded.csv
│   │   ├── intents_train.csv
│   │   └── logs_interaccion.csv
│   ├── train_intents.py
│   ├── train_intents.py.txt
│   └── validate_model.py.txt
│
├── models/                    # Modelos entrenados
│   ├── intent_clf.joblib
│   ├── label_encoder.joblib
│   ├── vectorizer.joblib
│   ├── tfm_env/
│   └── fichero_user.wav
│
└── README.md                  # Este archivo
```

---

## 🚀 Instalación y ejecución

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/usuario/IA_que_Cuida.git
cd IA_que_Cuida
```

### 2️⃣ Instalar dependencias

```bash
pip install -r packages.txt
```

### 3️⃣ Instalar el paquete localmente

```bash
pip install .
```

### 4️⃣ Ejecutar el asistente

```bash
python ia_que_cuida.py
```

---

## 📦 Generar distribución del proyecto

```bash
python setup.py sdist
```

El paquete generado aparecerá en:

```
dist/
```

---

## 👥 Autores
- Esther Lueje Alonso  
- Pilar Hernández Lozano  
- Alfredo Cueva Escudero  

**Máster en Inteligencia Artificial Avanzada y Generativa – MBIT School**  
**Fecha:** 14/11/2025

