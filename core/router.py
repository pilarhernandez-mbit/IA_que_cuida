# core/router.py

"""
Módulo de enrutamiento del asistente.
Decide si la entrada se resuelve con reglas, ML o se delega al LLM.
"""

from core.intent_classifier import IntentClassifier
from core.response_templates import TemplateEngine
import os
import sys
from pathlib import Path

BASE = Path(os.getcwd())
print("Ruta base:", BASE)
CORE_DIR   = BASE / 'core'
MODELS_DIR = BASE / 'models'
DATA_DIR   = BASE / 'scripts' / 'data'
LOG_CSV    = DATA_DIR / 'logs_interaccion.csv'

def route_text(text: str, use_ml: bool = True) -> dict:
    """
    Procesa una entrada de texto y devuelve un diccionario con:
      - intent: intención detectada
      - used: método usado (regla o ML)
      - score: confianza
      - response: texto generado
    """


    # Crear instancia
    intent_clf = IntentClassifier(
        model_path=MODELS_DIR / 'intent_clf.joblib',
        vectorizer_path=MODELS_DIR / 'vectorizer.joblib',
        label_enc_path=MODELS_DIR / 'label_encoder.joblib'
    )
    result = intent_clf.predict_intent(text, use_ml=use_ml)
    engine = TemplateEngine()
    response = engine.render(result["intent"], {"user_name": "Esther"})
    result["response"] = response
    return result


if __name__ == "__main__":
    # Test rápido
    out = route_text("pon una alarma mañana a las 8", use_ml=True)
    print(out)


# Reglas simples para activar LLM 
OPEN_ENDED = {'abierta', 'conversación abierta', 'pregunta abierta', 'social'}

def should_use_llm(texto: str, intent: str, score: float = 1.0, emotion: str = 'neutral') -> bool:
    # Si la intención es abierta/social y score alto → LLM; si es recordatorio/ayuda/rechazo → plantilla.
    if (intent or '').lower() in OPEN_ENDED:
        return True
    # Emoción muy negativa + baja confianza → LLM para redacción más natural
    if emotion in {'anger','fear','sadness'} and score < 0.6:
        return True
    return False

