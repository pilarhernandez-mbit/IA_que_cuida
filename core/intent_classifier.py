# intent_classifier.py
from __future__ import annotations
import re, unicodedata
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class IntentClassifier:
    """
    Clase que unifica reglas + ML para clasificación de intents.
    """

    # Constantes de clase
    PRIMARY_INTENTS = ["recordatorio", "alerta", "rechazo", "social", "abierta"]

    PATTERNS: Dict[str, Dict[str, List[str]]] = {
        "recordatorio": {
            "must_any": [
                r"\brecuerdame\b",
                r"\bpon(?:er)? (?:un )?recordatorio\b",
                r"\balarma\b",
                r"\bav[ií]same\b",
                r"\bagenda\b",
                r"\bno me (?:olvide|olvide[s])\b",
                r"\bap[úu]ntame\b",
            ]
        },
        "alerta": {
            "must_any": [
                r"\bnecesito ayuda\b",
                r"\bllama (?:a|al)\b",
                r"\bemergencia\b|\burgencias\b|\b112\b",
                r"\bno puedo (?:levantarme|respirar)\b",
                r"\bme duele\b|\bme sangra\b|\bme ahogo\b",
                r"\bca[ií]da\b|\bmare[oa]\b",
            ]
        },
        "rechazo": {
            "must_any": [
                r"\bno\b.*\bquiero\b",
                r"\bno ahora\b",
                r"\bno necesito\b",
                r"\bno me apetece\b",
                r"\bdejame\b|\bdeja me\b",
                r"\bno molestes\b",
                r"\bprefiero que no\b",
            ]
        },
        "social": {
            "must_any": [
                r"^(hola|buen(?:os|as) d[ií]as|buenas|gracias|ad[ií]os|hasta luego)[\s!,.]*$",
                r"\bh[aá]blame\b|\bcharlar?\b|\bconversar?\b",
                r"\bcu[eé]ntame algo\b",
                r"\bme siento (?:solo|sola|triste)\b|\bestoy (?:solo|sola|triste)\b",
                r"\bpon (?:m[uú]sica|radio)\b",
            ]
        },
        "abierta": {
            "must_any": [
                r"[¿\?]",
                r"^(qu[eé]\s+|c[oó]mo\s+|cu[aá]ndo\s+|d[oó]nde\s+|por\s+qu[eé]\s+)",
                r"\breceta\b|\bcocinar\b|\bcocido\b|\btiempo\b|\bnoticias\b|\bpol[ií]tica\b",
            ]
        },
    }

    def __init__(
        self,
        model_path: Path = None,
        vectorizer_path: Path = None,
        label_enc_path: Path = None
    ):
        """
        Si se pasan las rutas a modelo/vectorizador/label encoder, se usan.
        Si no, busca por defecto en "models/".
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.label_enc_path = label_enc_path
        self.clf, self.le, self.vec = self.load_intent_model()

        

    # ------------------------------
    # Métodos internos
    # ------------------------------
    def _normalize(self, text: str) -> str:
        if text is None: return ""
        text = text.lower().strip()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"\s+", " ", text)
        return text

    def _score_match(self, text: str, patterns: Dict[str, List[str]]) -> float:
        must_any = patterns.get("must_any", [])
        if not must_any: return 0.0
        hits = sum(1 for pat in must_any if re.search(pat, text))
        return hits / max(1, len(must_any))

    # ------------------------------
    # Clasificación por reglas
    # ------------------------------
    def rule_based_intent(self, text: str) -> Tuple[str, float, str]:
        raw = text or ""
        ntext = self._normalize(raw)
        rationale_bits = []
        scores = []
        for intent in self.PRIMARY_INTENTS:
            s = self._score_match(ntext, self.PATTERNS.get(intent, {}))
            if s > 0:
                scores.append((intent, s))
                rationale_bits.append(f"{intent}={s:.2f}")

        if not scores:
            return "social", 0.30, "sin coincidencias; fallback a social"

        scores.sort(key=lambda x: x[1], reverse=True)
        best_intent, best_score = scores[0]

        # Boost explícito para 112/emergencia
        if best_intent == "alerta" and re.search(r"\b(llama (al|a) 112|emergencia|urgencias)\b", ntext):
            best_score = min(1.0, best_score + 0.2)
            rationale_bits.append("+0.2 por mención a 112/emergencia")

        return best_intent, float(best_score), "; ".join(rationale_bits) or "reglas básicas"

    # ------------------------------
    # Carga modelo ML
    # ------------------------------
    def load_intent_model(self):
        model_path = self.model_path or Path("models") / "intent_clf.joblib"
        vec_path = self.vectorizer_path or Path("models") / "vectorizer.joblib"
        label_enc_path = self.label_enc_path or Path("models") / "label_encoder.joblib"

        if model_path.exists() and vec_path.exists() and label_enc_path.exists():
            clf = joblib.load(model_path)
            vec = joblib.load(vec_path)
            le = joblib.load(label_enc_path)
            return clf, le, vec
        return None, None, None


    # ------------------------------
    # Predicción híbrida (reglas + ML)
    # ------------------------------
    def predict_intent(self, text: str, use_ml: bool = True,
                       rule_strict: float = 0.60, rule_soft: float = 0.20,
                       ml_threshold: float = 0.50, gap: float = 0.10) -> Dict[str, Any]:

        intent_r, score_r, why_r = self.rule_based_intent(text)

        LOCKED = {"alerta", "recordatorio"}
        if intent_r in LOCKED and score_r >= 0.50:
            return {
                "intent": intent_r,
                "score": float(score_r),
                "used": "rules",
                "rationale": "Intent bloqueado por seguridad",
                "rule": {"intent": intent_r, "score": float(score_r), "why": why_r},
                "ml": None
            }

        # Regla fuerte
        if score_r >= rule_strict:
            return {
                "intent": intent_r,
                "score": float(score_r),
                "used": "rules",
                "rationale": f"Regla fuerte (≥{rule_strict:.2f}): {why_r}",
                "rule": {"intent": intent_r, "score": float(score_r), "why": why_r},
                "ml": None
            }

        # ML
        ml_out = None
        if use_ml and self.clf is not None:
            probs = self.clf.predict_proba(self.vec.transform([text]))[0]
            idx = probs.argmax()
            intent_ml = self.le.inverse_transform([idx])[0]
            score_ml = float(probs[idx])
            ml_out = {"intent": intent_ml, "score": score_ml}
        else:
            score_ml = None

        # Comparación regla suave vs ML
        if rule_soft <= score_r < rule_strict:
            if ml_out and (ml_out["score"] >= ml_threshold) and (ml_out["score"] >= score_r + gap):
                return {
                    "intent": ml_out["intent"],
                    "score": ml_out["score"],
                    "used": "ml",
                    "rationale": f"ML fuerte (≥{ml_threshold:.2f}) y supera regla por ≥{gap:.2f}",
                    "rule": {"intent": intent_r, "score": float(score_r), "why": why_r},
                    "ml": ml_out
                }
            return {
                "intent": intent_r,
                "score": float(score_r),
                "used": "rules",
                "rationale": f"Regla moderada (≥{rule_soft:.2f}) y ML no supera gap",
                "rule": {"intent": intent_r, "score": float(score_r), "why": why_r},
                "ml": ml_out
            }

        # ML fuerte si regla débil
        if ml_out and ml_out["score"] >= ml_threshold:
            return {
                "intent": ml_out["intent"],
                "score": ml_out["score"],
                "used": "ml",
                "rationale": f"Sin regla fuerte; ML ≥ {ml_threshold:.2f}",
                "rule": {"intent": intent_r, "score": float(score_r), "why": why_r},
                "ml": ml_out
            }

        # Fallback
        return {
            "intent": "social",
            "score": 0.30,
            "used": "rules",
            "rationale": "Sin señal clara; fallback social",
            "rule": {"intent": intent_r, "score": float(score_r), "why": why_r},
            "ml": ml_out
        }

    # ------------------------------
    # Entrenamiento seguro ML
    # ------------------------------
    def train_intent_classifier_safe(
        self,
        train_csv_path: str = "intents_train.csv",
        text_col: str = "text",
        label_col: str = "intent",
        C: float = 2.0,
        max_features: int = 5000,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:

        train_csv_path = str(train_csv_path)
        p = Path(train_csv_path)
        if not p.exists():
            raise FileNotFoundError(f"No encuentro el CSV en: {p.resolve()}")

        df = pd.read_csv(p, encoding="utf-8-sig")
        df = df.dropna(subset=[text_col, label_col])
        df[text_col] = df[text_col].astype(str).str.strip()
        df[label_col] = df[label_col].astype(str).str.strip()

        counts = df[label_col].value_counts().to_dict()
        print("👉 Conteos por intent:", counts)
        if len(counts) < 2:
            raise ValueError("Se necesita al menos 2 clases distintas para entrenar.")

        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=1)
        X = vec.fit_transform(df[text_col].tolist())

        le = LabelEncoder()
        y = le.fit_transform(df[label_col].tolist())

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=None
            )

        clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs", multi_class="auto")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        report = classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0)
        print("\n=== Classification Report ===\n", report)

        MODEL_DIR = Path("models")
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(clf, MODEL_DIR / "intent_clf.joblib")
        joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
        joblib.dump(vec, MODEL_DIR / "vectorizer.joblib")

        # actualizar atributos de la clase
        self.clf, self.le, self.vec = clf, le, vec

        return {
            "report": report,
            "labels": list(le.classes_),
            "model_path": str((MODEL_DIR / "intent_clf.joblib").resolve()),
            "counts": counts,
        }

    # ------------------------------
    # Método legacy
    # ------------------------------
    def predict_intent_legacy(self, text, use_ml=True, rule_threshold=0.35, ml_threshold=0.50, gap=0.10):
        rule_soft = float(rule_threshold)
        rule_strict = max(0.60, rule_soft)
        return self.predict_intent(
            text,
            use_ml=use_ml,
            rule_strict=rule_strict,
            rule_soft=rule_soft,
            ml_threshold=float(ml_threshold),
            gap=float(gap)
        )
