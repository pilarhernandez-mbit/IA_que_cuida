
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class TemplateEngine:
    language_simple: bool = True
    tono_cercano: bool = True
    default_tts_mode: str = "voz"
    templates: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.templates:
            self.templates = {
                "recordatorio": ("{saludo} {user}, te lo apunto.\n🗓️ Recordatorio: {actividad}{cuando}.\n{extra}\n{tts}"),
                "alerta": ("{saludo} {user}. Estoy aquí para ayudarte.\n⚠️ He entendido: {motivo}.\n{accion_alerta}\n¿Quieres que avise a {contacto} ahora? {telefono}\n{consejo_breve}\n{tts}"),
                "social": ("{saludo} {user}.\n{apertura}\n{pregunta_social}\n{tts}"),
                "rechazo": ("Entiendo, {user}. No insistiré.\n{cierre_respetuoso}\nSi cambias de idea, solo dímelo.\n{tts}"),
                "fallback": ("{saludo} {user}. No estoy seguro de lo que necesitas.\n¿Quieres que lo intentemos de otra forma o que avise a alguien de confianza?\n{tts}"),
            }

    def _saludo(self): return "Hola" if self.tono_cercano else "Buenos días"
    def _user(self, name): return "" if not name else name
    def _cuando(self, fecha, hora):
        partes=[]; 
        if fecha: partes.append(f"el {fecha}")
        if hora: partes.append(f"a las {hora}")
        return " " + " ".join(partes) if partes else ""
    def _extra_recordatorio(self): 
        return "Te avisaré a tiempo. ¿Quieres una repetición diaria?" if self.language_simple else "He creado el recordatorio. Puedo programarlo repetido (diario/semanal)."
    def _accion_alerta(self, motivo, contacto_nombre, telefono):
        grave=False; t=(motivo or "").lower()
        for kw in ["caida","ahogo","sangra","muy mal","dolor fuerte","mareo intenso"]:
            if kw in t: grave=True; break
        base = "Voy a priorizar tu seguridad. Si es una emergencia, marca 112. " if grave else "Vamos a gestionarlo con calma. "
        dest = contacto_nombre or "tu contacto de confianza"
        tel = f"(tel: {telefono})" if telefono else ""
        return base + f"Puedo llamar a {dest} {tel} si me lo confirmas."
    def _consejo_breve(self):
        return "Mientras tanto, siéntate y respira despacio. Estoy contigo." if self.language_simple else "Mientras tanto, adopta una postura cómoda, controla la respiración y mantén el teléfono cerca."
    def _apertura_social(self, emocion):
        if emocion and emocion.lower() in {"triste","solo","sola"}:
            return "Siento que te sientas así. ¿Te apetece que hablemos un rato?"
        return "¿Te cuento una curiosidad del día o prefieres música?" if self.language_simple else "Puedo proponerte una actividad breve, poner música o charlar un poco."
    def _pregunta_social(self): 
        return "¿Quieres que llame a alguien de tu familia?" if self.language_simple else "Si quieres, puedo contactar con un familiar o amigo para que os saludéis."
    def _cierre_rechazo(self):
        return "Gracias por decírmelo. Me quedaré atento por si me necesitas." if self.language_simple else "De acuerdo, respetaré tu decisión y permaneceré disponible si cambias de opinión."
    def _tts_tag(self, tts_mode):
        mode=(tts_mode or self.default_tts_mode).lower()
        if mode not in {"voz","texto","ambos"}: mode=self.default_tts_mode
        return f"(modo salida: {mode})"

    def render(self, intent: str, context: Dict[str, Any]) -> str:
        intent=(intent or "fallback").lower()
        tpl=self.templates.get(intent, self.templates["fallback"])
        user=self._user(context.get("user_name"))
        fecha=context.get("fecha"); hora=context.get("hora")
        actividad=context.get("actividad") or "la tarea"
        motivo=context.get("motivo")
        emocion=context.get("emocion")
        contacto_nombre=context.get("contacto_nombre")
        contacto_telefono=context.get("contacto_telefono")
        tts_mode=context.get("tts_mode")
        filled = tpl.format(
            saludo=self._saludo(),
            user=user,
            actividad=actividad,
            cuando=self._cuando(fecha, hora),
            extra=self._extra_recordatorio(),
            motivo=(motivo or "necesitas ayuda"),
            accion_alerta=self._accion_alerta(motivo, contacto_nombre, contacto_telefono),
            contacto=(contacto_nombre or "tu contacto de confianza"),
            telefono=(f"(tel: {contacto_telefono})" if contacto_telefono else ""),
            consejo_breve=self._consejo_breve(),
            apertura=self._apertura_social(emocion),
            pregunta_social=self._pregunta_social(),
            cierre_respetuoso=self._cierre_rechazo(),
            tts=self._tts_tag(tts_mode),
        )
        return filled.strip()


# Templates muy simples de ejemplo
def get_template_for_intent(intent: str, perfil_usuario: str, nombre: str, entities=None, emotion=None) -> str:
    intent = (intent or '').lower()
    if intent == 'recordatorio':
        return f"{nombre}, he apuntado el recordatorio. Te avisaré a la hora indicada."
    if intent == 'ayuda':
        return f"{nombre}, te ayudo con eso. Dime exactamente qué necesitas y lo gestiono."
    if intent == 'rechazo':
        return f"De acuerdo, {nombre}. No haré esa acción. Si cambias de idea, dímelo."
    if intent == 'social':
        return f"Me alegra hablar contigo, {nombre}. ¿Te apetece que escuchemos tu música favorita?"
    # abierta / fallback
    return f"Te he entendido, {nombre}. ¿Quieres que lo anote o que te recuerde algo concreto?"

