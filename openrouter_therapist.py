# openrouter_therapist.py
import requests
import json

class OpenRouterTherapist:
    def __init__(self, api_key: str = None, model: str = "meta-llama/llama-3-8b-instruct:free"):
        if not api_key:
            raise ValueError("Necesitas una API key de OpenRouter. Ve a https://openrouter.ai/")
        
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
        print(f"OpenRouterTherapist conectado correctamente (modelo: {self.model})")

    def get_therapy_response(self, user_message: str, emotional_analysis: dict, session_id: str = "default"):
        ea = emotional_analysis
        
        nombre = "Amigo/a"
        if any(palabra in user_message.lower() for palabra in ["me llamo", "soy", "mi nombre"]):
            for palabra in user_message.split():
                if palabra.istitle() and len(palabra) > 2:
                    nombre = palabra
                    break

        prompt = f"""
Eres un terapeuta experto en Terapia Cognitivo-Conductual (TCC), muy cálido, empático y humano.

INFORMACIÓN DEL PACIENTE:
- Nombre: {nombre}
- Mensaje: "{user_message}"
- Análisis emocional: {ea.get('emocion_principal', 'neutral')} ({ea.get('confianza_principal', 0):.0f}% confianza)
- Conflicto emocional detectado: {'SÍ' if ea.get('hay_conflicto') else 'NO'}

INSTRUCCIONES:
- Responde en español, máximo 3-4 oraciones
- Sé extremadamente empático y comprensivo
- Usa el nombre del paciente
- Aplica una técnica breve de TCC apropiada
- Mantén un tono cálido y profesional
- Enfócate en validar emociones y ofrecer apoyo concreto
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "Eres un terapeuta profesional especializado en TCC. Eres cálido, empático y efectivo."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                respuesta = data['choices'][0]['message']['content'].strip()
                print(f"🤖 OpenRouter: {respuesta}")
                return {
                    "response": respuesta, 
                    "success": True,
                    "tokens_used": data.get('usage', {}).get('total_tokens', 0)
                }
            else:
                print(f"❌ Error OpenRouter {response.status_code}: {response.text}")
                return {
                    "response": f"{nombre}, entiendo que estás pasando por un momento difícil. Estoy aquí para escucharte y apoyarte. ¿Podrías contarme más sobre cómo te sientes?",
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            print("❌ Timeout conectando con OpenRouter")
            return {
                "response": f"{nombre}, estoy aquí contigo. Parece que hay problemas de conexión, pero quiero que sepas que tus sentimientos son importantes. ¿Qué te gustaría compartir?",
                "success": False,
                "error": "timeout"
            }
        except Exception as e:
            print(f"❌ Error conexión OpenRouter: {e}")
            return {
                "response": f"{nombre}, estoy procesando tus emociones. Todo lo que sientes es válido y merece ser escuchado. ¿En qué puedo ayudarte hoy?",
                "success": False,
                "error": str(e)
            }