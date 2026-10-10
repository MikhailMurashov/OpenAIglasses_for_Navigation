# omni_client.py
# -*- coding: utf-8 -*-
import os, base64
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple

from openai import OpenAI

# ===== Совместимость с OpenAI (режим совместимости DashScope от DAMO Academy) =====
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a9440db694924559ae4ebdc2023d2b9a")
if not API_KEY:
    raise RuntimeError("Не задан DASHSCOPE_API_KEY")

QWEN_MODEL = "qwen-omni-turbo"

# Режим совместимости
oai_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class OmniStreamPiece:
    """Единый внешний формат инкрементальных данных: text/audio — одно из двух или оба сразу."""
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64

async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",
    audio_format: str = "wav",
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    Запускает один раунд потокового диалога Omni-Turbo ChatCompletions:
    - content_list: content для OpenAI chat, мультимодальный (image_url/text)
    - возвращается с stream=True
    - инкрементальный вывод: OmniStreamPiece(text_delta=?, audio_b64=?)
    """
    completion = oai_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content_list}],
        modalities=["text", "audio"],
        audio={"voice": voice, "format": audio_format},
        stream=True,
        stream_options={"include_usage": True},
    )

    # Внимание: поток OpenAI SDK — синхронный итератор; в async-сценарии делаем yield поэлементно
    for chunk in completion:
        text_delta: Optional[str] = None
        audio_b64: Optional[str] = None

        if getattr(chunk, "choices", None):
            c0 = chunk.choices[0]
            delta = getattr(c0, "delta", None)
            # Инкремент текста
            if delta and getattr(delta, "content", None):
                piece = delta.content
                if piece:
                    text_delta = piece
            # Фрагмент аудио
            if delta and getattr(delta, "audio", None):
                aud = delta.audio
                audio_b64 = aud.get("data") if isinstance(aud, dict) else getattr(aud, "data", None)
            if audio_b64 is None:
                msg = getattr(c0, "message", None)
                if msg and getattr(msg, "audio", None):
                    ma = msg.audio
                    audio_b64 = ma.get("data") if isinstance(ma, dict) else getattr(ma, "data", None)

        if (text_delta is not None) or (audio_b64 is not None):
            yield OmniStreamPiece(text_delta=text_delta, audio_b64=audio_b64)
