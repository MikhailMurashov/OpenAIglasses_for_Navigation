# qwen_extractor.py
# -*- coding: utf-8 -*-
from typing import List, Tuple
import os
from openai import OpenAI

# —— Локальное приоритетное сопоставление (можно в любой момент расширить/переименовать) ——
LOCAL_CN2EN = {
    "红牛": "Red_Bull",
    "ad钙奶": "AD_milk",
    "ad 钙奶": "AD_milk",
    "ad": "AD_milk",
    "钙奶": "AD_milk",
    "矿泉水": "bottle",
    "水瓶": "bottle",
    "可乐": "coke",
    "雪碧": "sprite",
}

def _make_client() -> OpenAI:
    # Переиспользуем ваш совместимый эндпоинт Bailian; поддерживается чтение из переменных окружения
    base_url = os.getenv("DASHSCOPE_COMPAT_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key  = "sk-a9440db694924559ae4ebdc2023d2b9a"
    return OpenAI(api_key=api_key, base_url=base_url)

PROMPT_SYS = (
    "You are a label normalizer. Convert the given Chinese object "
    "description into a short, lowercase English YOLO/vision class name "
    "(1~3 words). If multiple are given, return the single most likely one. "
    "Output ONLY the label, no punctuation."
)

def extract_english_label(query_cn: str) -> Tuple[str, str]:
    """
    Возвращает (label_en, source); source ∈ {'local', 'qwen', 'fallback'}
    """
    q = (query_cn or "").strip().lower()
    if q in LOCAL_CN2EN:
        return LOCAL_CN2EN[q], "local"

    # Простое правило: убираем префиксные определения
    for k, v in LOCAL_CN2EN.items():
        if k in q:
            return v, "local"

    # Вызов Qwen Turbo (совместимый с Chat Completions)
    try:
        client = _make_client()
        msgs = [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user",   "content": query_cn.strip()},
        ]
        rsp = client.chat.completions.create(
            model=os.getenv("QWEN_MODEL", "qwen-turbo"),
            messages=msgs,
            stream=False
        )
        label = (rsp.choices[0].message.content or "").strip()
        # Немного чистим
        label = label.replace(".", "").replace(",", "").replace("  ", " ").strip()
        # Запасной вариант: если пусто — возвращаем 'bottle'
        return (label or "bottle"), "qwen"
    except Exception:
        return "bottle", "fallback"
