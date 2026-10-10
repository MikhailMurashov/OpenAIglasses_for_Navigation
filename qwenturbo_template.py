from openai import OpenAI
import os

client = OpenAI(
    # Если переменная окружения не настроена, замените на API Key Alibaba Cloud Bailian: api_key="sk-xxx"
    api_key="sk-a9440db694924559ae4ebdc2023d2b9a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "你是谁"}]
completion = client.chat.completions.create(
    model="qwen-turbo",  # При необходимости можно заменить на другую модель с глубоким рассуждением
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
is_answering = False  # перешли ли к этапу ответа
print("\n" + "=" * 20 + "Процесс рассуждения" + "=" * 20)
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Полный ответ" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)