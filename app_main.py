# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, audioop
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re
# Добавить после остальных import:
from qwen_extractor import extract_english_label
from navigation_master import NavigationMaster, OrchestratorResult 
# Новое: импорт навигатора по тактильной дорожке
from workflow_blindpath import BlindPathNavigator
# Новое: импорт навигатора перехода через дорогу
from workflow_crossstreet import CrossStreetNavigator
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from obstacle_detector_client import ObstacleDetectorClient

import torch  # добавить эту строку


import mediapipe as mp
import bridge_io
import threading
import yolomedia  # должен лежать в одном каталоге с app_main.py, имя файла — yolomedia.py
# ---- Политика цикла событий для Windows ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- Основа DashScope ASR ----
from dashscope import audio as dash_audio  # если не установлен, исходный проект выбросит ошибку с подсказкой

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a9440db694924559ae4ebdc2023d2b9a")
if not API_KEY:
    raise RuntimeError("Не задан DASHSCOPE_API_KEY")

MODEL        = "paraformer-realtime-v2"
SAMPLE_RATE  = 16000
AUDIO_FMT    = "pcm"
CHUNK_MS     = 20
BYTES_CHUNK  = SAMPLE_RATE * CHUNK_MS // 1000 * 2
SILENCE_20MS = bytes(BYTES_CHUNK)

# ---- Подключаем наши модули ----
from audio_stream import (
    register_stream_route,         # монтирует /stream.wav
    broadcast_pcm16_realtime,      # рассылка 16k PCM по подключениям в реальном времени
    hard_reset_audio,              # общий рубильник воспроизведения аудио + AI
    BYTES_PER_20MS_16K,
    is_playing_now,
    current_ai_task,
)
from omni_client import stream_chat, OmniStreamPiece
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
)
from audio_player import initialize_audio_system, play_voice_text

# ---- Синхронный рекордер ----
import sync_recorder
import signal
import atexit

# ---- IMU UDP ----
UDP_IP   = "0.0.0.0"
UDP_PORT = 12345

app = FastAPI()

# ====== Состояние и контейнеры ======
app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None

# [Новое] глобальные переменные навигации по тактильной дорожке
blind_path_navigator = None
navigation_active = False
yolo_seg_model = None
obstacle_detector = None

# [Новое] глобальные переменные навигации перехода через дорогу
cross_street_navigator = None
cross_street_active = False
orchestrator = None  # новое

# [Новое] флаг состояния omni-диалога
omni_conversation_active = False  # отмечает, идёт ли сейчас omni-диалог
omni_previous_nav_state = None  # сохраняет состояние навигации до активации omni для последующего восстановления

# [Новое] функция загрузки моделей
def load_navigation_models():
    """Загружает модели, необходимые для навигации по тактильной дорожке"""
    global yolo_seg_model, obstacle_detector

    try:
        seg_model_path = os.getenv("BLIND_PATH_MODEL", r"C:\Users\Administrator\Desktop\rebuild1002\model\yolo-seg.pt")
        #print(f"[NAVIGATION] Пытаемся загрузить модель: {seg_model_path}")

        if os.path.exists(seg_model_path):
            print(f"[NAVIGATION] Файл модели найден, начинаем загрузку...")
            yolo_seg_model = YOLO(seg_model_path)

            # принудительно переносим на GPU
            if torch.cuda.is_available():
                yolo_seg_model.to("cuda")
                print(f"[NAVIGATION] Модель сегментации тактильной дорожки загружена и перенесена на GPU: {yolo_seg_model.device}")
            else:
                print("[NAVIGATION] CUDA недоступна, модель остаётся на CPU")

            # проверяем, что модель нормально работает
            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                results = yolo_seg_model.predict(
                    test_img,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=False
                )
                print(f"[NAVIGATION] Тест модели пройден, число поддерживаемых классов: {len(yolo_seg_model.names) if hasattr(yolo_seg_model, 'names') else 'неизвестно'}")
                if hasattr(yolo_seg_model, 'names'):
                    print(f"[NAVIGATION] Классы модели: {yolo_seg_model.names}")
            except Exception as e:
                print(f"[NAVIGATION] Тест модели не пройден: {e}")
        else:
            print(f"[NAVIGATION] Ошибка: файл модели не найден: {seg_model_path}")
            print(f"[NAVIGATION] Текущий рабочий каталог: {os.getcwd()}")
            print(f"[NAVIGATION] Проверьте правильность пути к файлу")
            
        # [Начало изменений] используем ObstacleDetectorClient вместо прямого YOLO
        obstacle_model_path = os.getenv("OBSTACLE_MODEL", r"C:\Users\Administrator\Desktop\rebuild1002\model\yoloe-11l-seg.pt")
        print(f"[NAVIGATION] Пытаемся загрузить модель детекции препятствий: {obstacle_model_path}")
        
        if os.path.exists(obstacle_model_path):
            print(f"[NAVIGATION] Файл модели детекции препятствий найден, начинаем загрузку...")
            try:
                # используем YOLO-E, обёрнутый в ObstacleDetectorClient
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                print(f"[NAVIGATION] ========== Детектор препятствий YOLO-E успешно загружен ==========")
                
                # проверяем, успешно ли загружена модель
                if hasattr(obstacle_detector, 'model') and obstacle_detector.model is not None:
                    print(f"[NAVIGATION] Модель YOLO-E инициализирована")
                    print(f"[NAVIGATION] Устройство модели: {next(obstacle_detector.model.parameters()).device}")
                else:
                    print(f"[NAVIGATION] Предупреждение: ошибка инициализации модели YOLO-E")
                
                # проверяем, успешно ли загружен белый список
                if hasattr(obstacle_detector, 'WHITELIST_CLASSES'):
                    print(f"[NAVIGATION] Число классов в белом списке: {len(obstacle_detector.WHITELIST_CLASSES)}")
                    print(f"[NAVIGATION] Первые 10 классов белого списка: {', '.join(obstacle_detector.WHITELIST_CLASSES[:10])}")
                else:
                    print(f"[NAVIGATION] Предупреждение: классы белого списка не определены")
                
                # проверяем, успешно ли предвычислены текстовые признаки
                if hasattr(obstacle_detector, 'whitelist_embeddings') and obstacle_detector.whitelist_embeddings is not None:
                    print(f"[NAVIGATION] Текстовые признаки YOLO-E предвычислены")
                    print(f"[NAVIGATION] Форма тензора текстовых признаков: {obstacle_detector.whitelist_embeddings.shape if hasattr(obstacle_detector.whitelist_embeddings, 'shape') else 'неизвестно'}")
                else:
                    print(f"[NAVIGATION] Предупреждение: текстовые признаки YOLO-E не предвычислены")
                
                # тестируем детекцию препятствий
                print(f"[NAVIGATION] Начинаем тест детекции YOLO-E...")
                try:
                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    # рисуем на тестовом изображении белый прямоугольник, имитирующий объект
                    cv2.rectangle(test_img, (200, 200), (400, 400), (255, 255, 255), -1)
                    
                    # тестовая детекция (без path_mask)
                    test_results = obstacle_detector.detect(test_img)
                    print(f"[NAVIGATION] Тест детекции YOLO-E пройден!")
                    print(f"[NAVIGATION] Число результатов тестовой детекции: {len(test_results)}")
                    
                    if len(test_results) > 0:
                        print(f"[NAVIGATION] Объекты, найденные в тесте:")
                        for i, obj in enumerate(test_results):
                            print(f"  - Объект {i+1}: {obj.get('name', 'unknown')}, "
                                  f"доля площади: {obj.get('area_ratio', 0):.3f}, "
                                  f"позиция: ({obj.get('center_x', 0):.0f}, {obj.get('center_y', 0):.0f})")
                except Exception as e:
                    print(f"[NAVIGATION] Тест детекции YOLO-E не пройден: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"[NAVIGATION] ========== Загрузка детектора препятствий YOLO-E завершена ==========")
                
            except Exception as e:
                print(f"[NAVIGATION] Не удалось загрузить детектор препятствий: {e}")
                import traceback
                traceback.print_exc()
                obstacle_detector = None
        else:
            print(f"[NAVIGATION] Предупреждение: файл модели детекции препятствий не найден: {obstacle_model_path}")
        
    except Exception as e:
        print(f"[NAVIGATION] Ошибка загрузки моделей: {e}")
        import traceback
        traceback.print_exc()

# Загружаем модели при запуске программы
print("[NAVIGATION] Начинаем загрузку моделей навигации...")
load_navigation_models()
print(f"[NAVIGATION] Загрузка моделей завершена - yolo_seg_model: {yolo_seg_model is not None}")

# [Новое] запуск синхронной записи
print("[RECORDER] Запуск системы синхронной записи...")
sync_recorder.start_recording()
print("[RECORDER] Система записи запущена, видео и аудио будут сохраняться автоматически")

# [Новое] регистрируем обработчик выхода, чтобы при Ctrl+C файлы записи сохранялись
def cleanup_on_exit():
    """Очистка при завершении программы"""
    print("\n[SYSTEM] Закрываем рекордер...")
    try:
        sync_recorder.stop_recording()
        print("[SYSTEM] Файлы записи сохранены")
    except Exception as e:
        print(f"[SYSTEM] Ошибка при закрытии рекордера: {e}")

def signal_handler(sig, frame):
    """Обработка сигнала Ctrl+C"""
    print("\n[SYSTEM] Получен сигнал прерывания, выполняем безопасный выход...")
    cleanup_on_exit()
    import sys
    sys.exit(0)

# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # сигнал завершения
atexit.register(cleanup_on_exit)  # вызывается и при обычном выходе

print("[RECORDER] Обработчик выхода зарегистрирован - при Ctrl+C файлы записи сохранятся автоматически")



# [Новое] предзагрузка модели детекции светофора (чтобы не было подвисания при входе в состояние WAIT_TRAFFIC_LIGHT)
try:
    import trafficlight_detection
    print("[TRAFFIC_LIGHT] Начинаем предзагрузку модели детекции светофора...")
    if trafficlight_detection.init_model():
        print("[TRAFFIC_LIGHT] Модель детекции светофора успешно предзагружена")
        # выполняем один тестовый инференс, чтобы полностью прогреть модель
        try:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = trafficlight_detection.process_single_frame(test_img)
            print("[TRAFFIC_LIGHT] Прогрев модели завершён")
        except Exception as e:
            print(f"[TRAFFIC_LIGHT] Ошибка прогрева модели: {e}")
    else:
        print("[TRAFFIC_LIGHT] Не удалось предзагрузить модель детекции светофора")
except Exception as e:
    print(f"[TRAFFIC_LIGHT] Ошибка предзагрузки модели светофора: {e}")

# ============== Ключевое: общесистемный рубильник «жёсткого сброса» =================
interrupt_lock = asyncio.Lock()

# ============== Управление медиапотоком YOLO =================
yolomedia_thread: Optional[threading.Thread] = None
yolomedia_stop_event = threading.Event()
yolomedia_running = False
yolomedia_sending_frames = False  # новое: отмечает, начал ли YOLO отправлять обработанные кадры

# Сопоставление названий предметов классам YOLO
ITEM_TO_CLASS_MAP = {
    "红牛": "Red_Bull",
    "AD钙奶": "AD_milk",
    "ad钙奶": "AD_milk",
    "钙奶": "AD_milk",
}

async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)

async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)

async def full_system_reset(reason: str = ""):
    """
    Возврат в состояние сразу после запуска:
    1) остановка воспроизведения + отмена задачи AI + обрыв всех /stream.wav (hard_reset_audio)
    2) остановка потока распознавания ASR в реальном времени (ключевое)
    3) очистка состояния UI
    4) очистка последнего кадра камеры (чтобы старый кадр не попал в следующий раунд)
    5) уведомление ESP32: RESET (опционально)
    """
    # 1) Аудио и AI
    await hard_reset_audio(reason or "full_system_reset")

    # 2) ASR
    await stop_current_recognition()

    # 3) UI
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    # 4) Кадр камеры
    try:
        last_frames.clear()
    except Exception:
        pass

    # 5) Уведомляем ESP32
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    print("[SYSTEM] full reset done.", flush=True)

# ========= Запуск/остановка медиаобработки YOLO =========
def start_yolomedia_with_target(target_name: str):
    """Запускает поток yolomedia для поиска указанного предмета"""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    # если уже запущен, сначала останавливаем
    if yolomedia_running:
        stop_yolomedia()
    
    # ищем соответствующий класс YOLO
    yolo_class = ITEM_TO_CLASS_MAP.get(target_name, target_name)
    print(f"[YOLOMEDIA] Starting with target: {target_name} -> YOLO class: {yolo_class}", flush=True)
    print(f"[YOLOMEDIA] Available mappings: {ITEM_TO_CLASS_MAP}", flush=True)  # добавлена эта строка для отладки
    
    yolomedia_stop_event.clear()
    yolomedia_running = True
    yolomedia_sending_frames = False  # сбрасываем состояние отправки кадров
    
    def _run():
        try:
            # передаём имя целевого класса и событие остановки
            yolomedia.main(headless=True, prompt_name=yolo_class, stop_event=yolomedia_stop_event)
        except Exception as e:
            print(f"[YOLOMEDIA] worker stopped: {e}", flush=True)
        finally:
            global yolomedia_running, yolomedia_sending_frames
            yolomedia_running = False
            yolomedia_sending_frames = False
    
    yolomedia_thread = threading.Thread(target=_run, daemon=True)
    yolomedia_thread.start()
    print(f"[YOLOMEDIA] background worker started for: {yolo_class} (идёт инициализация, пока показываем исходное изображение)", flush=True)

def stop_yolomedia():
    """Останавливает поток yolomedia"""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    if yolomedia_running:
        print("[YOLOMEDIA] Stopping worker...", flush=True)
        yolomedia_stop_event.set()
        
        # ждём завершения потока (не более 5 секунд)
        if yolomedia_thread and yolomedia_thread.is_alive():
            yolomedia_thread.join(timeout=5.0)
        
        yolomedia_running = False
        yolomedia_sending_frames = False
        
        # [Новое] если orchestrator в режиме поиска предмета, при завершении автоматически не восстанавливаем (управляется командами)
        # достаточно сбросить флаги
        print("[YOLOMEDIA] Worker stopped, ожидаем переключения состояния.", flush=True)

# ========= Собственная версия start_ai_with_text с распознаванием специальных команд =========
async def start_ai_with_text_custom(user_text: str):
    """Расширенная функция запуска AI с распознаванием специальных команд"""
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator
    
    # [Изменено] в режимах навигации и детекции светофора omni-диалог запускается только по определённым словам
    if orchestrator:
        current_state = orchestrator.get_state()
        # если в режиме навигации или детекции светофора (не CHAT)
        if current_state not in ["CHAT", "IDLE"]:
            # проверяем, является ли фраза разрешённым триггером диалога
            allowed_keywords = ["帮我看", "帮我看下", "帮我找", "找一下", "看看", "识别一下"]
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)
            
            # проверяем, является ли фраза командой управления навигацией
            nav_control_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", 
                                   "检测红绿灯", "看红绿灯", "停止检测", "停止红绿灯"]
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)
            
            # если это ни разрешённый запрос, ни команда управления навигацией — отбрасываем
            if not is_allowed_query and not is_nav_control:
                mode_name = "детекция светофора" if current_state == "TRAFFIC_LIGHT_DETECTION" else "навигация"
                print(f"[Режим: {mode_name}] Отброшена речь вне диалога: {user_text}")
                return  # сразу отбрасываем, в omni не передаём
    
    # [Изменено] проверяем команды перехода через дорогу - управление через orchestrator
    if "开始过马路" in user_text or "帮我过马路" in user_text:
        # [Новое] если идёт поиск предмета, сначала останавливаем
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] Переключение из режима поиска предмета в режим перехода через дорогу")
        
        if orchestrator:
            orchestrator.start_crossing()
            print(f"[CROSS_STREET] Режим перехода через дорогу запущен, состояние: {orchestrator.get_state()}")
            # воспроизводим голосовое сообщение о запуске и рассылаем в UI
            play_voice_text("过马路模式已启动。")
            await ui_broadcast_final("[Система] Режим перехода через дорогу запущен")
        else:
            print("[CROSS_STREET] Предупреждение: оркестратор навигации не инициализирован!")
            play_voice_text("启动过马路模式失败，请稍后重试。")
            await ui_broadcast_final("[Система] Система навигации не готова")
        return
    
    if "过马路结束" in user_text or "结束过马路" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[CROSS_STREET] Навигация остановлена, состояние: {orchestrator.get_state()}")
            # воспроизводим голосовое сообщение об остановке и рассылаем в UI
            play_voice_text("已停止导航。")
            await ui_broadcast_final("[Система] Режим перехода через дорогу остановлен")
        else:
            await ui_broadcast_final("[Система] Система навигации не запущена")
        return
    
    # [Изменено] проверяем команды детекции светофора - взаимоисключение с навигацией по тактильной дорожке
    if "检测红绿灯" in user_text or "看红绿灯" in user_text:
        try:
            import trafficlight_detection
            
            # переключаем orchestrator в режим детекции светофора (навигация по тактильной дорожке приостанавливается)
            if orchestrator:
                orchestrator.start_traffic_light_detection()
                print(f"[TRAFFIC] Переключение в режим детекции светофора, состояние: {orchestrator.get_state()}")
            
            # [Улучшено] работаем в основном потоке, а не в отдельном, чтобы избежать пропуска кадров
            success = trafficlight_detection.init_model()  # только инициализируем модель, поток не запускаем
            trafficlight_detection.reset_detection_state()  # сбрасываем состояние
            
            if success:
                await ui_broadcast_final("[Система] Детекция светофора запущена")
            else:
                await ui_broadcast_final("[Система] Не удалось загрузить модель светофора")
        except Exception as e:
            print(f"[TRAFFIC] Не удалось запустить детекцию светофора: {e}")
            await ui_broadcast_final(f"[Система] Ошибка запуска: {e}")
        return
    
    if "停止检测" in user_text or "停止红绿灯" in user_text:
        try:
            # возвращаемся в режим диалога
            if orchestrator:
                orchestrator.stop_navigation()  # возврат в режим CHAT
                print(f"[TRAFFIC] Детекция светофора остановлена, восстановлен режим {orchestrator.get_state()}")
            
            await ui_broadcast_final("[Система] Детекция светофора остановлена")
        except Exception as e:
            print(f"[TRAFFIC] Не удалось остановить детекцию светофора: {e}")
            await ui_broadcast_final(f"[Система] Ошибка остановки: {e}")
        return
    
    # [Изменено] проверяем команды навигации - управление через orchestrator
    if "开始导航" in user_text or "盲道导航" in user_text or "帮我导航" in user_text:
        # [Новое] если идёт поиск предмета, сначала останавливаем
        if yolomedia_running:
            stop_yolomedia()
            print("[ITEM_SEARCH] Переключение из режима поиска предмета в навигацию по тактильной дорожке")
        
        if orchestrator:
            orchestrator.start_blind_path_navigation()
            print(f"[NAVIGATION] Навигация по тактильной дорожке запущена, состояние: {orchestrator.get_state()}")
            await ui_broadcast_final("[Система] Навигация по тактильной дорожке запущена")
        else:
            print("[NAVIGATION] Предупреждение: оркестратор навигации не инициализирован!")
            await ui_broadcast_final("[Система] Система навигации не готова")
        return
    
    if "停止导航" in user_text or "结束导航" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[NAVIGATION] Навигация остановлена, состояние: {orchestrator.get_state()}")
            await ui_broadcast_final("[Система] Навигация по тактильной дорожке остановлена")
        else:
            await ui_broadcast_final("[Система] Система навигации не запущена")
        return

    nav_cmd_keywords = ["开始过马路", "过马路结束", "开始导航", "盲道导航", "停止导航", "结束导航", "立即通过", "现在通过", "继续"]
    if any(k in user_text for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[Система] Режим навигации обновлён")
        else:
            await ui_broadcast_final("[Система] Оркестратор навигации не инициализирован")
        return    

    # проверяем команду вида "帮我找/识别一下xxx" (найди/распознай xxx)
    # расширенное регулярное выражение с поддержкой большего числа ключевых слов
    find_pattern = r"(?:^\s*帮我)?\s*找一下\s*(.+?)(?:。|！|？|$)"
    match = re.search(find_pattern, user_text)
        
    if match:
        # извлекаем китайское название предмета
        item_cn = match.group(1).strip()
        if item_cn:
            # [Новое] получаем английское имя класса через локальное сопоставление + Qwen
            label_en, src = extract_english_label(item_cn)
            print(f"[COMMAND] Finder request: '{item_cn}' -> '{label_en}' (src={src})", flush=True)

            # [Новое] переключаемся в режим поиска предмета (навигация приостанавливается)
            if orchestrator:
                orchestrator.start_item_search()
                print(f"[ITEM_SEARCH] Переключено в режим поиска предмета, состояние: {orchestrator.get_state()}")
            
            # [Ключевое] передаём английское имя класса в yolomedia (если класс не найден, он автоматически переключится на YOLOE)
            start_yolomedia_with_target(label_en)

            # даём подтверждение во фронтенд/голосом
            try:
                await ui_broadcast_final(f"[Поиск предмета] Ищем {item_cn}...")
            except Exception:
                pass

            return
    
    # проверяем команду "找到了" (нашёл)
    if "找到了" in user_text or "拿到了" in user_text:
        print("[COMMAND] Found command detected", flush=True)
        # останавливаем yolomedia
        stop_yolomedia()
        
        # [Новое] выходим из режима поиска предмета и восстанавливаем прежнее состояние навигации
        if orchestrator:
            orchestrator.stop_item_search(restore_nav=True)
            current_state = orchestrator.get_state()
            print(f"[ITEM_SEARCH] Поиск предмета завершён, текущее состояние: {current_state}")
            
            # даём обратную связь в зависимости от восстановленного состояния
            if current_state in ["BLINDPATH_NAV", "SEEKING_CROSSWALK", "WAIT_TRAFFIC_LIGHT", "CROSSING", "SEEKING_NEXT_BLINDPATH"]:
                await ui_broadcast_final("[Поиск предмета] Предмет найден, продолжаем навигацию.")
            else:
                await ui_broadcast_final("[Поиск предмета] Предмет найден.")
        else:
            await ui_broadcast_final("[Поиск предмета] Предмет найден.")
        
        return
    
    # [Изменено] при начале omni-диалога переключаемся в режим CHAT
    global omni_conversation_active, omni_previous_nav_state
    omni_conversation_active = True
    
    # сохраняем текущее состояние навигации и переключаемся в режим CHAT
    if orchestrator:
        current_state = orchestrator.get_state()
        # сохранять и переключать нужно только в режиме навигации
        if current_state not in ["CHAT", "IDLE"]:
            omni_previous_nav_state = current_state
            orchestrator.force_state("CHAT")
            print(f"[OMNI] Начало диалога, переключение из {current_state} в режим CHAT")
        else:
            omni_previous_nav_state = None
            print(f"[OMNI] Начало диалога (уже в режиме {current_state})")
    
    # если это не специальная команда, выполняем исходную логику AI-диалога
    # но если работает yolomedia, обычный диалог пока не обрабатываем
    if yolomedia_running:
        print("[AI] YOLO media is running, skipping normal AI response", flush=True)
        return
    
    # исходная логика AI-диалога
    await start_ai_with_text(user_text)

# ========= Запуск воспроизведения Omni =========
async def start_ai_with_text(user_text: str):
    """После жёсткого сброса запускает новый голосовой вывод AI."""
    async def _runner():
        txt_buf: List[str] = []
        rate_state = None

        # собираем (изображение + текст)
        content_list = []
        if last_frames:
            try:
                _, jpeg_bytes = last_frames[-1]
                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})

        try:
            async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
                # инкремент текста (только UI)
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception:
                        pass

                # фрагмент аудио: Omni возвращает wav audio.data 24k (PCM16) в Base64; для нисходящего потока нужен 8k PCM16
                if piece.audio_b64:
                    try:
                        pcm24 = base64.b64decode(piece.audio_b64)
                    except Exception:
                        pcm24 = b""
                    if pcm24:
                        # 24k → 8k (ratecv сохраняет высоту тона и скорость)
                        pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                        pcm8k = audioop.mul(pcm8k, 2, 0.60)
                        if pcm8k:
                            await broadcast_pcm16_realtime(pcm8k)

        except asyncio.CancelledError:
            # прервано новым раундом
            raise
        except Exception as e:
            try:
                await ui_broadcast_final(f"[AI] Произошла ошибка: {e}")
            except Exception:
                pass
        finally:
            # [Изменено] отмечаем конец omni-диалога и восстанавливаем прежний режим навигации
            global omni_conversation_active, omni_previous_nav_state
            omni_conversation_active = False
            
            # восстанавливаем прежнее состояние навигации
            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                print(f"[OMNI] Диалог завершён, восстановлен режим {omni_previous_nav_state}")
                omni_previous_nav_state = None
            else:
                print(f"[OMNI] Диалог завершён (восстанавливать состояние навигации не нужно)")
            
            # при естественном завершении отправляем текущим подключениям сигнал «конец»
            from audio_stream import stream_clients  # локальный импорт, чтобы избежать циклической зависимости
            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try: sc.q.put_nowait(b"\x00"*BYTES_PER_20MS_16K)  # один кадр тишины
                    except Exception: pass
                    try: sc.q.put_nowait(None)
                    except Exception: pass

            final_text = ("".join(txt_buf)).strip() or "(пустой ответ)"
            try:
                await ui_broadcast_final("[AI] " + final_text)
            except Exception:
                pass

    # перед реальным запуском делаем жёсткий сброс, чтобы **гарантированно** не осталось старого аудио
    await hard_reset_audio("start_ai_with_text")
    loop = asyncio.get_running_loop()
    from audio_stream import current_ai_task as _task_holder  # чтение/запись глобальной переменной модуля
    from audio_stream import __dict__ as _as_dict
    # устанавливаем current_ai_task внутри модуля
    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task

# ---------- Страница / health ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"

# Регистрируем /stream.wav
register_stream_route(app)

# ---------- WebSocket: текст WebUI (рассылка статуса ASR/AI) ----------
@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)

# ---------- WebSocket: аудиовход ESP32 (восходящий поток ASR) ----------
@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] client connected")
    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None

    async def stop_rec(send_notice: Optional[str] = None):
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try: await keepalive_task
            except Exception: pass
        keepalive_task = None
        if recognition:
            try: recognition.stop()
            except Exception: pass
            recognition = None
        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try: await ws.send_text(send_notice)
            except Exception: pass

    async def on_sdk_error(_msg: str):
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop():
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):  # ~600 мс тишины
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("keepalive send failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    print("[AUDIO] START received")
                    await stop_rec()
                    loop = asyncio.get_running_loop()
                    def post(coro):
                        asyncio.run_coroutine_threadsafe(coro, loop)

                    # собираем колбэк ASR (внедряем все зависимости)
                    cb = ASRCallback(
                        on_sdk_error=lambda s: post(on_sdk_error(s)),
                        post=post,
                        ui_broadcast_partial=ui_broadcast_partial,
                        ui_broadcast_final=ui_broadcast_final,
                        is_playing_now_fn=is_playing_now,
                        start_ai_with_text_fn=start_ai_with_text_custom,  # используем собственную версию
                        full_system_reset_fn=full_system_reset,
                        interrupt_lock=interrupt_lock,
                    )

                    recognition = dash_audio.asr.Recognition(
                        api_key=API_KEY, model=MODEL, format=AUDIO_FMT,
                        sample_rate=SAMPLE_RATE, callback=cb
                    )
                    recognition.start()
                    await set_current_recognition(recognition)
                    streaming = True
                    last_ts = time.monotonic()
                    keepalive_task = asyncio.create_task(keepalive_loop())
                    await ui_broadcast_partial("(приём аудио начат…)")
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    if recognition:
                        for _ in range(15):  # ~300 мс тишины
                            try: recognition.send_audio_frame(SILENCE_20MS)
                            except Exception: break
                    await stop_rec(send_notice="OK:STOPPED")

                elif raw.startswith("PROMPT:"):
                    # устройство само инициирует раунд: та же строгая семантика «сначала жёсткий сброс, потом воспроизведение»
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text) # используем собственную функцию запуска
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and recognition:
                    try:
                        recognition.send_audio_frame(msg["bytes"])
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("send_audio_frame failed")

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[WS] connection closed")

# ---------- WebSocket: вход камеры ESP32 (двоичный JPEG) ----------
@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected")
    
    # [Новое] инициализация навигатора по тактильной дорожке
    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        print("[NAVIGATION] Навигатор по тактильной дорожке инициализирован")
    else:
        if blind_path_navigator is not None:
            print("[NAVIGATION] Навигатор уже существует, повторная инициализация не нужна")
        elif yolo_seg_model is None:
            print("[NAVIGATION] Предупреждение: модель YOLO не загружена, инициализировать навигатор невозможно")
    
    # [Новое] инициализация навигатора перехода через дорогу
    if cross_street_navigator is None:
        if yolo_seg_model:
            cross_street_navigator = CrossStreetNavigator(
                seg_model=yolo_seg_model,
                coco_model=None,  # детекцию светофора не используем
                obs_model=None    # детекцию препятствий пока тоже не используем, чтобы работало быстрее
            )
            print("[CROSS_STREET] Навигатор перехода через дорогу инициализирован (упрощённая версия - только детекция зебры)")
        else:
            print("[CROSS_STREET] Ошибка: отсутствует модель сегментации, инициализировать навигатор перехода невозможно")
            
            if not yolo_seg_model:
                print("[CROSS_STREET] - отсутствует модель сегментации (yolo_seg_model)")
            if not obstacle_detector:
                print("[CROSS_STREET] - отсутствует детектор препятствий (obstacle_detector)")
    
    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[NAV MASTER] Оркестрирующий конечный автомат инициализирован (управляемый режим)")
    frame_counter = 0  # добавляем счётчик кадров
    
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frame_counter += 1
                
                # [Новое] запись исходного кадра
                try:
                    sync_recorder.record_frame(data)
                except Exception as e:
                    if frame_counter % 100 == 0:  # чтобы не засорять лог
                        print(f"[RECORDER] Ошибка записи кадра: {e}")
                
                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass
                
                # отправляем в bridge_io (для yolomedia)
                bridge_io.push_raw_jpeg(data)
                
                # [Отладка] проверяем условия навигации
                if frame_counter % 30 == 0:  # вывод раз в 30 кадров
                    state_dbg = orchestrator.get_state() if orchestrator else "N/A"
                    print(f"[NAVIGATION DEBUG] кадр:{frame_counter}, state={state_dbg}, yolomedia_running={yolomedia_running}")
                
                # единое декодирование (с более строгой обработкой исключений)
                try:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    # проверяем результат декодирования
                    if bgr is None or bgr.size == 0:
                        if frame_counter % 30 == 0:
                            print(f"[JPEG] Ошибка декодирования: длина данных={len(data)}")
                        bgr = None
                except Exception as e:
                    if frame_counter % 30 == 0:
                        print(f"[JPEG] Исключение при декодировании: {e}")
                    bgr = None

                # [Управление] в первую очередь передаём оркестрирующему автомату (когда изображение не занято поиском предмета)
                # [Изменено] в режиме поиска предмета навигационная обработка не выполняется, изображением управляет yolomedia
                if orchestrator and not yolomedia_running and bgr is not None:
                    current_state = orchestrator.get_state()
                    
                    # [Новое] режим поиска предмета: изображение не обрабатываем, ждём обработанные кадры от yolomedia
                    if current_state == "ITEM_SEARCH":
                        # в режиме поиска предмета, пока yolomedia не начал отправлять кадры, показываем исходное изображение
                        if not yolomedia_sending_frames and camera_viewers:
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok:
                                jpeg_data = enc.tobytes()
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for d in dead:
                                    camera_viewers.discard(d)
                        continue  # пропускаем дальнейшую навигационную обработку
                    
                    out_img = bgr
                    try:
                        # [Новое] проверяем, включён ли режим детекции светофора
                        if current_state == "TRAFFIC_LIGHT_DETECTION":
                            # режим детекции светофора: обрабатываем прямо в основном потоке, чтобы не терять кадры
                            import trafficlight_detection
                            result = trafficlight_detection.process_single_frame(bgr, ui_broadcast_callback=ui_broadcast_final)
                            out_img = result['vis_image'] if result['vis_image'] is not None else bgr
                        else:
                            # прочие режимы: обычная навигационная обработка
                            res = orchestrator.process_frame(bgr)

                            # голосовые подсказки (троттлинг уже внутри)
                            # примечание: во время omni-диалога режим уже переключён на CHAT, навигационные подсказки не генерируются
                            if res.guidance_text:
                                try:
                                    # сначала воспроизводим голос, затем рассылаем в UI
                                    play_voice_text(res.guidance_text)
                                    await ui_broadcast_final(f"[导航] {res.guidance_text}")
                                except Exception:
                                    pass

                            # вывод изображения
                            out_img = res.annotated_image if res.annotated_image is not None else bgr
                    except Exception as e:
                        if frame_counter % 100 == 0:
                            print(f"[NAV MASTER] Ошибка при обработке кадра: {e}")

                    # рассылка изображения
                    if camera_viewers and out_img is not None:
                        ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    # обработано оркестратором, переходим к следующему кадру
                    continue

                # [Запасной путь] изображение занято поиском или не декодировано — возвращаем исходное
                if not yolomedia_sending_frames and camera_viewers:
                    try:
                        if bgr is None:
                            arr = np.frombuffer(data, dtype=np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok:
                                jpeg_data = enc.tobytes()
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for ws in dead:
                                    camera_viewers.discard(ws)
                    except Exception as e:
                        print(f"[CAMERA] Broadcast error: {e}")

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        print("[CAMERA] ESP32 disconnected")
        
        # [Новое] очищаем состояние навигации
        if blind_path_navigator:
            blind_path_navigator.reset()
        if cross_street_navigator:
            cross_street_navigator.reset()
        if orchestrator:
            orchestrator.reset()
            print("[NAV MASTER] Оркестратор сброшен")

# ---------- WebSocket: подписка браузера на кадры камеры ----------
@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            # поддерживаем соединение активным
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try: 
            camera_viewers.remove(ws)
        except Exception: 
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)

# ---------- WebSocket: подписка браузера на IMU ----------
@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)

async def imu_broadcast(msg: str):
    if not imu_ws_clients: return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        imu_ws_clients.discard(ws)

# ---------- Серверная оценка IMU (оставлено как есть) ----------
from math import atan2, hypot, pi
GRAV_BETA   = 0.98
STILL_W     = 0.4
YAW_DB      = 0.08
YAW_LEAK    = 0.2
ANG_EMA     = 0.15
AUTO_REZERO = True
USE_PROJ    = True
FREEZE_STILL= True
G     = 9.807
A_TOL = 0.08 * G
gLP = {"x":0.0, "y":0.0, "z":0.0}
gOff= {"x":0.0, "y":0.0, "z":0.0}
BIAS_ALPHA = 0.002
yaw  = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll":0.0, "pitch":0.0, "yaw":0.0}
holdStart = 0.0
isStill   = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store: List[Dict[str, Any]] = []

def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0: a -= 360.0
    if a < -180.0: a += 360.0
    return a

def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall

    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0:
        t_ms = (now_wall * 1000.0)
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0:
        dt = 0.02
    else:
        dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms

    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro")  or {}).get("x", 0.0)))
    wy = float(((d.get("gyro")  or {}).get("y", 0.0)))
    wz = float(((d.get("gyro")  or {}).get("z", 0.0)))

    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"]/gmag, "y": gLP["y"]/gmag, "z": gLP["z"]/gmag}

    roll  = (atan2(az, ay)   * 180.0 / pi)
    pitch = (atan2(-ax, ay)  * 180.0 / pi)

    aNorm = hypot(ax, ay, az); wNorm = hypot(wx, wy, wz)
    nearFlat = (abs(roll) < 2.0 and abs(pitch) < 2.0)
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)

    if stillCond:
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0: isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA)*gOff["x"] + BIAS_ALPHA*wx
        gOff["y"] = (1.0 - BIAS_ALPHA)*gOff["y"] + BIAS_ALPHA*wy
        gOff["z"] = (1.0 - BIAS_ALPHA)*gOff["z"] + BIAS_ALPHA*wz
    else:
        holdStart = 0.0; isStill = False

    if USE_PROJ:
        yawdot = ((wx - gOff["x"])*gHat["x"] + (wy - gOff["y"])*gHat["y"] + (wz - gOff["z"])*gHat["z"])
    else:
        yawdot = (wy - gOff["y"])

    if abs(yawdot) < YAW_DB: yawdot = 0.0
    if FREEZE_STILL and stillCond: yawdot = 0.0

    yaw = _wrap180(yaw + yawdot * dt)

    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step): yaw = 0.0
        else: yaw += step

    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll  + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw   + (1.0 - ANG_EMA) * Yf

    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True

    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])

    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {
            "ts": t_ms/1000.0,
            "angles": {"roll": R, "pitch": P, "yaw": Y},
            "accel":  {"x": ax, "y": ay, "z": az},
            "gyro":   {"x": wx, "y": wy, "z": wz},
        }
        imu_store.append(item)

# ---------- Приём IMU по UDP и пересылка ----------
class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")
    def datagram_received(self, data, addr):
        try:
            s = data.decode('utf-8', errors='ignore').strip()
            d = json.loads(s)
            if 'ts' not in d and 'timestamp_ms' in d:
                d['ts'] = d.pop('timestamp_ms')
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass



# === Новое: колбэк отправки, регистрируемый в bridge_io (рассылает JPEG в /ws/viewer) ===
@app.on_event("startup")
async def on_startup_register_bridge_sender():
    # сохраняем цикл событий основного потока
    main_loop = asyncio.get_event_loop()
    
    def _sender(jpeg_bytes: bytes):
        # внимание: функция может вызываться из потока без корутин, нужно переключиться на основной цикл событий
        try:
            # проверяем состояние цикла событий, чтобы не отправлять во время закрытия
            if main_loop.is_closed():
                return
            
            # отмечаем, что YOLO начал отправлять обработанные кадры
            global yolomedia_sending_frames
            if not yolomedia_sending_frames:
                yolomedia_sending_frames = True
                print("[YOLOMEDIA] Начата отправка обработанных кадров, переключаемся на изображение YOLO", flush=True)
            
            async def _broadcast():
                if not camera_viewers:
                    return
                dead = []
                for ws in list(camera_viewers):
                    try:
                        await ws.send_bytes(jpeg_bytes)
                    except Exception as e:
                        dead.append(ws)
                for ws in dead:
                    try:
                        camera_viewers.remove(ws)
                    except Exception:
                        pass
            
            # используем сохранённый цикл событий основного потока
            future = asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
            # не ждём результат, чтобы не блокировать поток-производитель
        except Exception as e:
            # логируем только неожиданные ошибки
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)

    bridge_io.set_sender(_sender)

@app.on_event("startup")
async def on_startup_init_audio():
    """Инициализация аудиосистемы при запуске"""
    # инициализируем в фоновом потоке, чтобы не блокировать запуск
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] Ошибка инициализации: {e}")
    
    threading.Thread(target=_init, daemon=True).start()

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))

@app.on_event("shutdown")
async def on_shutdown():
    """Очистка при завершении приложения"""
    print("[SHUTDOWN] Начинаем освобождение ресурсов...")
    
    # останавливаем медиаобработку YOLO
    stop_yolomedia()
    
    # останавливаем аудио и задачи AI
    await hard_reset_audio("shutdown")
    
    print("[SHUTDOWN] Освобождение ресурсов завершено")

# app_main.py —— после уже имеющегося в файле @app.on_event("startup") добавляем ещё один startup-хук


# --- Экспортируемый интерфейс (опционально) ---
def get_last_frames():
    return last_frames

def get_camera_ws():
    return esp32_camera_ws

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8081,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False
    )
