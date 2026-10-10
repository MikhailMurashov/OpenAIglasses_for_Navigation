# bridge_io.py
# Минималистичный мост: принимает сырой JPEG → отдаёт BGR-кадры внешним алгоритмам; внешние алгоритмы выдают BGR → рассылка во фронтенд
import threading
from collections import deque
import time
import cv2
import numpy as np

# Буфер сырых JPEG-кадров (хранятся только последние N кадров)
_MAX_BUF = 4
_frames = deque(maxlen=_MAX_BUF)
_cond = threading.Condition()

# Колбэк отправки JPEG во фронтенд, регистрируется app_main.py при запуске
_sender_lock = threading.Lock()
_sender_cb = None

# Колбэк отправки текста UI во фронтенд (регистрируется app_main.py при запуске)
_ui_sender_lock = threading.Lock()
_ui_sender_cb = None

def set_sender(cb):
    """Вызывается из app_main.py, регистрирует функцию: cb(jpeg_bytes)->None"""
    global _sender_cb
    with _sender_lock:
        _sender_cb = cb

def set_ui_sender(cb):
    """Вызывается из app_main.py, регистрирует функцию: cb(text:str)->None"""
    global _ui_sender_cb
    with _ui_sender_lock:
        _ui_sender_cb = cb

def push_raw_jpeg(jpeg_bytes: bytes):
    """Вызывается из app_main.py при получении кадра из /ws/camera"""
    if not jpeg_bytes:
        return
    with _cond:
        _frames.append((time.time(), jpeg_bytes))
        _cond.notify_all()

def wait_raw_bgr(timeout_sec: float = 0.5):
    """Вызывается скриптами YOLO/MediaPipe: ждёт и получает последний BGR-кадр; при таймауте возвращает None"""
    t_end = time.time() + timeout_sec
    last = None
    while time.time() < t_end:
        with _cond:
            if _frames:
                last = _frames[-1]
        if last is None:
            time.sleep(0.01)
            continue
        # Декодируем JPEG в BGR
        ts, jpeg = last
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            # Зеркалирование выполняется в самом источнике
            #bgr = cv2.flip(bgr, 1)
            return bgr
        # Ошибка декодирования, ждём немного и повторяем
        time.sleep(0.01)
    return None

def send_vis_bgr(bgr, quality: int = 80):
    """Вызывается скриптами YOLO/MediaPipe: отправляет обработанный кадр во фронтенд-viewer"""
    if bgr is None:
        return
    
    # Кодируем напрямую, без какой-либо дополнительной обработки
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return
    with _sender_lock:
        cb = _sender_cb
    if cb:
        try:
            cb(enc.tobytes())
        except Exception:
            pass

def send_ui_final(text: str):
    """Отправляет строку текста UI во фронтенд как final answer (потокобезопасный колбэк)"""
    if not text:
        return
    with _ui_sender_lock:
        cb = _ui_sender_cb
    if cb:
        try:
            cb(str(text))
        except Exception:
            pass
