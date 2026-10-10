# -*- coding: utf-8 -*-
"""
Модуль детекции светофора — версия как отдельный сценарий
В реальном времени определяет состояние светофора моделью YOLO и сообщает голосом
Управляется голосовыми командами "检测红绿灯" (начать детекцию светофора) и "停止检测" (остановить детекцию)
"""

import os
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import bridge_io
from audio_player import play_voice_text  # Используем единый интерфейс воспроизведения речи
import logging

logger = logging.getLogger(__name__)

# ========= Параметры конфигурации =========
YOLO_MODEL_PATH = r"C:\Users\Administrator\Desktop\rebuild1002\model\trafficlight.pt"

# ========= Параметры отображения =========
CONF_THRESHOLD = 0.25  # Порог уверенности
FONT_SIZE = 20
STROKE_WIDTH = 3

# ========= Параметры голосовых подсказок =========
TTS_INTERVAL_SEC = 2.0  # Интервал голосовых подсказок (чтобы не озвучивать слишком часто)
ENABLE_TTS = False  # [Отключено] модуль детекции светофора не озвучивает, озвучкой централизованно занимается workflow_crossstreet.py

# ========= Управление потоками =========
_detection_thread = None
_stop_event = None
_detection_running = False

# ========= Режим покадровой обработки (новое) =========
_model = None  # Глобальный экземпляр модели
_last_tts_ts = 0.0
_last_detected_light = None
_detection_history = []

# ========= Цвета фронтенда (BGR) =========
FRONTEND_COLORS = {
    "text": (230, 237, 243),   # белый текст
    "red": (0, 0, 255),        # красный
    "yellow": (0, 255, 255),   # жёлтый
    "green": (0, 255, 0),      # зелёный
    "muted": (159, 176, 195),  # серый
}

# Сопоставление состояний светофора с цветами
LIGHT_COLORS = {
    "stop": FRONTEND_COLORS["red"],
    "countdown_go": FRONTEND_COLORS["yellow"],
    "go": FRONTEND_COLORS["green"],
}

# [Исправлено] сопоставление состояний светофора с китайскими названиями
# Только реальные классы светофора, без зебры (crossing) и пустых
LIGHT_NAMES = {
    "stop": "红灯",              # красный для автомобилей
    "go": "绿灯",                # зелёный для автомобилей
    "countdown_go": "黄灯",      # обратный отсчёт зелёного (сообщаем как жёлтый)
    "countdown_stop": "红灯",    # обратный отсчёт красного
}

# Сопоставление состояний светофора с голосовыми файлами
LIGHT_VOICE_MAP = {
    "stop": "红灯",              # → voice/红灯.WAV
    "go": "绿灯",                # → voice/绿灯.WAV
    "countdown_go": "黄灯",      # → voice/黄灯.WAV (обратный отсчёт зелёного сообщаем как жёлтый)
    "countdown_stop": "红灯",    # → voice/红灯.WAV
}

# Классы, которые отфильтровываются (не детектируются и не отображаются)
FILTERED_CLASSES = {
    "crossing",          # зебра (не нужна)
    "blank",            # пусто
    "countdown_blank"   # пустой обратный отсчёт
}

# Управление текстом UI
_UI_LINE = 0
_UI_H = 0
_UI_TR_LINE = 0
_UI_TOP_MARGIN = 12
_UI_RIGHT_MARGIN = 12
UNIFIED_FONT_PX = 12

def ui_reset_overlay(img_h: int):
    """Вызывается раз за кадр, сбрасывает счётчик строк наложения"""
    global _UI_LINE, _UI_H, _UI_TR_LINE
    _UI_LINE = 0
    _UI_TR_LINE = 0
    _UI_H = int(img_h)

def _ui_next_y_top(font_size: int) -> int:
    """Возвращает координату y следующей строки в правом верхнем углу"""
    global _UI_TR_LINE
    line_gap = max(4, int(font_size * 0.25))
    y_top = _UI_TOP_MARGIN + (_UI_TR_LINE * (font_size + line_gap))
    _UI_TR_LINE += 1
    return y_top

# ======== Отрисовка китайского текста ========
_PIL_OK = False
_FONT_PATH = None

def _init_font():
    global _PIL_OK, _FONT_PATH
    try:
        from PIL import ImageFont
        _PIL_OK = True
    except Exception:
        _PIL_OK = False
        return
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyh.ttf",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simfang.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        r"C:\\Windows\\Fonts\\simsunb.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            _FONT_PATH = p
            return
    _PIL_OK = False

_init_font()

def draw_text_cn(img_bgr, text, xy, font_size=20, color=(255,255,255), ui_hint=True):
    """Единая отрисовка китайского текста"""
    color = (255, 255, 255)
    font_size = int(UNIFIED_FONT_PX)

    H, W = img_bgr.shape[:2]
    y_top = _ui_next_y_top(font_size) if ui_hint else xy[1]
    tw = th = 0
    font_obj = None

    if _PIL_OK and _FONT_PATH:
        try:
            from PIL import Image, ImageDraw, ImageFont
            font_obj = ImageFont.truetype(_FONT_PATH, font_size)
            bbox = ImageDraw.Draw(Image.new('RGB', (1,1))).textbbox((0,0), text, font=font_obj)
            tw = max(1, bbox[2] - bbox[0])
            th = max(1, bbox[3] - bbox[1])
        except Exception:
            pass
    
    if _PIL_OK and _FONT_PATH and font_obj is not None:
        try:
            from PIL import Image, ImageDraw
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            if ui_hint:
                x = max(8, W - _UI_RIGHT_MARGIN - tw)
                y = y_top
            else:
                x = xy[0]
                y = xy[1]
            draw.text((x, y), text, fill=color, font=font_obj)
            img_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    
    # Запасной вариант через OpenCV
    if tw <= 0 or th <= 0:
        scale = font_size/24.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    if ui_hint:
        x = max(8, W - _UI_RIGHT_MARGIN - int(tw))
        y_baseline = int(y_top + th)
    else:
        x = xy[0]
        y_baseline = xy[1] + int(th)
    cv2.putText(img_bgr, text, (x, y_baseline), cv2.FONT_HERSHEY_SIMPLEX, font_size/24.0, color, 2, cv2.LINE_AA)

def main(headless: bool = True, stop_event=None):
    """
    Главная функция детекции светофора
    
    Параметры:
        headless: режим без интерфейса (не показывать окно OpenCV)
        stop_event: threading.Event для остановки детекции
    """
    
    print("[TRAFFIC] Загрузка модели YOLO для детекции светофора...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"[TRAFFIC] Модель успешно загружена: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[TRAFFIC] Ошибка загрузки модели: {e}")
        return

    # Получение названий классов
    class_names = model.names if hasattr(model, 'names') else {}
    print(f"[TRAFFIC] Классы модели: {class_names}")

    # Отслеживание состояния
    last_tts_ts = 0.0
    last_detected_light = None
    fps_hist = []
    
    # [Оптимизация] оценка стабильности состояния — голосование большинством вместо подряд идущих кадров
    detection_history = []  # результаты детекции за последние N кадров
    HISTORY_SIZE = 5        # храним последние 5 кадров
    MAJORITY_THRESHOLD = 3  # стабильно, если совпадают минимум 3 из 5 кадров
    
    # [Новое] статистика кадров
    frame_count = 0
    frame_received_count = 0
    frame_none_count = 0
    last_frame_log_time = time.time()

    print("[TRAFFIC] Ожидание изображения с ESP32...")

    try:
        while True:
            # Проверка события остановки
            if stop_event and stop_event.is_set():
                print("[TRAFFIC] Сработало событие остановки, выход из детекции")
                break

            # [Оптимизация] получение исходного BGR-кадра из bridge_io — увеличен таймаут
            frame = bridge_io.wait_raw_bgr(timeout_sec=2.0)  # увеличено с 0,5 до 2 секунд
            
            frame_count += 1
            
            if frame is None:
                frame_none_count += 1
                # Печать статистики кадров раз в 3 секунды
                current_time = time.time()
                if current_time - last_frame_log_time > 3.0:
                    print(f"[TRAFFIC] Статистика кадров: всего={frame_count}, получено={frame_received_count}, "
                          f"потеряно={frame_none_count}, доля потерь={frame_none_count/frame_count*100:.1f}%")
                    last_frame_log_time = current_time
                
                if headless:
                    cv2.waitKey(1)
                continue
            
            frame_received_count += 1

            # Сброс наложения UI
            H, W = frame.shape[:2]
            ui_reset_overlay(H)

            vis = frame.copy()
            t_now = time.time()

            # [Оптимизация] инференс YOLO — добавлен замер времени
            inference_start = time.time()
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            inference_time = (time.time() - inference_start) * 1000
            
            # Контроль времени инференса
            if inference_time > 100:
                print(f"[TRAFFIC] WARNING: инференс занял {inference_time:.0f}ms")

            # Обработка результатов детекции
            detected_light = None
            max_conf = 0.0

            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    # [Фильтр] перебираем все рамки и находим светофор с наибольшей уверенностью (без зебры)
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = class_names.get(cls_id, f"class_{cls_id}")
                        class_name_lower = class_name.lower()
                        
                        # Пропускаем ненужные классы
                        if class_name_lower in FILTERED_CLASSES:
                            continue
                        
                        if conf > max_conf:
                            max_conf = conf
                            detected_light = class_name_lower

                    # [Фильтр] отрисовка рамок (только светофоры)
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = class_names.get(cls_id, f"class_{cls_id}")
                        class_name_lower = class_name.lower()
                        
                        # Пропускаем ненужные классы
                        if class_name_lower in FILTERED_CLASSES:
                            continue
                        
                        # Получение координат рамки
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Определение цвета
                        color = LIGHT_COLORS.get(class_name_lower, FRONTEND_COLORS["text"])
                        
                        # Отрисовка рамки
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, STROKE_WIDTH)
                        
                        # Отрисовка китайской подписи (через PIL)
                        label = f"{LIGHT_NAMES.get(class_name.lower(), class_name)}: {conf:.2f}"
                        
                        if _PIL_OK and _FONT_PATH:
                            try:
                                from PIL import Image, ImageDraw, ImageFont
                                # Подпись рисуется более крупным шрифтом
                                font_obj = ImageFont.truetype(_FONT_PATH, 20)
                                # Преобразование в изображение PIL
                                img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img_rgb)
                                draw = ImageDraw.Draw(pil_img)
                                
                                # Расчёт размера текста
                                bbox = draw.textbbox((0, 0), label, font=font_obj)
                                text_w = bbox[2] - bbox[0]
                                text_h = bbox[3] - bbox[1]
                                
                                # Положение подписи
                                label_y = max(y1 - text_h - 8, text_h)
                                
                                # Отрисовка фонового прямоугольника
                                bg_x1 = x1
                                bg_y1 = label_y - text_h - 4
                                bg_x2 = x1 + text_w + 8
                                bg_y2 = label_y + 4
                                cv2.rectangle(vis, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                                
                                # Повторное преобразование (прямоугольник нарисован через OpenCV)
                                img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img_rgb)
                                draw = ImageDraw.Draw(pil_img)
                                
                                # [Удалено] отрисовка текста
                                # draw.text((x1 + 4, label_y - text_h), label, fill=(0, 0, 0), font=font_obj)
                                
                                # Обратное преобразование в формат OpenCV
                                vis[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
                            except Exception as e:
                                # [Удалено] текстовая подпись при сбое PIL
                                pass
                        else:
                            # [Удалено] текстовая подпись
                            pass

            # [Оптимизация] оценка стабильности состояния: голосование большинством вместо подряд идущих кадров
            detection_history.append(detected_light)
            if len(detection_history) > HISTORY_SIZE:
                detection_history.pop(0)
            
            # Проверка стабильности состояния (голосование большинством)
            stable_light = None
            if len(detection_history) >= MAJORITY_THRESHOLD:
                # Подсчёт количества появлений каждого состояния за последние N кадров
                valid_detections = [d for d in detection_history if d and d in LIGHT_NAMES]
                if len(valid_detections) >= MAJORITY_THRESHOLD:
                    # Поиск самого частого состояния
                    from collections import Counter
                    counter = Counter(valid_detections)
                    most_common = counter.most_common(1)
                    if most_common and most_common[0][1] >= MAJORITY_THRESHOLD:
                        stable_light = most_common[0][0]
                        # Печать отладочной информации
                        if frame_received_count % 30 == 0:
                            print(f"[TRAFFIC] История детекции: {detection_history[-5:]}, стабильное состояние: {stable_light}")
            
            # [Голосовые подсказки отключены] только детекция без озвучки; озвучкой централизованно занимается вызывающий код (workflow_crossstreet.py)
            # Только обновляем отслеживание состояния
            if stable_light:
                # При смене состояния записываем (но не озвучиваем)
                if stable_light != last_detected_light:
                    last_detected_light = stable_light
                    print(f"[TRAFFIC] Обнаружена смена стабильного состояния: {LIGHT_NAMES[stable_light]} (без озвучки)")
                    last_tts_ts = t_now
                # Интервал превышен — обновляем метку времени (но не озвучиваем)
                elif (t_now - last_tts_ts) > TTS_INTERVAL_SEC:
                    print(f"[TRAFFIC] Стабильное состояние сохраняется: {LIGHT_NAMES[stable_light]} (без озвучки)")
                    last_tts_ts = t_now

            # [Удалено] отображение текущего состояния детекции
            # if detected_light and detected_light in LIGHT_NAMES:
            #     status_text = f"检测: {LIGHT_NAMES[detected_light]} ({max_conf:.2f})"
            #     color = LIGHT_COLORS[detected_light]
            # else:
            #     status_text = "检测: 无"
            #     color = FRONTEND_COLORS["muted"]
            # draw_text_cn(vis, status_text, (10, 40), font_size=18, color=color)
            
            # [Удалено] отображение стабильного состояния
            # if stable_light:
            #     stable_text = f"稳定状态: {LIGHT_NAMES[stable_light]}"
            #     stable_color = LIGHT_COLORS[stable_light]
            # else:
            #     stable_text = f"稳定状态: 等待中 ({len(detection_history)}/{HISTORY_SIZE})"
            #     stable_color = FRONTEND_COLORS["muted"]
            # draw_text_cn(vis, stable_text, (10, 60), font_size=18, color=stable_color)

            # [Удалено] расчёт и отображение FPS
            # fps_hist.append(t_now)
            # if len(fps_hist) > 30:
            #     fps_hist.pop(0)
            # fps = 0.0 if len(fps_hist) < 2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
            # draw_text_cn(vis, f"FPS: {fps:.1f}", (10, 20), font_size=16, color=FRONTEND_COLORS["text"])

            # Отправка визуализации на фронтенд
            bridge_io.send_vis_bgr(vis)

            # В режиме не-headless показываем окно
            if not headless:
                cv2.imshow("Traffic Light Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
                cv2.waitKey(1)

    except Exception as e:
        print(f"[TRAFFIC] Ошибка в процессе детекции: {e}")
    finally:
        if not headless:
            cv2.destroyAllWindows()
        print("[TRAFFIC] Детекция светофора остановлена")


def start_detection():
    """Запускает детекцию светофора (в фоновом потоке)"""
    global _detection_thread, _stop_event, _detection_running
    
    if _detection_running:
        print("[TRAFFIC] Детекция светофора уже запущена")
        return False
    
    _stop_event = threading.Event()
    _detection_thread = threading.Thread(
        target=main,
        args=(True, _stop_event),  # headless=True, stop_event
        daemon=True,
        name="TrafficLightDetection"
    )
    _detection_thread.start()
    _detection_running = True
    print("[TRAFFIC] Детекция светофора запущена (фоновый поток)")
    return True

def stop_detection():
    """Останавливает детекцию светофора"""
    global _detection_thread, _stop_event, _detection_running
    
    if not _detection_running:
        print("[TRAFFIC] Детекция светофора не запущена")
        return False
    
    print("[TRAFFIC] Остановка детекции светофора...")
    if _stop_event:
        _stop_event.set()
    
    if _detection_thread:
        _detection_thread.join(timeout=2.0)
        _detection_thread = None
    
    _stop_event = None
    _detection_running = False
    print("[TRAFFIC] Детекция светофора остановлена")
    return True

def is_detection_running():
    """Проверяет, запущена ли детекция светофора"""
    return _detection_running

def init_model():
    """Инициализирует модель YOLO (режим покадровой обработки)"""
    global _model
    if _model is not None:
        print("[TRAFFIC] Модель уже загружена")
        return True
    
    try:
        print("[TRAFFIC] Загрузка модели YOLO для детекции светофора...")
        _model = YOLO(YOLO_MODEL_PATH)
        print(f"[TRAFFIC] Модель успешно загружена: {YOLO_MODEL_PATH}")
        class_names = _model.names if hasattr(_model, 'names') else {}
        print(f"[TRAFFIC] Классы модели: {class_names}")
        return True
    except Exception as e:
        print(f"[TRAFFIC] Ошибка загрузки модели: {e}")
        _model = None
        return False

def process_single_frame(image: np.ndarray, ui_broadcast_callback=None) -> dict:
    """
    Обрабатывает один кадр (режим основного потока, без пропуска кадров)
    Параметры:
        image: входное изображение
        ui_broadcast_callback: колбэк рассылки на фронтенд (для отображения состояния светофора)
    Возвращает: {'vis_image': изображение визуализации, 'detected_light': обнаруженный сигнал, 'stable_light': стабильное состояние}
    """
    global _model, _last_tts_ts, _last_detected_light, _detection_history
    
    if _model is None:
        if not init_model():
            return {'vis_image': image, 'detected_light': None, 'stable_light': None}
    
    vis = image.copy()
    t_now = time.time()
    
    # Инференс YOLO
    results = _model(image, conf=CONF_THRESHOLD, verbose=False)
    
    # Обработка результатов детекции
    detected_light = None
    max_conf = 0.0
    class_names = _model.names if hasattr(_model, 'names') else {}
    
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            # Перебираем все рамки и находим светофор с наибольшей уверенностью (crossing и т.п. отфильтровываются)
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names.get(cls_id, f"class_{cls_id}")
                class_name_lower = class_name.lower()
                
                # [Фильтр] пропускаем ненужные классы (зебра, пустые и т.п.)
                if class_name_lower in FILTERED_CLASSES:
                    continue
                
                if conf > max_conf:
                    max_conf = conf
                    detected_light = class_name_lower
            
            # Отрисовка рамок (только светофоры, без зебры)
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names.get(cls_id, f"class_{cls_id}")
                class_name_lower = class_name.lower()
                
                # [Фильтр] пропускаем ненужные классы
                if class_name_lower in FILTERED_CLASSES:
                    continue
                
                # Получение координат рамки
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Определение цвета
                color = LIGHT_COLORS.get(class_name_lower, FRONTEND_COLORS["text"])
                
                # Отрисовка рамки
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, STROKE_WIDTH)
    
    # [Ослаблено] оценка стабильности состояния (голосование большинством) — требования снижены
    _detection_history.append(detected_light)
    if len(_detection_history) > 5:
        _detection_history.pop(0)
    
    stable_light = None
    if len(_detection_history) >= 2:  # снижено с 3 до 2 кадров
        from collections import Counter
        valid_detections = [d for d in _detection_history if d and d in LIGHT_NAMES]
        if len(valid_detections) >= 2:  # снижено с 3 до 2 кадров
            counter = Counter(valid_detections)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:  # снижено с 3 до 2 раз
                stable_light = most_common[0][0]
    
    # [Отладка] печать результатов детекции (отключено)
    # print(f"[TRAFFIC-DEBUG] detected={detected_light}, stable={stable_light}, history={_detection_history}")
    
    # [Голосовые подсказки отключены] только детекция без озвучки; озвучкой централизованно занимается workflow_crossstreet.py
    # Только обновляем отслеживание состояния, play_voice_text не вызываем
    if stable_light:
        # Обновление отслеживания состояния (для обнаружения смены состояния)
        if stable_light != _last_detected_light:
            _last_detected_light = stable_light
            print(f"[TRAFFIC] Обнаружена смена стабильного состояния: {LIGHT_NAMES[stable_light]} (без озвучки)")
            _last_tts_ts = t_now
        elif (t_now - _last_tts_ts) > TTS_INTERVAL_SEC:
            # Интервал превышен — обновляем метку времени (но не озвучиваем)
            print(f"[TRAFFIC] Стабильное состояние сохраняется: {LIGHT_NAMES[stable_light]} (без озвучки)")
            _last_tts_ts = t_now
    
    # [Удалено] отображение текста состояния
    # if detected_light and detected_light in LIGHT_NAMES:
    #     status_text = f"{LIGHT_NAMES[detected_light]} ({max_conf:.2f})"
    # else:
    #     status_text = "无检测"
    # 
    # if stable_light:
    #     stable_text = f"稳定: {LIGHT_NAMES[stable_light]}"
    # else:
    #     stable_text = f"等待稳定 ({len(_detection_history)}/5)"
    # 
    # # Добавление простого отображения текста
    # cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # cv2.putText(vis, stable_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return {
        'vis_image': vis,
        'detected_light': detected_light,
        'stable_light': stable_light
    }

def reset_detection_state():
    """Сбрасывает состояние детекции"""
    global _last_tts_ts, _last_detected_light, _detection_history
    _last_tts_ts = 0.0
    _last_detected_light = None
    _detection_history = []
    print("[TRAFFIC] Состояние детекции сброшено")

if __name__ == "__main__":
    main(headless=False)



