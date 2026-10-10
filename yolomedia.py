# -*- coding: utf-8 -*-
"""
YOLOv8 сегментация одного класса + MediaPipe Hand Landmarker + трекинг оптическим потоком (многоугольник)
Что нового (главное в этой версии):
- Вторая полоса прогресса в левом нижнем углу «расстояние (≈1)» полностью заменена визуализацией «близости к 1» для ratio = площадь объекта / площадь руки
  -> range_score = 1 - clamp(|ratio - 1| / RATIO_TOL, 0..1)
  -> на кадре также выводится значение ratio; при ratio<1 подсказка «вперёд ближе», при ratio>1 — «назад», в пределах [1±RATIO_TOL] — «держать»
Прочие возможности:
- Захват по Enter: точки оптического потока берутся на внутренней границе маски сегментации, «сжатой на 5px»
- В режиме TRACK: отслеживается сегментация в окрестности текущего многоугольника, расширенного на 40px; при попадании — повторный захват
- Критерий успеха: смягчённая эвристика «хвата (Grasp)» (бутылку не нужно сжимать особенно крепко)
- Одноцветная отрисовка скелета руки; стрелка измерения (засечки на концах + стрелка + значение в пикселях)
- Текст с иероглифами рисуется преимущественно через Pillow + системный шрифт с поддержкой CJK (чтобы не было знаков вопроса)
"""

import os
import time
import threading
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
import bridge_io
import pygame  # для воспроизведения локальных аудиофайлов

from audio_player import play_audio_threadsafe
PERF_DEBUG = False        # печать отладочной информации (False — выключено)
HAND_DOWNSCALE = 0.8      # масштаб входа HandLandmarker: 0.5 = ширина и высота вдвое меньше (≈1/4 пикселей)
HAND_FPS_DIV = 1          # как часто запускать детекцию руки (1 = каждый кадр; 2 = через кадр; 3 = каждый 3-й кадр)


# === Цвета в стиле фронтенда (BGR) + управление UI-оверлеем (построчная укладка в левом нижнем углу) ===
FRONTEND_COLORS = {
    "text": (230, 237, 243),   # --text: #e6edf3
    "muted": (159, 176, 195),  # --muted: #9fb0c3
    "ok": (126, 231, 135),     # --ok: #7ee787
    "err": (128, 128, 255),    # --err: #ff8080 (BGR)
    "accent": (251, 218, 97),  # акцентный цвет, близкий к #61dafb (в BGR взят близкий яркий оттенок)
}

# Текст кнопки команды внизу
CURRENT_COMMAND_TEXT = "—"

_UI_LINE = 0
_UI_H = 0
_UI_TR_LINE = 0  # счётчик строк, укладываемых в правом верхнем углу
_UI_TOP_MARGIN = 12
_UI_RIGHT_MARGIN = 12
UNIFIED_FONT_PX = 12  # единый размер шрифта


def ui_reset_overlay(img_h: int):
    """Вызывается раз в кадр: сбрасывает счётчик строк оверлея (раскладка перенесена в правый верхний угол)."""
    global _UI_LINE, _UI_H, _UI_TR_LINE
    _UI_LINE = 0
    _UI_TR_LINE = 0
    _UI_H = int(img_h)


def _ui_next_y_top(font_size: int) -> int:
    """Возвращает y следующей строки в правом верхнем углу (выравнивание по верху) и увеличивает счётчик строк."""
    global _UI_TR_LINE
    line_gap = max(4, int(font_size * 0.25))
    y_top = _UI_TOP_MARGIN + (_UI_TR_LINE * (font_size + line_gap))
    _UI_TR_LINE += 1
    return y_top


def set_current_command(text: str):
    global CURRENT_COMMAND_TEXT
    try:
        CURRENT_COMMAND_TEXT = str(text) if text else "—"
    except Exception:
        CURRENT_COMMAND_TEXT = "—"


def draw_command_pill(img_bgr: np.ndarray, label: str):
    """Теперь выводится белым текстом в правом верхнем углу. Скруглённая кнопка внизу больше не рисуется."""
    text_prefix = "Текущая команда: "
    full_text = f"{text_prefix}{label if label else '—'}"
    # рисуем напрямую единым механизмом вывода текста
    draw_text_cn(img_bgr, full_text, (0, 0), font_size=UNIFIED_FONT_PX, color=(255,255,255), ui_hint=True)

try:
    from yoloe_backend import YoloEBackend
    _YOLOE_READY = True
except Exception as e:
    _YOLOE_READY = False
    print(f"[DETECTOR] YOLOE backend not ready: {e}", flush=True)

# ========= Пути (изменить при необходимости) =========
YOLO_MODEL_PATH = r'C:\Users\Administrator\Desktop\rebuild1002\model\shoppingbest5.pt'
HAND_TASK_PATH  = r"C:\Users\Administrator\Desktop\rebuild1002\model\hand_landmarker.task"

# ========= Камера =========
CAM_INDEX = 0
INPUT_W, INPUT_H = 600, 480

# ========= Отображение сегментации =========
STROKE_WIDTH = 5  # увеличенная толщина обводки, чтобы жёлтый и зелёный контуры были толще
MASK_ALPHA   = 0.45
CONF_THRESHOLD = 0.20

# —— Распознавание по одному prompt (показывается только один класс) ——
PROMPT_NAME   = "AD_milk"
PROMPT_STRICT = True

# ========= Параметры полосы выравнивания =========
ALIGN_LOOSE_PCT      = 0.12   # порог нормированного расстояния (относительно диагонали кадра)

# ========= Параметры полосы расстояния (в этой версии цель — «ratio≈1») =========
RATIO_IDEAL          = 1.0    # идеальное значение: площадь объекта / площадь руки ≈ 1
RATIO_TOL            = 0.25   # допустимое отклонение: в пределах ±25% расстояние считается подходящим

# ========= Голосовые подсказки =========
TTS_INTERVAL_SEC     = 1.0
ENABLE_TTS           = True

# ========= Оптический поток (LK) и особые точки =========
LK_PARAMS = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))
FEATURE_PARAMS = dict(maxCorners=600,
                      qualityLevel=0.001,
                      minDistance=5,
                      blockSize=7)

# ========= Ключевые параметры: сжатие внутрь и мониторинг окрестности =========
INNER_OFFSET_PX_LOCK = 5     # захват по Enter: эрозия маски в пикселях, чтобы точки были внутри объекта
EDGE_DILATE_PX       = 2     # небольшое расширение после взятия внутренней границы, чтобы легче находить точки
PERI_MONITOR_PX      = 40    # TRACK: отслеживать полосу вокруг многоугольника, расширенного на 40px
PERI_CHECK_EVERY     = 5     # проверка сегментации в окрестности раз в N кадров (изменено на каждый кадр)

# ========= Параметры точности контура =========
CONTOUR_EPSILON_FACTOR = 0.002  # коэффициент точности алгоритма Дугласа — Пекера: чем меньше, тем детальнее
TRACK_EPSILON_FACTOR = 0.003    # коэффициент точности контура в режиме трекинга

# ========= Параметры коррекции YOLO в реальном времени =========
YOLO_CORRECTION_IOU_THRESHOLD = 0.2  # порог IoU: чем ниже, тем активнее коррекция
YOLO_CORRECTION_CONF_THRESHOLD = 0.15  # порог уверенности: чем ниже, тем чувствительнее детекция

# ========= Пути к аудио подсказок направления =========
AUDIO_DIR = r"E:\沙粒云\自媒体\2025视频制作\20250925AI眼镜\AI眼镜合并\audio"  # замените на фактический путь
AUDIO_FILES = {
    "向上": os.path.join(AUDIO_DIR, "up.wav"),
    "向下": os.path.join(AUDIO_DIR, "down.wav"),
    "向左": os.path.join(AUDIO_DIR, "left.wav"),
    "向右": os.path.join(AUDIO_DIR, "right.wav"),
    "向前": os.path.join(AUDIO_DIR, "forward.wav"),
    "后退": os.path.join(AUDIO_DIR, "backward.wav"),
    "OK": os.path.join(AUDIO_DIR, "ok.wav"),  # добавлен звук OK
}
GUIDANCE_INTERVAL_SEC = 1.5  # интервал голосовых подсказок

# инициализация аудио pygame
pygame.mixer.init()

# ========= Окно =========
WINDOW = "YOLO Seg + Flow Polygon (Peri-Relock) (Grab Guidance)"

# ======== Псевдонимы MediaPipe ========
BaseOptions           = mp.tasks.BaseOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HAND_CONNECTIONS      = mp.solutions.hands.HAND_CONNECTIONS

# ======== Кэш колбэка HandLandmarker ========
_last_result = None  # (result, timestamp_ms)

def on_result(result: mp.tasks.vision.HandLandmarkerResult,
              output_image: mp.Image, timestamp_ms: int):
    global _last_result
    _last_result = (result, timestamp_ms)

def _to_proto(hand_lms) -> landmark_pb2.NormalizedLandmarkList:
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in hand_lms
    ])
    return proto

# —— Одноцветная отрисовка скелета руки —— #
def draw_hands_mono(img_bgr, hand_lms, color=(0, 255, 255), r=2, t=2):
    mp_drawing = mp.solutions.drawing_utils
    landmark_spec   = mp_drawing.DrawingSpec(color=color, thickness=-1, circle_radius=r)
    connection_spec = mp_drawing.DrawingSpec(color=color, thickness=t,  circle_radius=r)
    if hasattr(hand_lms, "landmark"):
        proto = hand_lms
    else:
        proto = _to_proto(hand_lms)
    mp_drawing.draw_landmarks(
        img_bgr,
        landmark_list=proto,
        connections=HAND_CONNECTIONS,
        landmark_drawing_spec=landmark_spec,
        connection_drawing_spec=connection_spec,
    )

def norm_name(s: str) -> str:
    return "".join(str(s).lower().split())

# ======== TTS（pyttsx3）========
class Speaker:
    def __init__(self, enable=True):
        self.enable = enable
        self._engine = None
        self._lock = threading.Lock()
        if enable:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', 190)
                self._engine.setProperty('volume', 1.0)
            except Exception:
                self._engine = None
                self.enable = False

    def say_async(self, text: str):
        if not self.enable or not text:
            return
        def _run():
            try:
                with self._lock:
                    self._engine.stop()
                    self._engine.say(text)
                    self._engine.iterate()
                    t0 = time.time()
                    while self._engine.isBusy() and (time.time() - t0) < 1.2:
                        self._engine.iterate()
                        time.sleep(0.01)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

# ======== Отрисовка текста (CJK/кириллица, преимущественно через Pillow) ========
_PIL_OK = False
_FONT_PATH = None
def _init_font():
    global _PIL_OK, _FONT_PATH
    try:
        from PIL import ImageFont  # noqa
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

def draw_text_cn(img_bgr, text, xy, font_size=20, color=(255,255,255), stroke=None, ui_hint=True):
    """
    Единая отрисовка текста:
    - по умолчанию в стиле фронтенда: мелкий шрифт, построчная укладка в левом нижнем углу (ui_hint=True).
    - при ui_hint=False — точное позиционирование по переданному xy (для небольших подписей рядом с целью).
    """
    # единый стиль: Microsoft YaHei + фиксированный размер + чисто белый
    color = (255, 255, 255)
    font_size = int(UNIFIED_FONT_PX)

    H, W = img_bgr.shape[:2]
    # укладка в правом верхнем углу: вычисляем верхний y и выравниваем вправо по ширине текста
    y_top = _ui_next_y_top(font_size) if ui_hint else _ui_next_y_top(font_size)
    # сначала оцениваем размер текста
    tw = th = 0
    font_obj = None

    if _PIL_OK and _FONT_PATH:
        try:
            from PIL import Image, ImageDraw, ImageFont
            font_obj = ImageFont.truetype(_FONT_PATH, font_size)
            # вычисляем размер текста
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
            x = max(8, W - _UI_RIGHT_MARGIN - tw)
            y = y_top
            draw.text((x, y), text, fill=(255,255,255), font=font_obj)
            img_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    # запасной вариант OpenCV: оцениваем размер и выравниваем вправо
    if tw <= 0 or th <= 0:
        scale = font_size/24.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(8, W - _UI_RIGHT_MARGIN - int(tw))
    y_baseline = int(y_top + th)
    cv2.putText(img_bgr, text, (x, y_baseline), cv2.FONT_HERSHEY_SIMPLEX, font_size/24.0, color, 2, cv2.LINE_AA)

# ======== Вспомогательные функции ========
def clamp01(x): return max(0.0, min(1.0, x))

def draw_progress_bars(vis, align_score, range_score):
    """Первая полоса = выравнивание, вторая = расстояние (≈1), т. е. насколько ratio близок к 1"""
    H, W = vis.shape[:2]
    bar_w = int(W * 0.28)
    bar_h = 12
    gap   = 8
    x0    = 12
    y0    = H - 2*bar_h - gap - 12
    # фон
    cv2.rectangle(vis, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(vis, (x0, y0 + bar_h + gap), (x0 + bar_w, y0 + 2*bar_h + gap), (50, 50, 50), -1)
    # заполнение
    cv2.rectangle(vis, (x0, y0), (x0 + int(bar_w * clamp01(align_score)), y0 + bar_h), (0, 220, 0), -1)
    cv2.rectangle(vis, (x0, y0 + bar_h + gap), (x0 + int(bar_w * clamp01(range_score)), y0 + 2*bar_h + gap), (0, 180, 255), -1)
    draw_text_cn(vis, "Выравнивание", (x0, y0 - 18),                 font_size=18, color=(180,180,180))
    draw_text_cn(vis, "Расстояние (≈1)", (x0, y0 + bar_h + gap - 18),   font_size=18, color=(180,180,180))

def polygon_center_and_area(poly):
    if poly is None or len(poly) < 3:
        return None, 0.0
    poly = np.array(poly, dtype=np.float32)
    M = cv2.moments(poly)
    if abs(M["m00"]) < 1e-6:
        c = np.mean(poly, axis=0)
        return (float(c[0]), float(c[1])), 0.0
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    area = float(cv2.contourArea(poly.astype(np.int32)))
    return (cx, cy), area

def hand_bbox_and_area(lms, W, H):
    xs = [int(p.x * W) for p in lms]
    ys = [int(p.y * H) for p in lms]
    if not xs or not ys:
        return None, 0.0
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    area = float(w * h)
    return (x0, y0, w, h), area

# ======== Жест: распознавание хвата (Grasp) (смягчённая эвристика) ========
THUMB_INDEX_CLOSE = 0.34   # смягчено
FINGERTIP_NEAR    = 0.44   # смягчено
MIN_CURLED_COUNT  = 1      # смягчено

def detect_grasp(hand_lms, W, H):
    box, _ = hand_bbox_and_area(hand_lms, W, H)
    if not box:
        return False, 0.0
    x0, y0, w0, h0 = box
    hand_diag = float(np.hypot(w0, h0)) + 1e-6
    palm_idx = [0, 5, 9, 13, 17]
    px = np.mean([hand_lms[i].x * W for i in palm_idx])
    py = np.mean([hand_lms[i].y * H for i in palm_idx])
    palm = np.array([px, py], dtype=np.float32)
    t4 = np.array([hand_lms[4].x * W, hand_lms[4].y * H], dtype=np.float32)
    t8 = np.array([hand_lms[8].x * W, hand_lms[8].y * H], dtype=np.float32)
    thumb_index_dist = float(np.linalg.norm(t4 - t8)) / hand_diag
    tips = [12, 16, 20]
    dists = []
    for i in tips:
        ti = np.array([hand_lms[i].x * W, hand_lms[i].y * H], dtype=np.float32)
        dists.append(float(np.linalg.norm(ti - palm)) / hand_diag)
    curled_cnt = sum(1 for d in dists if d < FINGERTIP_NEAR)
    cond1 = (thumb_index_dist < THUMB_INDEX_CLOSE)
    cond2 = (curled_cnt >= MIN_CURLED_COUNT)
    score = 0.5 * (1.0 - min(thumb_index_dist / THUMB_INDEX_CLOSE, 1.0)) + \
            0.5 * min(curled_cnt / 3.0, 1.0)
    return (cond1 and cond2), score

# ======== Извлечение точек на границе после сжатия внутрь ========
def inner_offset_edge(mask_bin, offset_px=5, edge_dilate_px=2):
    if offset_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*offset_px+1, 2*offset_px+1))
        eroded = cv2.erode(mask_bin.astype(np.uint8), k, iterations=1)
    else:
        eroded = mask_bin.astype(np.uint8)
    edges = cv2.Canny(eroded*255, 50, 150)
    if edge_dilate_px > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*edge_dilate_px+1, 2*edge_dilate_px+1))
        edges = cv2.dilate(edges, k2, iterations=1)
    return edges  # uint8 0/255

# ======== Сегментация YOLO: выбор лучшей маски по всему кадру или внутри ROI ========
def find_best_mask(frame_bgr, yolo, W, H, target_cls_id, conf_thr=0.10, roi_rect=None):
    results = yolo(frame_bgr, verbose=False)
    best_mask = None
    best_score = 0.0
    if results and results[0].masks is not None:
        r0 = results[0]
        for mask_t, conf_t, cls_t in zip(r0.masks.data, r0.boxes.conf, r0.boxes.cls):
            cls_id = int(cls_t.item())
            conf_value = float(conf_t.item())
            if target_cls_id is not None and cls_id != target_cls_id:
                continue
            if conf_value < conf_thr:
                continue
            mask_np = mask_t.detach().cpu().numpy()
            mask_rz = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_rz > 0.5).astype(np.uint8)

            if roi_rect is not None:
                x0, y0, x1, y1 = roi_rect
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W-1, x1), min(H-1, y1)
                roi = np.zeros_like(mask_bin, dtype=np.uint8)
                roi[y0:y1+1, x0:x1+1] = 1
                overlap = (mask_bin & roi).sum()
                score = float(overlap)
            else:
                score = float(mask_bin.sum())

            if score > best_score:
                best_score = score
                best_mask = mask_bin
    return best_mask

# ======== Инженерный вид: стрелка измерения (засечки на концах + стрелка + значение в пикселях) ========
def draw_measure_arrow(img, p1, p2, txt=None):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    # засечки на концах
    def end_cap(pt, size=8, color=(255,255,255), t=1):
        x, y = pt
        cv2.line(img, (x - size, y), (x + size, y), color, t, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, t, cv2.LINE_AA)
    end_cap(p1, size=7, color=(255,255,255), t=1)
    end_cap(p2, size=7, color=(255,255,255), t=1)
    # стрелка
    cv2.arrowedLine(img, p1, p2, (255,255,255), 2, cv2.LINE_AA, tipLength=0.18)
    # текст
    if txt is None:
        d = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        txt = f"{d}px"
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 2
    (tw, th_text), _ = cv2.getTextSize(txt, font, fs, th)
    pad = 4
    x0 = mid[0] - tw//2 - pad
    y0 = mid[1] - th_text - 6
    x1 = mid[0] + tw//2 + pad
    y1 = mid[1] + 6
    cv2.rectangle(img, (x0, y0), (x1, y1), (32,32,32), -1)
    cv2.putText(img, txt, (x0+pad, y1-6), font, fs, (255,255,255), th, cv2.LINE_AA)

# функция отрисовки пунктирной линии
def draw_dashed_line(img, pt1, pt2, color=(255, 255, 255), thickness=2, dash_length=10, gap_length=5):
    """Рисует пунктирную линию"""
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    line_vec = pt2 - pt1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1:
        return
    
    line_vec = line_vec / line_len  # единичный вектор
    
    # рисуем отрезки пунктира
    current_pos = 0
    while current_pos < line_len:
        start_pos = current_pos
        end_pos = min(current_pos + dash_length, line_len)
        
        start_pt = pt1 + line_vec * start_pos
        end_pt = pt1 + line_vec * end_pos
        
        cv2.line(img, tuple(start_pt.astype(int)), tuple(end_pt.astype(int)), color, thickness)
        
        current_pos += dash_length + gap_length

# функция отрисовки контура руки
def draw_hand_contour(img, hand_lms, W, H, color=(255, 255, 255), thickness=1):
    """Рисует контур выпуклой оболочки по landmarks руки"""
    # получаем все ключевые точки руки
    points = []
    for lm in hand_lms:
        x = int(lm.x * W)
        y = int(lm.y * H)
        points.append([x, y])
    
    if len(points) > 3:
        points = np.array(points, dtype=np.int32)
        # вычисляем выпуклую оболочку
        hull = cv2.convexHull(points)
        # рисуем контур выпуклой оболочки
        cv2.polylines(img, [hull], True, color, thickness)

# проверка контакта руки с объектом
def check_hand_object_contact(hand_box, poly, overlap_threshold=0.15):
    """
    Проверяет, пересекаются ли ограничивающий прямоугольник руки и многоугольник объекта
    Возвращает: (есть ли контакт, доля перекрытия)
    """
    if hand_box is None or poly is None or len(poly) < 3:
        return False, 0.0
    
    # получаем ограничивающий прямоугольник руки
    hx, hy, hw, hh = hand_box
    hand_rect = np.array([
        [hx, hy],
        [hx + hw, hy],
        [hx + hw, hy + hh],
        [hx, hy + hh]
    ], dtype=np.int32)
    
    # создаём маски для вычисления перекрытия
    H = int(max(hy + hh, np.max(poly[:, 1])) + 10)
    W = int(max(hx + hw, np.max(poly[:, 0])) + 10)
    
    hand_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(hand_mask, [hand_rect], 1)
    
    obj_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(obj_mask, [poly.astype(np.int32)], 1)
    
    # вычисляем перекрытие
    intersection = np.logical_and(hand_mask, obj_mask).sum()
    hand_area = hand_mask.sum()
    
    # доля перекрытия (относительно площади руки)
    overlap_ratio = intersection / max(1.0, hand_area)
    
    return overlap_ratio > overlap_threshold, overlap_ratio

# функция определения направления
def get_guidance_direction(hand_center, object_center, hand_area, object_area, hand_box=None, poly=None):
    """
    По положению центра ладони и центра объекта, а также по соотношению площадей возвращает направление подсказки
    Возвращает: (текст направления, нужна ли корректировка вперёд/назад)
    """
    if hand_center is None or object_center is None:
        return None, None
    
    # сначала проверяем, касается ли рука объекта
    is_touching = False
    overlap_ratio = 0.0
    if hand_box is not None and poly is not None:
        is_touching, overlap_ratio = check_hand_object_contact(hand_box, poly, overlap_threshold=0.1)
    
    hx, hy = hand_center
    ox, oy = object_center
    
    # вычисляем горизонтальное и вертикальное отклонение
    dx = ox - hx  # положительное значение — объект справа
    dy = oy - hy  # положительное значение — объект снизу
    
    # если рука уже касается объекта, сразу возвращаем "向前" (вперёд)
    if is_touching:
        return "向前", f"Контакт: {overlap_ratio:.1%}"
    
    # если контакта нет — подсказываем вверх/вниз/влево/вправо
    # определяем основное направление
    h_threshold = 30  # порог горизонтального отклонения (пиксели)
    v_threshold = 30  # порог вертикального отклонения (пиксели)
    
    h_dir = None
    v_dir = None
    
    # горизонтальное направление
    if abs(dx) > h_threshold:
        h_dir = "向右" if dx > 0 else "向左"
    
    # вертикальное направление
    if abs(dy) > v_threshold:
        v_dir = "向下" if dy > 0 else "向上"
    
    # выбираем направление с наибольшим смещением
    if abs(dx) > abs(dy) and h_dir:
        # горизонтальное смещение больше
        return h_dir, v_dir
    elif v_dir:
        # вертикальное смещение больше или равно
        return v_dir, h_dir
    else:
        # уже около центра, но контакта ещё нет — подсказываем приблизиться
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 50:  # очень близко, но контакта ещё нет
            return "向前", "Медленно приближайтесь"
        else:
            return "保持", None

# функция воспроизведения аудио
def play_guidance_audio(direction):
    """Воспроизводит аудиоподсказку направления"""
    # напрямую вызываем новую функцию воспроизведения аудио
    play_audio_threadsafe(direction)
    # синхронно обновляем текст команды на кнопке внизу
    try:
        if isinstance(direction, str) and direction.strip():
            set_current_command(direction.strip())
    except Exception:
        pass

# функция проверки центрирования
def get_center_guidance(object_center, frame_center, threshold=30):
    """
    Определяет, находится ли объект в центре кадра, и возвращает направление подсказки
    Возвращает: (текст направления, отцентрирован ли объект)
    """
    if object_center is None:
        return None, False
    
    ox, oy = object_center
    cx, cy = frame_center
    
    dx = cx - ox  # положительное значение — нужно сдвинуть вправо
    dy = cy - oy  # положительное значение — нужно сдвинуть вниз
    
    # проверяем, отцентрирован ли уже объект
    distance = np.sqrt(dx**2 + dy**2)
    if distance < threshold:
        return "已居中", True
    
    # определяем основное направление (лево/право и верх/низ поменяны местами)
    if abs(dx) > abs(dy):
        return "向左" if dx > 0 else "向右", False  # поменяно местами
    else:
        return "向上" if dy > 0 else "向下", False  # поменяно местами

def main(headless: bool = False, prompt_name: str = None, stop_event=None):

    # оптимизации OpenCV
    try:
        import cv2
        cv2.setUseOptimized(True)
        cv2.setNumThreads(2)   # зависит от числа ядер CPU; на устройствах типа Raspberry Pi можно 1
    except Exception:
        pass




    # если передан prompt_name, заменяем им глобальный PROMPT_NAME
    global PROMPT_NAME
    if prompt_name:
        PROMPT_NAME = prompt_name
        print(f"[YOLOMEDIA] Using dynamic prompt: {PROMPT_NAME}")
    
    speaker = Speaker(ENABLE_TTS)
    last_tts_ts = 0.0
    MODE = "SEGMENT"  # режимы: SEGMENT -> FLASH -> CENTER_GUIDE -> TRACK
    colors = Colors()

    FRAME_IDX    = 0
    last_mask    = None      # «маска цели» предыдущего кадра (для подавления шума по IoU)
    flow_mask    = None      # маска, экстраполированная оптическим потоком (обновляется в существующем коде)
    flow_grace   = 0         # счётчик кадров, которые оптический поток может «продержаться» после потери детекции YOLOE
    last_seen_ts = 0.0       # метка времени последней успешной детекции YOLOE
    locked_id    = None      # (опционально) если tracker сохраняет id, ниже можно выбирать тот же id
    # параметры обновления/отказоустойчивости (можно подстроить)
    REDETECT_EVERY = 5       # каждые 5 кадров принудительно «доверяем YOLOE»
    FLOW_GRACE_MAX = 8       # при непрерывной потере детекции YOLOE оптический поток держит максимум 8 кадров
    IOU_MIN_KEEP   = 0.20    # при слишком низком IoU новой/старой маски — плавное смешивание, чтобы избежать мерцания



    print("[INIT] Загрузка модели YOLO...")
    # NOTE: shoppingbest больше не используется в сценарии поиска предметов; если он нужен в других режимах, можно оставить yolo = YOLO(...), но в этом сценарии он не используется
    # yolo = YOLO(YOLO_MODEL_PATH)

    # —— Сразу включаем бэкенд YOLOE с текстовыми подсказками (без предварительной проверки shoppingbest) ——
    use_yoloe = False
    yoloe_backend = None
    if _YOLOE_READY:
        try:
            yoloe_backend = YoloEBackend()                  # модель можно задать переменной окружения YOLOE_MODEL_PATH
            yoloe_backend.set_text_classes([PROMPT_NAME])   # текстовые классы
            use_yoloe = True
            print(f"[DETECTOR] YOLOE text-prompt backend enabled for: {PROMPT_NAME}", flush=True)
        except Exception as e:
            print(f"[DETECTOR] YOLOE init failed: {e}", flush=True)
    else:
        print("[DETECTOR] YOLOE backend not ready (import failed)", flush=True)

    # сопоставление имён классов (упрощено для режима YOLOE)
    if use_yoloe:
        # в режиме YOLOE только один целевой класс
        id_to_name = {0: PROMPT_NAME}
        name_to_id = {norm_name(PROMPT_NAME): 0}
        target_cls_id = 0
    else:
        # если в будущем понадобится классический YOLO, его можно инициализировать здесь
        id_to_name = {}
        name_to_id = {}
        target_cls_id = None

    # целевой класс уже задан выше в режиме YOLOE

    print(f"[CLASS] target id={target_cls_id}, name={id_to_name.get(target_cls_id, 'N/A')}")
    print(f"[ПОРОГ] conf >= {CONF_THRESHOLD:.2f}")

    # Hand Landmarker
    print("[INIT] Инициализация Hand Landmarker...")
    base = BaseOptions(model_asset_path=HAND_TASK_PATH)
    hand_options = HandLandmarkerOptions(
        base_options=base,
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.70,
        result_callback=on_result
    )
    landmarker = HandLandmarker.create_from_options(hand_options)

    W = None
    H = None
    print("[Bridge] Ожидание кадров с ESP32 ...")

    # [headless] окно создаётся только не в headless-режиме (исходная логика сохранена, добавлена внешняя проверка)
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # кэш оптического потока
    old_gray = None
    p0 = None
    lock_edge_debug = None     # отладочная визуализация: внутренняя граница
    track_frame_count = 0      # управляет частотой мониторинга окрестности
    last_poly_box = None       # ограничивающий прямоугольник текущего многоугольника

    fps_hist = []
    
    # переменные автозахвата
    auto_lock_start_time = None  # время начала обнаружения объекта
    auto_lock_delay = 1.0        # автозахват через 1 секунду
    last_detected_mask = None    # последняя обнаруженная маска
    
    # переменные анимации мигания
    flash_start_time = None      # время начала мигания
    flash_duration = 1.0         # длительность мигания (секунды)
    flash_frequency = 1          # частота мигания (Гц) — мигает только один раз
    flash_mask = None            # маска для мигания
    flash_color = (0, 255, 255)  # цвет мигания (жёлтый)

    # переменные подсказок
    last_guidance_time = 0
    last_guidance_direction = None

    # переменные подсказок центрирования
    center_guide_mask = None      # маска для подсказок центрирования
    center_guide_start = None     # время начала подсказок центрирования
    center_threshold = 30         # порог центрирования (пиксели)
    last_center_guide_time = 0   # время последней голосовой подсказки центрирования
    center_reached = False        # достигнут ли уже центр

    # переменные отслеживания захвата
    grasp_tracking_frames = []  # последние положения руки и объекта
    grasp_tracking_duration = 1.0  # должно длиться 1 секунду
    grasp_movement_threshold = 10  # минимальный порог перемещения в пикселях (порог повышен)
    grasp_detected = False  # обнаружен ли уже захват
    grasp_start_time = None  # время начала обнаружения совместного движения
    
    # опорные точки фона (для обнаружения движения камеры) — инициализация перенесена сюда
    background_points = None
    old_background_gray = None

    try:
        while True:
            # проверяем событие остановки
            if stop_event and stop_event.is_set():
                print("[YOLOMEDIA] Stop event detected, exiting...")
                break
                
            frame = bridge_io.wait_raw_bgr(timeout_sec=0.5)
            if frame is None:
                # кадр не получен — продолжаем ждать (ESP32 ещё не подключён или временно нет новых кадров)
                # [headless] уступаем планировщику 1 мс, чтобы не крутиться вхолостую
                if headless:
                    cv2.waitKey(1)
                continue
            
            # каждый кадр сбрасываем текстовый UI-оверлей в левом нижнем углу
            H, W = frame.shape[:2]
            ui_reset_overlay(H)

            vis = frame.copy()
            t_now = time.time()

            # прореживание кадров + понижение разрешения (распознавание руки)
            if FRAME_IDX % HAND_FPS_DIV == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if HAND_DOWNSCALE and HAND_DOWNSCALE != 1.0:
                    small = cv2.resize(rgb, None, fx=HAND_DOWNSCALE, fy=HAND_DOWNSCALE, interpolation=cv2.INTER_AREA)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                landmarker.detect_async(mp_image, int(t_now * 1000))
            # иначе пропускаем и переиспользуем прошлый _last_result; Landmarker сам выполняет tracking


            # получаем центр ладони, рамку руки, хват (смягчённый вариант)
            hand_center = None
            hand_area = None
            hand_box = None
            grasp_now = False
            grasp_score = 0.0
            if _last_result is not None:
                res, _ = _last_result
                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    l0 = res.hand_landmarks[0]
                    
                    # рисуем скелет руки
                    draw_hands_mono(vis, l0, color=(0, 255, 255), r=2, t=2)
                    
                    # рисуем контур руки (вместо прямоугольной рамки)
                    draw_hand_contour(vis, l0, W, H, color=(255, 255, 255), thickness=1)
                    
                    xs = [p.x * W for p in l0]
                    ys = [p.y * H for p in l0]
                    hand_center = (float(sum(xs)/len(xs)), float(sum(ys)/len(ys)))
                    hand_box, hand_area = hand_bbox_and_area(l0, W, H)
                    # отрисовка прямоугольной рамки закомментирована
                    # if hand_box:
                    #     x0, y0, w0, h0 = hand_box
                    #     cv2.rectangle(vis, (x0, y0), (x0+w0, y0+h0), (0,255,255), 1)
                    grasp_now, grasp_score = detect_grasp(l0, W, H)
                    draw_text_cn(vis, f"Оценка хвата: {grasp_score:.2f}", (10, 70), font_size=18, color=(0, 180, 255))
                    

            if MODE == "SEGMENT":
                # —— Только YOLOE: сегментация по текстовой подсказке на каждом кадре + выбор крупнейшей цели (shoppingbest и дублирующий блок YOLOE удалены) ——
                FRAME_IDX += 1
                candidate_masks = []
                detected_object = False

                if use_yoloe and yoloe_backend is not None:
                    # запускаем на каждом кадре; persist=True помогает сохранять ID цели
                    det = yoloe_backend.segment(frame, conf=0.20, iou=0.45, imgsz=640, persist=True)
                    H, W = frame.shape[:2]

                    # выбираем маску: приоритет у совпадающей с locked_id, иначе — с наибольшей площадью
                    chosen_idx = None
                    if det["masks"]:
                        if locked_id is not None and det["ids"] and (locked_id in det["ids"]):
                            chosen_idx = det["ids"].index(locked_id)
                        else:
                            areas = [int(m.sum()) for m in det["masks"]]
                            chosen_idx = int(np.argmax(areas))

                    if chosen_idx is not None:
                        m = det["masks"][chosen_idx]
                        if m.shape[:2] != (H, W):
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

                        mask_bin = (m > 0).astype(np.uint8)
                        candidate_masks.append({
                            "mask": mask_bin,
                            "area": int(mask_bin.sum()),
                            "name": PROMPT_NAME,
                            "cls_id": 0,
                            "conf": 0.99,
                        })
                        detected_object = True

                        # простая визуализация (полупрозрачный слой + контур), не влияет на последующую логику
                        colored = np.zeros_like(frame, dtype=np.uint8)
                        colored[mask_bin == 1] = (0, 255, 255)
                        vis = cv2.addWeighted(vis, 1.0, colored, MASK_ALPHA, 0)
                        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if contours:
                            # выбираем наибольший контур и умеренно сглаживаем
                            largest_contour = max(contours, key=cv2.contourArea)
                            # умеренно упрощаем алгоритмом Дугласа — Пекера, сохраняя больше деталей
                            epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(largest_contour, True)  # меньший epsilon сохраняет больше деталей
                            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                            cv2.drawContours(vis, [smoothed_contour], -1, (0, 255, 255), STROKE_WIDTH)

                        # запоминаем id, чтобы цель меньше «скакала»
                        if det["ids"] and len(det["ids"]) > chosen_idx and det["ids"][chosen_idx] is not None:
                            locked_id = int(det["ids"][chosen_idx])

                else:
                    # YOLOE не готов: выводим подсказку и показываем исходный кадр (не блокируя фронтенд)
                    draw_text_cn(vis, "YOLOE не готов, показан исходный кадр", (10, 100), font_size=22, color=(0, 215, 255))

                # выбор маски с наибольшей площадью  ←—— ниже этой строки сохранён исходный код

                # выбор маски с наибольшей площадью
                if candidate_masks:
                    # сортировка по убыванию площади
                    candidate_masks.sort(key=lambda x: x['area'], reverse=True)
                    largest_mask_info = candidate_masks[0]
                    last_detected_mask = largest_mask_info['mask']
                    
                    # опционально: специальная метка на крупнейшем объекте
                    contours, _ = cv2.findContours(last_detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        # находим центр наибольшего контура
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            # рисуем круговую метку в центре крупнейшего объекта
                            cv2.circle(vis, (cx, cy), 8, (0, 255, 0), 2)
                            cv2.circle(vis, (cx, cy), 12, (0, 255, 0), 1)
                            # подпись цели: размещается прямо на месте
                            draw_text_cn(vis, "Цель", (cx + 15, cy - 5), font_size=16, color=FRONTEND_COLORS["ok"], ui_hint=False)
                    
                    # вывод информации о детекции
                    if len(candidate_masks) > 1:
                        draw_text_cn(vis, f"Обнаружено объектов: {len(candidate_masks)}, выбран крупнейший (площадь: {largest_mask_info['area']})", 
                                   (10, H - 30), font_size=16, color=(255, 255, 0))
                
                # логика автозахвата
                if detected_object and last_detected_mask is not None:
                    if auto_lock_start_time is None:
                        auto_lock_start_time = t_now
                        print(f"[AUTO] Обнаружен объект, выбран крупнейший (площадь: {np.sum(last_detected_mask)}), запуск обратного отсчёта...")
                        #play_guidance_audio("检测到物体")  # добавлена эта строка
                    
                    elapsed = t_now - auto_lock_start_time
                    remaining = auto_lock_delay - elapsed
                    
                    if remaining > 0:
                        # вывод обратного отсчёта (перенесён в левый нижний угол, стиль фронтенда)
                        draw_text_cn(vis, f"Объект обнаружен, автозахват через {remaining:.1f} с", (10, 100), font_size=16, color=FRONTEND_COLORS["text"], stroke=(0,0,0))
                        
                        # рисуем рамку захвата — пунктир означает подготовку к захвату
                        if last_detected_mask is not None:
                            contours, _ = cv2.findContours(last_detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            if contours:
                                # находим наибольший контур
                                largest_contour = max(contours, key=cv2.contourArea)
                                # упрощаем контур
                                epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(largest_contour, True)
                                smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                                
                                # меняем яркость цвета по ходу обратного отсчёта
                                progress = 1.0 - (remaining / auto_lock_delay)
                                color_intensity = int(100 + 155 * progress)  # от 100 до 255
                                lock_color = (0, color_intensity, color_intensity)  # жёлтый, постепенно ярче
                                
                                # рисуем пунктирный контур
                                pts = smoothed_contour.reshape(-1, 2)
                                for i in range(len(pts)):
                                    pt1 = tuple(pts[i])
                                    pt2 = tuple(pts[(i + 1) % len(pts)])
                                    # эффект пунктира (за счёт коротких отрезков)
                                    draw_dashed_line(vis, pt1, pt2, color=lock_color, thickness=3, 
                                                   dash_length=15, gap_length=8)
                    else:
                        # переход в режим мигания
                        print("[AUTO] Переход в режим анимации мигания")
                        MODE = "FLASH"
                        flash_start_time = t_now
                        flash_mask = last_detected_mask.copy()
                        auto_lock_start_time = None
                        play_guidance_audio("检测到物体") 
                else:
                    # объект не обнаружен — сбрасываем таймер
                    if auto_lock_start_time is not None:
                        print("[AUTO] Объект потерян, обратный отсчёт сброшен")
                    auto_lock_start_time = None
                    last_detected_mask = None
                    draw_text_cn(vis, "Сегментация... ожидание обнаружения объекта", (10, 100), font_size=16, color=FRONTEND_COLORS["muted"])

            elif MODE == "FLASH":
                # режим анимации мигания
                if flash_start_time is not None and flash_mask is not None:
                    elapsed = t_now - flash_start_time
                    
                    if elapsed < flash_duration:
                        # вычисляем эффект появления/затухания
                        # первые 0.3 с — появление, средние 0.4 с — удержание, последние 0.3 с — затухание
                        if elapsed < 0.3:
                            # фаза появления
                            alpha = elapsed / 0.3 * 0.8  # от 0 до 0.8
                        elif elapsed < 0.7:
                            # фаза удержания
                            alpha = 0.8
                        else:
                            # фаза затухания
                            alpha = (1.0 - elapsed) / 0.3 * 0.8  # от 0.8 до 0
                        
                        # рисуем мигающую маску
                        colored = np.zeros_like(frame, dtype=np.uint8)
                        colored[flash_mask == 1] = flash_color
                        vis = cv2.addWeighted(vis, 1.0 - alpha, colored, alpha, 0)
                        
                        # рисуем контур (фиксированная толщина, плавная смена цвета)
                        contours, _ = cv2.findContours(flash_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if contours:
                            # цвет контура тоже меняется вместе с alpha
                            contour_color = tuple(int(c * (0.5 + alpha * 0.5)) for c in flash_color)
                            cv2.drawContours(vis, contours, -1, contour_color, STROKE_WIDTH + 1)
                        
                        # вывод текста подсказки (левый нижний угол)
                        draw_text_cn(vis, "Захват цели...", (10, 100), font_size=18, color=FRONTEND_COLORS["accent"]) 
                    else:
                        # мигание завершено: инициализируем трекинг оптическим потоком и переходим в режим подсказок центрирования
                        print("[AUTO] Мигание завершено, инициализация трекинга оптическим потоком")
                        edge_mask = inner_offset_edge(flash_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                        
                        if pts is not None and len(pts) >= 8:
                            p0 = pts
                            old_gray = gray
                            MODE = "CENTER_GUIDE"
                            lock_edge_debug = edge_mask.copy()
                            track_frame_count = 0
                            center_guide_start = t_now
                            center_reached = False
                            flash_start_time = None
                            flash_mask = None
                            last_detected_mask = None
                            print(f"[LOCK] Особых точек на внутренней границе={len(p0)} → CENTER_GUIDE")
                        else:
                            print("[LOCK] Недостаточно особых точек на внутренней границе, возврат в режим детекции")
                            MODE = "SEGMENT"
                            flash_start_time = None
                            flash_mask = None
                            last_detected_mask = None
            
            elif MODE == "CENTER_GUIDE":
                # режим подсказок центрирования (с трекингом оптическим потоком)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                poly_center = None
                poly_area = 0.0
                
                if old_gray is not None and p0 is not None and len(p0) >= 5:
                    # трекинг оптическим потоком
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **LK_PARAMS)
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:
                            p0 = good_new.reshape(-1, 1, 2)
                            hull = cv2.convexHull(good_new.reshape(-1,1,2))
                            poly = hull.reshape(-1, 2)
                            
                            if len(poly) >= 3:
                                H, W = frame.shape[:2]

                                # растеризуем текущий многоугольник оптического потока в маску (для расчёта IoU с маской YOLOE)
                                poly_mask = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(poly_mask, [poly.astype(np.int32)], 1)

                                # пониженная частота: каждые 3 кадра повторная детекция YOLOE, в остальных кадрах держимся на оптическом потоке
                                need_reseed = False
                                new_det_mask = None

                                if use_yoloe and yoloe_backend is not None and (FRAME_IDX % 3 == 0):
                                    # отладочная информация
                                    if FRAME_IDX % 30 == 0:  # печать раз в 30 кадров
                                        print(f"[YOLOE] Детекция в реальном времени, кадр {FRAME_IDX}")
                                    det = yoloe_backend.segment(frame, conf=0.20, iou=0.45, imgsz=640, persist=True)
                                    if det["masks"]:
                                        # берём маску с наибольшей площадью
                                        areas = [int(m.sum()) for m in det["masks"]]
                                        j = int(np.argmax(areas))
                                        m = det["masks"][j]
                                        if m.shape[:2] != (H, W):
                                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                                        new_det_mask = (m > 0).astype(np.uint8)

                                        # IoU с текущим многоугольником оптического потока
                                        inter = np.logical_and(new_det_mask, poly_mask).sum()
                                        union = np.logical_or(new_det_mask, poly_mask).sum() + 1e-6
                                        iou   = inter / union

                                        # слишком низкий IoU — трекинг «уплыл»: заново засеиваем оптический поток маской YOLOE
                                        # снижаем порог, чтобы YOLOE легче обновлял оптический поток
                                        if iou < 0.5:  # повышено с IOU_MIN_KEEP (0.20) до 0.5
                                            need_reseed = True
                                            # засеиваем «особыми точками внутренней границы» новой маски
                                            edge_mask = inner_offset_edge(new_det_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                                            gray2 = gray  # полутоновое изображение этого кадра уже вычислено выше
                                            pts = cv2.goodFeaturesToTrack(gray2, mask=edge_mask, **FEATURE_PARAMS)
                                            if pts is not None and len(pts) >= 8:
                                                p0 = pts
                                                old_gray = gray2
                                                # обновляем last_mask для согласованности последующей логики
                                                last_mask = new_det_mask.copy()
                                                last_seen_ts = time.time()
                                                flow_grace = 0
                                                print("[RESEED] Низкий IoU YOLOE — повторный засев (особые точки оптического потока обновлены)")

                                # если в этом кадре повторного засева не было, но у YOLOE есть результат, близкий к poly, делаем «плавное слияние» для подавления дрожания
                                if (not need_reseed) and (new_det_mask is not None):
                                    inter = np.logical_and(new_det_mask, poly_mask).sum()
                                    union = np.logical_or(new_det_mask, poly_mask).sum() + 1e-6
                                    iou   = inter / union
                                    # снижаем порог слияния, чтобы результат YOLOE принимался охотнее
                                    if iou < 0.95:  # повышено с 0.90 до 0.95
                                        # увеличиваем вес YOLOE, чтобы детекция в реальном времени была заметнее
                                        poly_mask = ((0.8 * new_det_mask + 0.2 * poly_mask) > 0.5).astype(np.uint8)
                                        # записываем обновлённую poly_mask в переменные визуализации и подсказок (если дальше используются last_detected_mask/last_mask)
                                        last_mask = poly_mask.copy()
                                                                                # обновляем контур многоугольника, чтобы визуализация обновлялась в реальном времени
                                        contours, _ = cv2.findContours(poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                        if contours:
                                            # находим наибольший контур
                                            largest_contour = max(contours, key=cv2.contourArea)
                                            # детальная обработка контура, сохраняющая больше деталей
                                            epsilon = TRACK_EPSILON_FACTOR * cv2.arcLength(largest_contour, True)
                                            poly = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)
                                            # обработка выпуклой оболочкой закомментирована, чтобы сохранить детали исходного контура
                                            # hull = cv2.convexHull(poly.reshape(-1,1,2))
                                            # poly = hull.reshape(-1, 2)
                                            # пересчитываем особые точки
                                            edge_mask = inner_offset_edge(poly_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                                            pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                                            if pts is not None and len(pts) >= 5:
                                                p0 = pts

                                # рисуем отслеживаемый многоугольник — более толстыми линиями
                                cv2.polylines(vis, [poly.astype(np.int32)], isClosed=True, color=(0,255,255), thickness=STROKE_WIDTH)
                                
                                # вычисляем центр многоугольника
                                poly_center, poly_area = polygon_center_and_area(poly)
                                
                                if poly_center:
                                    object_center = (int(poly_center[0]), int(poly_center[1]))
                                    
                                    # центр кадра
                                    frame_center = (W // 2, H // 2)
                                    
                                    # рисуем центральную точку предмета
                                    cv2.circle(vis, object_center, 8, (0, 255, 0), -1)
                                    cv2.circle(vis, object_center, 12, (0, 255, 0), 2)
                                    
                                    # рисуем перекрестие в центре кадра
                                    cv2.line(vis, (frame_center[0] - 20, frame_center[1]), 
                                            (frame_center[0] + 20, frame_center[1]), (255, 255, 255), 2)
                                    cv2.line(vis, (frame_center[0], frame_center[1] - 20), 
                                            (frame_center[0], frame_center[1] + 20), (255, 255, 255), 2)
                                    
                                    # рисуем пунктирную направляющую
                                    draw_dashed_line(vis, object_center, frame_center, 
                                                   color=(255, 255, 0), thickness=2, 
                                                   dash_length=10, gap_length=5)
                                    
                                    # получаем направление подсказки
                                    direction, is_centered = get_center_guidance(object_center, frame_center, center_threshold)
                                    
                                    if not center_reached:
                                        if is_centered:
                                            # центр достигнут, проигрываем звук OK
                                            center_reached = True
                                            last_center_guide_time = t_now
                                            play_guidance_audio("OK")
                                            try:
                                                bridge_io.send_ui_final("✓ Предмет в центре!")
                                            except Exception:
                                                pass
                                            draw_text_cn(vis, "✓ Предмет в центре!", (10, 60), font_size=18, color=FRONTEND_COLORS["ok"]) 
                                        else:
                                            # вывод текста подсказки
                                            msg = f"Переместите предмет в центр кадра: {direction}"
                                            try:
                                                # троттлинг: при каждой голосовой подсказке также отправляем final
                                                if t_now - last_center_guide_time > GUIDANCE_INTERVAL_SEC:
                                                    bridge_io.send_ui_final(msg)
                                            except Exception:
                                                pass
                                            draw_text_cn(vis, msg, 
                                                       (10, 40), font_size=18, color=FRONTEND_COLORS["text"])
                                            
                                            # вывод информации о расстоянии
                                            dx = frame_center[0] - object_center[0]
                                            dy = frame_center[1] - object_center[1]
                                            distance = int(np.sqrt(dx**2 + dy**2))
                                            draw_text_cn(vis, f"Расстояние: {distance}px", 
                                                       (10, 60), font_size=16, color=FRONTEND_COLORS["muted"])
                                            
                                            # воспроизводим голосовую подсказку
                                            if t_now - last_center_guide_time > GUIDANCE_INTERVAL_SEC:
                                                play_guidance_audio(direction)
                                                last_center_guide_time = t_now
                                    else:
                                        # уже в центре, выводим сообщение об успехе
                                        try:
                                            bridge_io.send_ui_final("✓ Предмет успешно перемещён в центр!")
                                        except Exception:
                                            pass
                                        draw_text_cn(vis, "✓ Предмет успешно перемещён в центр!", 
                                                   (10, 60), font_size=18, color=FRONTEND_COLORS["ok"])
                                        
                                        # через 1 секунду переходим в режим трекинга руки
                                        if t_now - last_center_guide_time > 1.0:
                                            print("[CENTER] Переход в режим трекинга руки")
                                            try:
                                                bridge_io.send_ui_final("Переход в режим трекинга руки")
                                            except Exception:
                                                pass
                                            MODE = "TRACK"
                                            # сохраняем текущее состояние трекинга оптическим потоком
                                else:
                                    # не удалось вычислить центр многоугольника, выводим предупреждение
                                    draw_text_cn(vis, "Трекинг объекта...", (10, 100), font_size=20, color=(255, 255, 0))
                        else:
                            # недостаточно точек оптического потока, пробуем повторную детекцию
                            MODE = "SEGMENT"
                            old_gray = None
                            p0 = None
                            print("[CENTER] Сбой трекинга оптическим потоком, возврат в режим детекции")
                
                old_gray = gray

            else:  # MODE == "TRACK"
                # режим трекинга руки (исходная логика без изменений)
                align_score = 0.0
                range_score = 0.0
                ratio = None

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                track_frame_count += 1

                relock_done = False
                poly_center = None
                poly_area = 0.0

                # инициализируем camera_movement значением по умолчанию
                camera_movement = np.array([0.0, 0.0])
                
                # инициализируем или обновляем опорные точки фона (берём точки вне многоугольника объекта)
                if background_points is None or track_frame_count % 30 == 0:
                    # берём несколько особых точек фона по углам кадра
                    mask_for_bg = np.ones((H, W), dtype=np.uint8) * 255
                    if last_poly_box:
                        x, y, w, h = last_poly_box
                        # расширяем область, исключая объект и руку
                        expand = 100
                        x1 = max(0, x - expand)
                        y1 = max(0, y - expand)
                        x2 = min(W, x + w + expand)
                        y2 = min(H, y + h + expand)
                        mask_for_bg[y1:y2, x1:x2] = 0
                    
                    # извлекаем особые точки в области фона
                    try:
                        bg_pts = cv2.goodFeaturesToTrack(gray, maxCorners=20, 
                                                       qualityLevel=0.1, 
                                                       minDistance=30, 
                                                       mask=mask_for_bg)
                        if bg_pts is not None and len(bg_pts) >= 5:
                            background_points = bg_pts
                            old_background_gray = gray.copy()
                    except Exception as e:
                        #print(f"[TRACK] Не удалось извлечь особые точки фона: {e}")
                        background_points = None
                
                # вычисляем смещение фона (движение камеры)
                if old_background_gray is not None and background_points is not None and len(background_points) > 0:
                    try:
                        bg_p1, bg_st, _ = cv2.calcOpticalFlowPyrLK(
                            old_background_gray, gray, background_points, None, **LK_PARAMS
                        )
                        if bg_p1 is not None and bg_st is not None:
                            good_bg_old = background_points[bg_st == 1]
                            good_bg_new = bg_p1[bg_st == 1]
                            if len(good_bg_new) >= 3 and len(good_bg_old) >= 3:
                                # вычисляем среднее смещение фона
                                bg_movement = np.mean(good_bg_new - good_bg_old, axis=0)
                                camera_movement = bg_movement.reshape(2)
                                background_points = good_bg_new.reshape(-1, 1, 2)
                                old_background_gray = gray.copy()
                    except Exception as e:
                        print(f"[TRACK] Сбой расчёта оптического потока фона: {e}")
                        camera_movement = np.array([0.0, 0.0])

                if old_gray is not None and p0 is not None and len(p0) >= 5:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **LK_PARAMS)
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:
                            p0 = good_new.reshape(-1, 1, 2)
                            hull = cv2.convexHull(good_new.reshape(-1,1,2))
                            poly = hull.reshape(-1, 2)
                            
                            if len(poly) >= 3:
                                # единая детекция и коррекция YOLOE в реальном времени (каждый кадр)
                                latest_det_mask = None
                                if use_yoloe and yoloe_backend is not None:
                                    # отладочная информация
                                    if track_frame_count % 30 == 0:  # печать раз в 30 кадров
                                        print(f"[YOLOE] Детекция в реальном времени в режиме TRACK, кадр {track_frame_count}")
                                    
                                    # детекция YOLOE в реальном времени (единый вызов, без дублирования)
                                    det = yoloe_backend.segment(frame, conf=YOLO_CORRECTION_CONF_THRESHOLD, iou=0.45, imgsz=640, persist=True)
                                    if det["masks"]:
                                        # берём маску с наибольшей площадью
                                        areas = [int(m.sum()) for m in det["masks"]]
                                        j = int(np.argmax(areas))
                                        m = det["masks"][j]
                                        if m.shape[:2] != (H, W):
                                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                                        latest_det_mask = (m > 0).astype(np.uint8)
                                        
                                        # IoU с текущим многоугольником оптического потока
                                        poly_mask = np.zeros((H, W), dtype=np.uint8)
                                        cv2.fillPoly(poly_mask, [poly.astype(np.int32)], 1)
                                        inter = np.logical_and(latest_det_mask, poly_mask).sum()
                                        union = np.logical_or(latest_det_mask, poly_mask).sum() + 1e-6
                                        iou = inter / union
                                        
                                        # снижаем порог IoU для более активной коррекции
                                        if iou > YOLO_CORRECTION_IOU_THRESHOLD:  # используем настраиваемый порог
                                            # обновляем многоугольник по результату YOLOE
                                            contours, _ = cv2.findContours(latest_det_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                            if contours:
                                                largest_contour = max(contours, key=cv2.contourArea)
                                                # более детальная обработка контура, меньше чрезмерного упрощения
                                                epsilon = TRACK_EPSILON_FACTOR * cv2.arcLength(largest_contour, True)
                                                poly = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)
                                                
                                                # обновляем особые точки оптического потока
                                                edge_mask = inner_offset_edge(latest_det_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                                                pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                                                if pts is not None and len(pts) >= 5:
                                                    p0 = pts
                                                    #print(f"[TRACK] Коррекция YOLOE в реальном времени, IoU: {iou:.3f}")
                                
                                # проверяем контакт, чтобы выбрать цвет контура
                                is_touching = False
                                overlap_ratio = 0.0
                                if hand_box is not None and poly is not None:
                                    is_touching, overlap_ratio = check_hand_object_contact(hand_box, poly, overlap_threshold=0.1)
                                
                                # рисуем многоугольник (возможно, обновлённый YOLOE) — более толстыми линиями
                                if is_touching:
                                    # при контакте — ярко-зелёный с эффектом свечения
                                    poly_color = (0, 255, 127)
                                    # рисуем более толстый внешний контур как эффект свечения
                                    cv2.polylines(vis, [poly.astype(np.int32)], isClosed=True, 
                                                color=(127, 255, 127), thickness=STROKE_WIDTH + 4)
                                    # добавляем полупрозрачную заливку
                                    overlay = vis.copy()
                                    cv2.fillPoly(overlay, [poly.astype(np.int32)], (0, 255, 0))
                                    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
                                else:
                                    # без контакта — обычный зелёный
                                    poly_color = (0, 255, 0)
                                cv2.polylines(vis, [poly.astype(np.int32)], isClosed=True, color=poly_color, thickness=STROKE_WIDTH)
                                # центроид и площадь многоугольника
                                poly_center, poly_area = polygon_center_and_area(poly)
                                if poly_center:
                                    pc = (int(poly_center[0]), int(poly_center[1]))
                                    cv2.circle(vis, pc, 6, (0,255,0), -1)

                                # ограничивающий прямоугольник многоугольника (для мониторинга окрестности)
                                x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                                last_poly_box = (x, y, w, h)

                                # ====== Оценка выравнивания (первая полоса) ======
                                if hand_center and poly_center:
                                    hc = np.array(hand_center, dtype=np.float32)
                                    oc = np.array(poly_center, dtype=np.float32)
                                    dist = float(np.linalg.norm(oc - hc))
                                    diag = float(np.linalg.norm([W, H]))
                                    align_score = 1.0 - min(dist/(ALIGN_LOOSE_PCT*diag + 1e-6), 1.0)
                                    
                                    # рисуем пунктирную направляющую (вместо прежней сплошной стрелки)
                                    draw_dashed_line(vis, (hc[0], hc[1]), (oc[0], oc[1]), 
                                                   color=(255, 255, 0), thickness=2, 
                                                   dash_length=15, gap_length=10)
                                    
                                    # подсказка направления
                                    direction, secondary = get_guidance_direction(
                                        hand_center, poly_center, hand_area, poly_area,
                                        hand_box, poly
                                    )
                                    
                                    if direction and direction != "保持":
                                        # цвет зависит от наличия контакта
                                        if direction == "向前":
                                            # рука уже касается объекта — выводим зелёным
                                            guide_color = (0, 255, 0)  # зелёный
                                            draw_text_cn(vis, f"Подсказка: {direction} — протяните руку и возьмите", (W//2 - 80, 40), 
                                                       font_size=24, color=guide_color, stroke=(0, 0, 0))
                                        else:
                                            # контакта ещё нет — выводим жёлтым
                                            guide_color = (0, 255, 255)  # жёлтый
                                            draw_text_cn(vis, f"Подсказка: {direction}", (W//2 - 60, 40), 
                                                       font_size=24, color=guide_color, stroke=(0, 0, 0))
                                        
                                        # вывод дополнительной информации (степень контакта или другое направление)
                                        if secondary:
                                            if isinstance(secondary, str):
                                                # информация о степени контакта
                                                draw_text_cn(vis, secondary, (W//2 - 60, 70), 
                                                           font_size=18, color=(0, 255, 0))
                                            else:
                                                # информация о другом направлении
                                                draw_text_cn(vis, f"(или {secondary})", (W//2 - 60, 70), 
                                                           font_size=18, color=(200, 200, 200))
                                        
                                        # воспроизводим голосовую подсказку — гарантируем, что каждое направление будет озвучено
                                        if t_now - last_guidance_time > GUIDANCE_INTERVAL_SEC:
                                            # проверяем, сменилось ли направление или прошло достаточно времени
                                            if direction != last_guidance_direction or t_now - last_guidance_time > GUIDANCE_INTERVAL_SEC * 2:
                                                play_guidance_audio(direction)
                                                last_guidance_direction = direction
                                                last_guidance_time = t_now
                                                print(f"[GUIDE] Воспроизведение аудиоподсказки: {direction}")
                                else:
                                    align_score = 0.0

                                # вывод состояния контакта
                                is_touching, overlap_ratio = check_hand_object_contact(hand_box, poly, overlap_threshold=0.1)
                                if is_touching:
                                    draw_text_cn(vis, f"Состояние: контакт ({overlap_ratio:.1%})", (10, 95), 
                                               font_size=16, color=(0, 255, 0))
                                else:
                                    # вычисляем расстояние между рукой и объектом
                                    if hand_center and poly_center:
                                        distance = np.sqrt((hand_center[0] - poly_center[0])**2 + 
                                                         (hand_center[1] - poly_center[1])**2)
                                        draw_text_cn(vis, f"Расстояние: {distance:.0f}px", (10, 95), 
                                                   font_size=16, color=FRONTEND_COLORS["muted"])

                                # условие успеха: хват (смягчённый)
                                if (_last_result and _last_result[0].hand_landmarks and len(_last_result[0].hand_landmarks) > 0):
                                    l0 = _last_result[0].hand_landmarks[0]
                                    grasp_now, grasp_score = detect_grasp(l0, W, H)
                                else:
                                    grasp_now, grasp_score = False, 0.0
             
                                # код, связанный с guidance_msg, уже встроен в логику подсказок выше

                                # ===== Мониторинг окрестности и повторный захват (переиспользуем результат YOLO) =====
                                if (track_frame_count % PERI_CHECK_EVERY == 0) and (last_poly_box is not None) and (latest_det_mask is not None):
                                    # используем только что полученный результат детекции YOLO, чтобы не вызывать повторно
                                    px, py, pw, ph = last_poly_box
                                    x0 = max(0, px - PERI_MONITOR_PX)
                                    y0 = max(0, py - PERI_MONITOR_PX)
                                    x1 = min(W - 1, px + pw + PERI_MONITOR_PX)
                                    y1 = min(H - 1, py + ph + PERI_MONITOR_PX)
                                    
                                    # проверяем, есть ли в окрестности лучший результат детекции
                                    peri_area = latest_det_mask[y0:y1, x0:x1].sum()
                                    total_area = latest_det_mask.sum()
                                    
                                    # если в окрестности есть значимый результат детекции — захватываем заново
                                    if peri_area > total_area * 0.1:  # в окрестности более 10% площади детекции
                                        edge_mask = inner_offset_edge(latest_det_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                                        pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                                        if pts is not None and len(pts) >= 8:
                                            p0 = pts
                                            old_gray = gray
                                            lock_edge_debug = edge_mask.copy()
                                            #print(f"[PERI] Повторный захват по окрестности, особых точек={len(p0)}")
                            else:
                                MODE = "SEGMENT"; old_gray = None; p0 = None; lock_edge_debug = None
                        else:
                            MODE = "SEGMENT"; old_gray = None; p0 = None; lock_edge_debug = None
                    else:
                        MODE = "SEGMENT"; old_gray = None; p0 = None; lock_edge_debug = None
                else:
                    MODE = "SEGMENT"; old_gray = None; p0 = None; lock_edge_debug = None

  

                if MODE == "SEGMENT":
                    draw_text_cn(vis, "Трекинг потерян → повторное распознавание. Нажмите Enter для повторного захвата", (10, 100), font_size=22, color=(0,0,255))

                old_gray = gray

            # FPS (перенесено в стиль левого нижнего угла)
            if 'fps_hist' not in locals():
                fps_hist = []
            fps_hist.append(t_now)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps = 0.0 if len(fps_hist) < 2 else (len(fps_hist)-1)/(fps_hist[-1]-fps_hist[0])
            draw_text_cn(vis, f"FPS: {fps:.1f}", (10, 40), font_size=16, color=FRONTEND_COLORS["ok"]) 

            # в правом нижнем углу показываем отладочное изображение «внутренняя граница / последний захват»
            if lock_edge_debug is not None:
                # сильно уменьшаем и размещаем в правом нижнем углу
                small = cv2.resize(lock_edge_debug, (0,0), fx=0.22, fy=0.22, interpolation=cv2.INTER_NEAREST)
                sh, sw = small.shape[:2]
                small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                # позиция в правом нижнем углу с отступом 10-12px
                x1 = max(8, W - sw - 12)
                y1 = max(8, H - sh - 12)
                y2 = y1 + sh
                x2 = x1 + sw
                vis[y1:y2, x1:x2] = small_bgr
                # подпись вплотную над изображением, более мелким шрифтом
                #draw_text_cn(vis, "Внутренняя граница", (x1, y1 - 8), font_size=12, color=FRONTEND_COLORS["muted"], ui_hint=False)

            # кнопка «Текущая команда» внизу по центру (рисуется всегда, текст синхронизирован с аудио)
            draw_command_pill(vis, CURRENT_COMMAND_TEXT)

            # вывод (отправляется на фронтенд независимо от headless-режима)
            bridge_io.send_vis_bgr(vis)

            # [headless] окно и клавиатура только не в headless-режиме; в headless через waitKey(1) уступаем планировщику
            if not headless:
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                elif key == ord('r'):
                    MODE = "SEGMENT"; old_gray = None; p0 = None; lock_edge_debug = None
                elif key == 13:  # Enter: захват из SEGMENT и запуск TRACK (сжатие внутрь на 5px)
                    if MODE == "SEGMENT":
                        # ручной захват с помощью YOLOE
                        if use_yoloe and yoloe_backend is not None:
                            det = yoloe_backend.segment(frame, conf=CONF_THRESHOLD, iou=0.45, imgsz=640, persist=True)
                            if det["masks"]:
                                # берём маску с наибольшей площадью
                                areas = [int(m.sum()) for m in det["masks"]]
                                j = int(np.argmax(areas))
                                m = det["masks"][j]
                                if m.shape[:2] != (H, W):
                                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                                best_mask = (m > 0.5).astype(np.uint8)
                            else:
                                best_mask = None
                        else:
                            best_mask = None
                        if best_mask is not None:
                            edge_mask = inner_offset_edge(best_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                            if pts is not None and len(pts) >= 8:
                                p0 = pts
                                old_gray = gray
                                MODE = "TRACK"
                                lock_edge_debug = edge_mask.copy()
                                track_frame_count = 0
                                print(f"[LOCK] Особых точек на внутренней границе={len(p0)} → TRACK")
                            else:
                                print("[LOCK] Недостаточно особых точек на внутренней границе, скорректируйте кадр и повторите.")
                        else:
                            print("[LOCK] В текущем кадре не найдена корректная сегментация, повторите.")
            else:
                # в headless тоже вызываем waitKey(1), чтобы таймеры/колбэки OpenCV получили управление и не было активного ожидания CPU
                cv2.waitKey(1)
                
                # проверка события остановки в headless-режиме
                if stop_event and stop_event.is_set():
                    print("[YOLOMEDIA] Received stop signal in headless mode")
                    break

    finally:
        try:
            landmarker.close()
        except Exception:
            pass
        #cap.release()
        # [headless] окна уничтожаются только не в headless-режиме
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
