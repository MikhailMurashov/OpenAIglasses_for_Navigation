# -*- coding: utf-8 -*-
"""
Рабочий процесс перехода дороги (упрощённая версия — только детекция пешеходного перехода, но с сохранением навигации)
- Версия с прямым подключением, без Celery/Redis
- Детектируется только пешеходный переход, без детекции светофора
- Сохранена навигация по пешеходному переходу (вычисление угла и смещения)
- Сохранена визуализация (направляющие линии, целевые точки и т. п.)
- Сегментация выполняется на каждом кадре; если на кадре она не удалась, маска восстанавливается трекингом оптическим потоком по особым точкам, расставленным по маске предыдущего кадра, и удерживает положение до следующего успешного обнаружения сегментацией
"""
import torch
import os
import time
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
# 【Удалено】from audio_player import play_voice_text — аудио не воспроизводится внутри рабочего процесса

# Опционально: для более аккуратной панели данных (как в blindpath)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageDraw, ImageFont = None, None, None

# Опционально: автоматическое включение детекции препятствий (как в blindpath)
try:
    from obstacle_detector_client import ObstacleDetectorClient
except Exception:
    ObstacleDetectorClient = None

# Модуль детекции светофора
try:
    import trafficlight_detection
    TRAFFIC_LIGHT_AVAILABLE = True
except Exception:
    TRAFFIC_LIGHT_AVAILABLE = False
    trafficlight_detection = None

logger = logging.getLogger(__name__)

# ========== Константы состояний ==========
STATE_SEEKING = "SEEKING_CROSSWALK"      # Поиск и выравнивание по удалённому пешеходному переходу
STATE_WAIT_LIGHT = "WAIT_TRAFFIC_LIGHT"  # Ожидание определения сигнала светофора
STATE_CROSSING = "CROSSING"              # Идёт переход дороги

# ========== Параметры конфигурации ==========
CROSSWALK_MIN_CONF = float(os.getenv('CROSSWALK_MIN_CONF', '0.3'))
CROSSWALK_MIN_AREA = int(os.getenv('CROSSWALK_MIN_AREA', '5000'))
BLIND_MIN_CONF = float(os.getenv('BLIND_MIN_CONF', '0.34'))  # Минимальная уверенность для тактильной дорожки (выше, против ложных срабатываний)
ANGLE_THRESH_DEG = float(os.getenv('CROSSWALK_ANGLE_THRESH_DEG', '5.0'))  # Порог по умолчанию слегка ослаблен
OFFSET_THRESH = float(os.getenv('CROSSWALK_OFFSET_THRESH', '0.08'))        # Порог по умолчанию слегка ослаблен

# Пороги выравнивания на дальней дистанции (мягче, чтобы избежать излишней чувствительности)
SEEKING_ANGLE_THRESH_DEG = 15.0  # Порог угла на дальней дистанции (мягче)
SEEKING_OFFSET_THRESH = 0.20     # Порог смещения на дальней дистанции (мягче)

# Пороги выравнивания на дальней дистанции (условия «очень близко», строже)
CROSSWALK_NEAR_AREA_RATIO = 0.30  # Переход занимает 30% кадра — считается «очень близко» (повышено)
CROSSWALK_NEAR_BOTTOM_RATIO = 0.80  # Нижний край перехода ниже 80% кадра — считается «очень близко» (повышено)
CROSSWALK_NEAR_MIN_HEIGHT_RATIO = 0.35  # Высота перехода более 35% кадра (новое условие)

# Параметры определения сигнала светофора
GREEN_LIGHT_STABLE_FRAMES = 5  # Число кадров стабильного зелёного

# Привязка ID классов (соответствует обучающему набору)
CW_ID = int(os.getenv("AIGLASS_SEG_CW_ID", "0"))  # пешеходный переход
BP_ID = int(os.getenv("AIGLASS_SEG_BP_ID", "1"))  # тактильная дорожка

# Наборы синонимов для пешеходного перехода и тактильной дорожки
_CW = {'zebra_crossing', 'zebra crossing', 'zebra', 'crosswalk', 'road_crossing', 'road crossing'}
_BP = {'blind_path', 'tactile_paving', 'tactile paving', 'blind path'}

# Пороги «проверки подлинности» тактильной дорожки
BP_VALID_IOU_THR = 0.40  # Если IoU с пешеходным переходом выше этого значения — считается «путаницей», не тактильная дорожка

# Параметры трекинга/расстановки точек
INNER_OFFSET_PX_LOCK = 5
EDGE_DILATE_PX = 2
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03)
)
FEATURE_PARAMS = dict(
    maxCorners=600,
    qualityLevel=0.001,
    minDistance=5,
    blockSize=7
)

# Временное сглаживание и удержание
MASK_EMA_ALPHA = 0.6   # Вес EMA-сглаживания
TRACK_MIN_POINTS = 30  # Минимальное число особых точек для трекинга
TRACK_RESEED_EVERY = 12  # Каждые N кадров при успешной сегментации заново расставлять особые точки

# Цвета визуализации (BGR)
VIS_COLORS = {
    "crosswalk": (0, 165, 255),   # оранжевый
    "centerline": (255, 255, 0),  # голубой — направляющая центральная линия
    "target_point": (255, 0, 255), # розовый — направляющая целевая точка
    "hint": (0, 255, 255),        # жёлтый
    "stripes": (0, 128, 255),     # оранжево-синий — отрезки полос
    "heading": (0, 0, 255),       # красный — стрелка направления
}

@dataclass
class CrossStreetResult:
    """Результат навигации при переходе дороги"""
    annotated_image: Optional[np.ndarray] = None
    guidance_text: str = ""
    visualizations: List[Dict[str, Any]] = None
    should_switch_to_blindpath: bool = False

    def __post_init__(self):
        if self.visualizations is None:
            self.visualizations = []

# ========== Вспомогательные функции ==========
def _score_of(d) -> float:
    """Извлечь уверенность из разных структур детекции; если не удалось — вернуть 0.0 (консервативно)"""
    for k in ("conf", "confidence", "score", "prob"):
        v = getattr(d, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                break
    return 0.0

def _norm_name(s: str) -> str:
    """Нормализовать имя"""
    return str(s).lower().replace('_', ' ').strip()

def _in_set(name: str, pool: set) -> bool:
    """Проверить, входит ли имя в набор"""
    return _norm_name(name) in {_norm_name(x) for x in pool}

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Вычислить IoU двух масок"""
    if a is None or b is None: 
        return 0.0
    ai = a > 0
    bi = b > 0
    inter = np.logical_and(ai, bi).sum()
    union = np.logical_or(ai, bi).sum()
    return float(inter) / float(union + 1e-6)

def _looks_like_blind_path(bp_mask: np.ndarray, cw_mask: np.ndarray, H: int, W: int) -> bool:
    """Геометрическая проверка + проверка на взаимоисключение, отсев ложных тактильных дорожек («поперечные полосы/бордюр»)"""
    if bp_mask is None: 
        return False
    ys, xs = np.where(bp_mask > 0)
    if xs.size < 80:  # Слишком маленькие фрагменты сразу отбрасываем
        return False

    # Вычислить угол главной оси
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    angle_deg = np.degrees(np.arctan2(v[1], v[0]))
    if angle_deg > 90: angle_deg -= 180
    if angle_deg < -90: angle_deg += 180
    
    h = (ys.max() - ys.min() + 1)
    w = (xs.max() - xs.min() + 1)
    aspect = h / float(w + 1e-6)  # Ожидаем, что тактильная дорожка «более вертикальная»
    iou_cw = _mask_iou(bp_mask, cw_mask)

    # 1) Отсев поперечных полос (ослаблено до 20°, больше запаса для дальнего края/лёгкого наклона)
    if abs(angle_deg) <= 20.0:
        return False
    # 2) Отсев по форме (ослаблено до 0.52)
    if aspect < 0.52:
        return False
    # 3) Сильное перекрытие с пешеходным переходом
    if iou_cw >= BP_VALID_IOU_THR:
        return False
    # 4) Отсев узкой полосы у нижнего края (вероятно, бордюр)
    bottom = bp_mask[int(0.88 * H):, :]
    if bottom.sum() > 0:
        bottom_share = bottom.sum() / float((bp_mask > 0).sum() + 1e-6)
        if bottom_share > 0.50 and (w / float(W)) < 0.35:
            return False
    return True

def _cls_of(d):
    """Извлечь ID класса объекта детекции"""
    for k in ("cls", "class_id", "category_id"):
        v = getattr(d, k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None

class CrossStreetNavigator:
    """Упрощённый навигатор перехода дороги — только детекция пешеходного перехода, но с навигацией (сегментация на каждом кадре + удержание оптическим потоком при неудаче)"""

    def __init__(self, seg_model=None, coco_model=None, obs_model=None, device_id: str = "esp32"):
        self.seg_model = seg_model
        self.device_id = device_id
        self.frame_counter = 0
        self.last_guidance = ""
        self.crosswalk_detected = False
        self.last_guide_time = 0
        self.guide_interval = 3.0  # Интервал голосовых подсказок (с)

        # —— Конечный автомат ——
        self.state = STATE_SEEKING           # Текущее состояние
        self.green_light_counter = 0         # Счётчик кадров стабильного зелёного
        self.last_traffic_light = None       # Светофор, обнаруженный на предыдущем кадре
        self.last_seeking_guidance = ""      # Текст подсказки последнего состояния SEEKING (для ограничения частоты)
        self.last_waiting_light_time = 0     # Время последнего озвучивания «ожидание зелёного»
        self.crossing_end_announced = False  # Озвучено ли «переход завершён» (для состояния CROSSING)
        self.last_crosswalk_seen_time = 0    # Время последнего обнаружения пешеходного перехода
        self.last_blindpath_announce_time = 0  # Время последней подсказки о тактильной дорожке (для ограничения повторов)

        # —— Состояние временной обработки/трекинга ——
        self.prev_mask = None            # Стабилизированная бинарная маска предыдущего кадра
        self.prev_mask_float = None      # Буфер EMA маски (float)
        self.prev_mask_ts = 0.0          # Время последнего обновления маски
        self.old_gray = None             # Полутоновое изображение предыдущего кадра (для LK)
        self.p0 = None                   # Особые точки предыдущего кадра (N,1,2)
        self.last_seed_frame = 0         # Номер кадра последней расстановки особых точек

        # —— Обход препятствий (как в blindpath) ——
        self.obstacle_detector = obs_model
        self.prev_gray = None
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.OBSTACLE_DETECTION_INTERVAL = int(os.getenv("AIGLASS_OBS_INTERVAL", "15"))
        self.OBSTACLE_CACHE_DURATION_FRAMES = int(os.getenv("AIGLASS_OBS_CACHE_FRAMES", "0"))
        
        # 【Новое】Настройка интервала детекции пешеходного перехода
        self.CROSSWALK_DETECTION_INTERVAL = int(os.getenv("AIGLASS_CROSSWALK_INTERVAL", "4"))  # Детекция раз в 4 кадра
        self.last_crosswalk_detection_frame = 0
        self.last_detected_crosswalk_mask = None
        self.last_detected_blindpath_mask = None

        # Автоматически включить детекцию препятствий (если obs_model не передан)
        if self.obstacle_detector is None and os.getenv("AIGLASS_OBS_AUTO", "1") != "0":
            try:
                if ObstacleDetectorClient is not None:
                    model_path = os.getenv("AIGLASS_OBS_MODEL", "model/yoloe-11l-seg.pt")
                    self.obstacle_detector = ObstacleDetectorClient(model_path)
                    logger.info("[CROSS_STREET] Детектор препятствий загружен автоматически")
                else:
                    logger.warning("[CROSS_STREET] ObstacleDetectorClient не найден, автозагрузка пропущена")
            except Exception as e:
                logger.warning(f"[CROSS_STREET] Не удалось автоматически загрузить детектор препятствий: {e}")

        # Если у модели есть метод predict, но нет detect — оборачиваем
        if self.seg_model and hasattr(self.seg_model, 'predict') and not hasattr(self.seg_model, 'detect'):
            logger.info("[CROSS_STREET] Оборачиваем модель YOLO")
            self.seg_model = YOLOModelWrapper(self.seg_model)
        
        # 【Новое】Вывести настройку интервала детекции
        logger.info(f"[CROSS_STREET] Интервал детекции пешеходного перехода: каждые {self.CROSSWALK_DETECTION_INTERVAL} кадр(а/ов)")

        # Убедиться, что модель на GPU
        if self.seg_model and torch.cuda.is_available():
            try:
                if hasattr(self.seg_model, 'model') and hasattr(self.seg_model.model, 'to'):
                    self.seg_model.model.to('cuda')
                elif hasattr(self.seg_model, 'to'):
                    self.seg_model.to('cuda')
                logger.info("[CROSS_STREET] Модель перенесена на GPU")
            except Exception as e:
                logger.warning(f"[CROSS_STREET] Не удалось перенести модель на GPU: {e}")

    def reset(self):
        """Сбросить состояние"""
        self.frame_counter = 0
        self.last_guidance = ""
        self.crosswalk_detected = False
        self.last_guide_time = 0
        # Конечный автомат
        self.state = STATE_SEEKING
        self.green_light_counter = 0
        self.last_traffic_light = None
        self.last_seeking_guidance = ""
        self.last_waiting_light_time = 0
        self.crossing_end_announced = False
        self.last_crosswalk_seen_time = 0
        self.last_blindpath_announce_time = 0
        # Трекинг
        self.prev_mask = None
        self.prev_mask_float = None
        self.prev_mask_ts = 0.0
        self.old_gray = None
        self.p0 = None
        self.last_seed_frame = 0
        # Кэш обхода препятствий
        self.prev_gray = None
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        # Сбросить состояние детекции светофора
        if TRAFFIC_LIGHT_AVAILABLE and trafficlight_detection:
            trafficlight_detection.reset_detection_state()
        logger.info("[CROSS_STREET] Навигатор сброшен")

    # —— Вспомогательное для расстановки точек/трекинга ——
    @staticmethod
    def _inner_offset_edge(mask_bin: np.ndarray, offset_px=5, edge_dilate_px=2) -> np.ndarray:
        """Сузить бинарную маску и выделить край, чтобы расставлять особые точки оптического потока внутри объекта"""
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

    @staticmethod
    def _hull_mask_from_points(points: np.ndarray, shape_hw: tuple) -> Optional[np.ndarray]:
        """Построить бинарную маску из выпуклой оболочки набора точек"""
        if points is None or len(points) < 3:
            return None
        H, W = shape_hw
        pts = points.reshape(-1, 2).astype(np.float32)
        hull = cv2.convexHull(pts.reshape(-1,1,2))
        poly = hull.reshape(-1, 2).astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        return mask

    def _seed_points_from_mask(self, gray: np.ndarray, mask_bin: np.ndarray) -> Optional[np.ndarray]:
        """Расставить особые точки LK-потока по суженной границе маски"""
        edge_mask = self._inner_offset_edge(mask_bin, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
        try:
            pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
            return pts
        except Exception as e:
            logger.warning(f"[CROSS_STREET] Ошибка goodFeaturesToTrack: {e}")
            return None

    @staticmethod
    def _ensure_binary_mask(mask: np.ndarray, shape_hw: tuple) -> np.ndarray:
        """Пороговая обработка и масштабирование до размера изображения, возвращает бинарную 0/1 uint8"""
        H, W = shape_hw
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def _postprocess_mask(self, mask_bin: np.ndarray) -> np.ndarray:
        """Морфологическая очистка + удаление мелких фрагментов, сглаживание рваных краёв и шума"""
        try:
            m = (mask_bin > 0).astype(np.uint8)
            H, W = m.shape[:2]
            # Лёгкие операции открытия/закрытия: убрать заусенцы и заполнить мелкие дыры
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)
            # Удалить слишком маленькие связные области
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                keep_area = max(int(0.003 * H * W), 1500)  # около 0.3% кадра или 1500 px
                keep_labels = np.where(areas >= keep_area)[0] + 1
                m2 = np.zeros_like(m)
                for lbl in keep_labels:
                    m2[labels == lbl] = 1
                if m2.sum() > 0:
                    m = m2
            return (m > 0).astype(np.uint8)
        except Exception:
            return (mask_bin > 0).astype(np.uint8)

    @staticmethod
    def _largest_contour(mask_bin: np.ndarray):
        cts, _ = cv2.findContours((mask_bin>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cts:
            return None
        return max(cts, key=cv2.contourArea)


    def _mask_center(self, mask: np.ndarray):
        """Вычислить центроид маски по моментам изображения; при неудаче вернуть None"""
        M = cv2.moments((mask > 0).astype(np.uint8))
        if abs(M["m00"]) < 1e-6:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def _is_crosswalk_near(self, mask: np.ndarray, h: int, w: int) -> bool:
        """Определить, «очень близко» ли пешеходный переход (прямо перед пользователем) — более строгие условия"""
        if mask is None:
            return False
        area = int(mask.sum())
        area_ratio = float(area) / float(h * w)
        
        # Получить положение нижнего края и высоту
        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return False
        top_y = int(ys.min())
        bottom_y = int(ys.max())
        mask_height = bottom_y - top_y + 1
        height_ratio = float(mask_height) / float(h)
        bottom_ratio = float(bottom_y) / float(h)
        
        # Должны одновременно выполняться несколько условий (логика AND, строже):
        # 1. Площадь достаточно большая
        # 2. Нижний край достаточно низко
        # 3. Доля высоты достаточно большая (защита от ложного срабатывания, когда пользователь просто поднял голову)
        is_near = (area_ratio >= CROSSWALK_NEAR_AREA_RATIO and 
                   bottom_ratio >= CROSSWALK_NEAR_BOTTOM_RATIO and
                   height_ratio >= CROSSWALK_NEAR_MIN_HEIGHT_RATIO)
        return is_near
    
    def _is_crosswalk_almost_done(self, mask: np.ndarray, h: int, w: int) -> bool:
        """Определить, «почти исчез» ли пешеходный переход (внизу кадра и с очень малой площадью) — более строгое условие"""
        if mask is None:
            return False
        area = int(mask.sum())
        area_ratio = float(area) / float(h * w)
        
        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return False
        
        # Вычислить положение верхнего и нижнего краёв пешеходного перехода
        top_y = int(ys.min())
        bottom_y = int(ys.max())
        
        top_ratio = float(top_y) / float(h)
        bottom_ratio = float(bottom_y) / float(h)
        
        # Более строгие условия (чтобы избежать преждевременного срабатывания):
        # 1. Верхний край уже ниже 70% кадра (>0.7) — переход в основном в самом низу кадра
        # 2. Нижний край близок к низу кадра (>0.85)
        # 3. Площадь очень мала (<0.08) — переход почти исчез
        is_almost_done = (top_ratio > 0.7 and bottom_ratio > 0.85 and area_ratio < 0.08)
        return is_almost_done
    
    def _compute_远_distance_alignment(self, mask: np.ndarray, h: int, w: int) -> tuple:
        """Вычислить угол и смещение для выравнивания на дальней дистанции (по геометрии маски, без опоры на полосы)"""
        ys, xs = np.where(mask > 0)
        if xs.size < 50:
            return 0.0, 0.0
        
        # Вычислить главное направление через PCA
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        eigvals, eigvecs = np.linalg.eig(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        
        # Вычислить угол (относительно горизонтали)
        angle = np.degrees(np.arctan2(v[1], v[0]))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180
        
        # Вычислить горизонтальное смещение (центроид относительно центра кадра)
        cx = float(mean[0])
        offset = (cx - (w / 2.0)) / max(1.0, w / 2.0)
        
        return float(angle), float(offset)

    def _draw_line_vertical_angle(self, image, center, angle_deg, length_ratio=0.7, color=(255, 255, 0), thickness=3):
        """
        За 0° принимается «вертикаль», angle_deg>0 — отклонение влево, <0 — вправо.
        Рисует прямую, проходящую через точку center.
        """
        H, W = image.shape[:2]
        half_len = int(0.5 * length_ratio * min(H, W))
        rad = np.radians(angle_deg)
        # Вертикальная база: единичный вектор вверх (0, -1)
        # Вектор направления после поворота на angle = (sin, -cos)
        vx = np.sin(rad);
        vy = -np.cos(rad)
        x0, y0 = center
        p1 = (int(x0 - vx * half_len), int(y0 - vy * half_len))
        p2 = (int(x0 + vx * half_len), int(y0 + vy * half_len))
        cv2.line(image, p1, p2, color, thickness)

    def _draw_dashed_line_vertical_angle(self, image, center, angle_deg, length_ratio=0.7,
                                         dash=12, gap=8, color=(255, 255, 255), thickness=2):
        """Так же с вертикалью за 0° рисует пунктир через center."""
        H, W = image.shape[:2]
        half_len = int(0.5 * length_ratio * min(H, W))
        rad = np.radians(angle_deg)
        vx = np.sin(rad);
        vy = -np.cos(rad)
        x0, y0 = center
        x1, y1 = int(x0 - vx * half_len), int(y0 - vy * half_len)
        x2, y2 = int(x0 + vx * half_len), int(y0 + vy * half_len)

        # Рисовать пунктир отрезками вдоль всей линии
        total_len = int(np.hypot(x2 - x1, y2 - y1))
        if total_len <= 0: return
        dx = (x2 - x1) / total_len
        dy = (y2 - y1) / total_len
        s = 0
        while s < total_len:
            e = min(s + dash, total_len)
            xa, ya = int(x1 + dx * s), int(y1 + dy * s)
            xb, yb = int(x1 + dx * e), int(y1 + dy * e)
            cv2.line(image, (xa, ya), (xb, yb), color, thickness)
            s += (dash + gap)

    def _offset_from_centerline(self, center_pt, angle_vertical_deg, width, height, y_ratio=0.75) -> float:
        """
        Вычисляет смещение влево/вправо по «голубой центральной линии-нормали»:
        - angle_vertical_deg: угол, где «вертикаль = 0°» (та же система координат, что в _draw_line_vertical_angle)
        - center_pt: центроид маски (cx, cy)
        - y_ratio: высота строки упреждения (доля высоты изображения), по умолчанию 0.75 (ниже — стабильнее)
        Возвращает: нормированное смещение (вправо — положительное, влево — отрицательное), по смыслу совпадает с прежним offset.
        """
        if center_pt is None:
            return 0.0
        x0, y0 = center_pt
        rad = np.radians(angle_vertical_deg)
        # Определение вектора направления полностью совпадает с _draw_line_vertical_angle
        vx = np.sin(rad)
        vy = -np.cos(rad)

        # Взять y строки упреждения
        y_target = float(int(height * y_ratio))

        # Если нормаль почти горизонтальна (бывает крайне редко), избегаем деления на 0
        if abs(vy) < 1e-6:
            x_at = float(x0)
        else:
            t = (y_target - float(y0)) / vy
            x_at = float(x0) + t * vx

        x_at = float(np.clip(x_at, 0, width - 1))
        # Как в старом определении offset: нормированное горизонтальное смещение относительно центра кадра (вправо +, влево −)
        return float((x_at - (width / 2.0)) / max(1.0, width / 2.0))

    def _compute_angle_and_offset(self, mask: np.ndarray) -> tuple:
        """Вычислить угол и смещение пешеходного перехода (запасной вариант через PCA)"""
        H, W = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if xs.size < 50:
            return 0.0, 0.0

        # Вычислить главное направление через PCA
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        eigvals, eigvecs = np.linalg.eig(cov)
        v = eigvecs[:, np.argmax(eigvals)]

        # Вычислить угол
        angle = np.degrees(np.arctan2(v[1], v[0]))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180

        # Вычислить горизонтальное смещение
        cx = float(mean[0])
        offset = (cx - (W / 2.0)) / max(1.0, W / 2.0)

        return float(angle), float(offset)

    def _estimate_angle_by_stripes(self, mask: np.ndarray, gray: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Оценка угла и визуализация по полосам внутри маски (линии Хафа) (ослабленные параметры + робастная кластеризация):
        Возвращает dict: {
          'angle_deg': float,       # угол отклонения от вертикали ([-45,45]), + = влево, − = вправо
          'lines': List[(x1,y1,x2,y2)],  # выбранные отрезки полос (координаты изображения)
          'confidence': float,      # [0,1] сила результирующего взвешенного кругового среднего
          'count': int              # число отрезков
        }
        """
        try:
            H, W = mask.shape[:2]
            roi_top = int(0.45 * H)  # Смотрим на нижнюю половину — стабильнее
            m_roi = (mask[roi_top:H, :] > 0).astype(np.uint8)
            g_roi = gray[roi_top:H, :]

            # Ослабленные пороги краёв
            g_blur = cv2.GaussianBlur(g_roi, (5, 5), 0)
            edges = cv2.Canny(g_blur, 50, 150)
            edges = cv2.bitwise_and(edges, edges, mask=m_roi * 255)

            # Ослабленные параметры Хафа
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=max(30, int(0.03 * W)),
                minLineLength=int(0.15 * W),
                maxLineGap=20
            )
            if lines is None:
                return None

            angles, weights = [], []
            all_lines = []
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx, dy = x2 - x1, y2 - y1
                length = float(np.hypot(dx, dy))
                if length < 8:
                    continue
                ang = float(np.degrees(np.arctan2(dy, dx)))  # относительно оси x
                if ang > 90: ang -= 180
                if ang < -90: ang += 180
                # Расширенный допустимый диапазон углов
                if abs(ang) > 65:
                    continue
                # Чем ближе к низу, тем больше вес
                ymid = (y1 + y2) * 0.5 + roi_top
                w = length * (0.5 + 0.5 * (ymid / max(1.0, H)))
                angles.append(ang)
                weights.append(w)
                all_lines.append((int(x1), int(y1 + roi_top), int(x2), int(y2 + roi_top)))

            if len(angles) < 5:
                return None

            # Робастная кластеризация углов: взвешенная медиана + отсев выбросов по MAD
            angs = np.array(angles, dtype=np.float32)
            wts = np.array(weights, dtype=np.float32)

            # Взвешенная медиана
            sort_idx = np.argsort(angs)
            angs_sorted = angs[sort_idx]
            wts_sorted = wts[sort_idx]
            cum = np.cumsum(wts_sorted)
            med_idx = np.searchsorted(cum, cum[-1] * 0.5)
            med = float(angs_sorted[min(max(med_idx, 0), len(angs_sorted) - 1)])

            # MAD (медиана абсолютных отклонений от медианы), порог шире
            dev = np.abs(angs - med)
            mad = float(np.median(dev) + 1e-6)
            deg_thr = max(12.0, 2.8 * mad)  # умеренно ослаблено
            keep = dev <= deg_thr

            if keep.sum() >= 3:
                angs_keep = angs[keep]
                wts_keep = wts[keep]
                lines_keep = [all_lines[i] for i, k in enumerate(keep) if k]
            else:
                angs_keep = angs
                wts_keep = wts
                lines_keep = all_lines

            # Взвешенное круговое среднее
            ang_rad = np.radians(angs_keep)
            C = float(np.sum(wts_keep * np.cos(ang_rad)))
            S = float(np.sum(wts_keep * np.sin(ang_rad)))
            norm = float(np.sum(wts_keep) + 1e-6)
            if abs(C) < 1e-6 and abs(S) < 1e-6:
                return None
            mean = float(np.degrees(np.arctan2(S, C)))
            confidence = float(np.hypot(C, S) / norm)

            return {
                "angle_deg": mean,
                "lines": lines_keep,
                "confidence": confidence,
                "count": len(lines_keep),
            }
        except Exception:
            return None

    def _get_crosswalk_guidance_features(self, mask: np.ndarray, image_shape: tuple) -> dict:
        """Вычислить признаки навигации по пешеходному переходу (робастная центральная линия + целевая точка + угол/смещение)"""
        try:
            height, width = image_shape[:2]
            min_run_px = max(12, int(width * 0.02))
            centerline_rows = []

            # Сканирование снизу вверх: берём середину между левой и правой границами наибольшего непрерывного участка, игнорируя разрозненный шум
            for y in range(height - 1, int(height * 0.4), -5):
                row = mask[y, :]
                xs = np.where(row > 0)[0]
                if xs.size <= min_run_px:
                    continue
                splits = np.where(np.diff(xs) > 1)[0] + 1
                segments = np.split(xs, splits) if xs.size else []
                if not segments:
                    continue
                seg = max(segments, key=lambda s: (s[-1] - s[0] + 1))
                if seg.size == 0 or (seg[-1] - seg[0] + 1) < min_run_px:
                    continue
                center_x = 0.5 * (seg[0] + seg[-1])
                centerline_rows.append([y, center_x])

            if len(centerline_rows) < 10:
                return None

            data = np.array(centerline_rows, dtype=np.float32)
            y_coords, x_coords = data[:, 0], data[:, 1]

            # Начальные веса (низ важнее)
            w_base = y_coords / float(height)
            coeffs = np.polyfit(y_coords, x_coords, 2, w=w_base)
            poly = np.poly1d(coeffs)

            # Одно робастное перевзвешивание (подавление изломов/выбросов)
            res = x_coords - poly(y_coords)
            mad = np.median(np.abs(res - np.median(res))) + 1e-6
            c = 2.5 * mad
            w_robust = 1.0 / (1.0 + (res / c) ** 2)
            w_total = w_base * w_robust
            coeffs = np.polyfit(y_coords, x_coords, 2, w=w_total)
            poly = np.poly1d(coeffs)

            # Целевая точка и точки отрисовки
            lookahead_y = int(height * 0.6)
            target_x = float(poly(lookahead_y))
            plot_y = np.arange(int(height * 0.4), height, 5).astype(int)
            plot_x = poly(plot_y).astype(int)
            centerline_points = np.vstack((plot_x, plot_y)).T.tolist()

            # Угол (по производной x(y)) и горизонтальное смещение
            dpoly = np.polyder(poly)
            dx_dy = float(dpoly(lookahead_y))
            angle_deg = float(np.degrees(np.arctan(dx_dy)))
            offset = float((target_x - (width / 2.0)) / max(1.0, width / 2.0))

            # Ограничить диапазон целевой точки
            tx = int(np.clip(target_x, 0, width - 1))
            return {
                "target_point": (tx, lookahead_y),
                "centerline_points": centerline_points,
                "angle_deg": angle_deg,
                "offset": offset,
            }
        except Exception:
            return None

    # —— Препятствия: вспомогательные методы оптического потока (как в blindpath) ——
    def _get_edge_mask(self, mask, offset=10):
        """Получить внутреннюю краевую область маски для поиска особых точек"""
        if mask is None:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset*2, offset*2))
        inner = cv2.erode(mask, kernel, iterations=1)
        edge = cv2.subtract(mask, inner)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.dilate(edge, kernel_small, iterations=1)
        return edge

    def _predict_mask_with_flow(self, prev_mask, prev_gray, curr_gray):
        """Предсказать положение маски оптическим потоком Лукаса — Канаде (как в blindpath)"""
        try:
            edge_mask = self._get_edge_mask(prev_mask, offset=10)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=edge_mask, **FEATURE_PARAMS)
            if p0 is None or len(p0) < 8:
                return None
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **LK_PARAMS)
            if p1 is None or st is None:
                return None
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) < 5:
                return None
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is None:
                return None
            H, W = curr_gray.shape[:2]
            flow_mask = cv2.warpAffine(prev_mask, M, (W, H),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
            return flow_mask
        except Exception:
            return None

    # —— Препятствия: детекция и визуализация (как в blindpath) ——
    def _detect_obstacles(self, image, path_mask=None):
        """Детекция препятствий через ObstacleDetectorClient.detect (синхронно с blindpath)"""
        logger.info(f"[_detect_obstacles] Запуск, Frame={self.frame_counter}, obstacle_detector={'загружен' if self.obstacle_detector else 'не загружен'}")
        if self.obstacle_detector is None:
            logger.warning("[_detect_obstacles] Детектор препятствий не загружен!")
            return []

        try:
            logger.info(f"[_detect_obstacles] Вызов ObstacleDetectorClient.detect()... image.shape={image.shape}")
            detected_obstacles = self.obstacle_detector.detect(image, path_mask=path_mask)
            logger.info(f"[_detect_obstacles] Возвращено объектов: {len(detected_obstacles)}")

            # Дополнить производные поля
            H, W = image.shape[:2]
            for i, obj in enumerate(detected_obstacles):
                if 'mask' in obj and obj['mask'] is not None:
                    y_coords, x_coords = np.where(obj['mask'] > 0)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                        obj['box_coords'] = (x1, y1, x2, y2)
                        if 'y_position_ratio' not in obj:
                            obj['y_position_ratio'] = obj.get('center_y', 0) / H
                        if 'label' not in obj:
                            obj['label'] = obj.get('name', 'unknown')
                        if 'center' not in obj:
                            obj['center'] = (obj.get('center_x', 0), obj.get('center_y', 0))
                        if 'confidence' not in obj:
                            obj['confidence'] = 0.5
            return detected_obstacles
        except Exception as e:
            logger.error(f"[_detect_obstacles] Ошибка детекции препятствий: {e}", exc_info=True)
            return []

    def _stabilize_obstacle_list(self, obstacles, prev_obstacles, prev_gray, curr_gray, image_shape, threshold=0.5):
        """Стабилизировать результаты детекции препятствий, избегая повторного наложения (как в blindpath)"""
        if not obstacles or prev_gray is None or curr_gray is None:
            return obstacles

        H, W = image_shape
        stabilized = []
        used_prev = set()
        for curr_obs in obstacles:
            if 'mask' not in curr_obs or curr_obs['mask'] is None:
                stabilized.append(curr_obs)
                continue
            curr_mask = curr_obs['mask']
            best_match = None
            best_iou = 0
            best_idx = -1

            if prev_obstacles:
                for idx, prev_obs in enumerate(prev_obstacles):
                    if idx in used_prev or 'mask' not in prev_obs:
                        continue
                    flow_mask = self._predict_mask_with_flow(prev_obs['mask'], prev_gray, curr_gray)
                    if flow_mask is None:
                        flow_mask = prev_obs['mask']
                    inter = np.logical_and(curr_mask > 0, flow_mask > 0).sum()
                    union = np.logical_or(curr_mask > 0, flow_mask > 0).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    if iou > best_iou and iou > threshold:
                        best_iou = iou
                        best_match = flow_mask
                        best_idx = idx

            if best_match is not None and best_idx >= 0:
                used_prev.add(best_idx)
                fused_mask = ((0.8 * curr_mask + 0.2 * best_match) > 128).astype(np.uint8) * 255
                curr_obs['mask'] = fused_mask
                self._update_obstacle_properties(curr_obs, H, W)
            stabilized.append(curr_obs)
        return stabilized

    def _update_obstacle_properties(self, obs, H, W):
        """Обновить производные атрибуты препятствий"""
        if 'mask' not in obs or obs['mask'] is None:
            return
        mask = obs['mask']
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            obs['area'] = int(len(y_coords))
            obs['center_x'] = float(np.mean(x_coords))
            obs['center_y'] = float(np.mean(y_coords))
            obs['y_position_ratio'] = obs['center_y'] / H
            obs['area_ratio'] = obs['area'] / float(H * W)
            obs['bottom_y_ratio'] = np.max(y_coords) / float(H)
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            obs['box_coords'] = (x1, y1, x2, y2)

    # —— Общие методы визуализации (как в blindpath) ——
    def _parse_color(self, color_str):
        """Разобрать строку цвета, вернуть в формате BGR"""
        try:
            if isinstance(color_str, tuple) and len(color_str) == 3:
                return color_str
            if color_str.startswith('rgba('):
                values = color_str[5:-1].split(',')
                r, g, b = int(values[0]), int(values[1]), int(values[2])
                return (b, g, r)  # OpenCV: BGR
            elif color_str == 'yellow':
                return (0, 255, 255)
            elif color_str == 'red':
                return (0, 0, 255)
            else:
                return (0, 0, 255)
        except:
            return (0, 0, 255)

    def _add_obstacle_visualization(self, obstacle, visualizations, pulse_effect=False):
        """Добавить визуализацию препятствий (упрощённо: только рамка, близкие — красные, дальние — жёлтые)"""
        try:
            bottom_y_ratio = obstacle.get('bottom_y_ratio', 0)
            area_ratio = obstacle.get('area_ratio', 0)
            is_near = bottom_y_ratio > 0.7 or area_ratio > 0.1

            if 'mask' in obstacle and obstacle['mask'] is not None:
                mask = obstacle['mask']
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_contour = max(contours, key=cv2.contourArea)
                    points = max_contour.squeeze(1)[::5].tolist()
                    
                    # Цвет рамки по расстоянию: близко — красный, далеко — жёлтый
                    if is_near:
                        outline_color = "rgba(255, 0, 0, 1.0)"  # красный
                        thickness = 3
                    else:
                        outline_color = "rgba(255, 255, 0, 0.8)"  # жёлтый
                        thickness = 2

                    # Только рамка, без заливки и текста
                    visualizations.append({
                        "type": "outline",
                        "points": points,
                        "color": outline_color,
                        "thickness": thickness
                    })
        except Exception as e:
            logger.error(f"[_add_obstacle_visualization] Не удалось добавить визуализацию препятствий: {e}")

    def _draw_command_button(self, image, text):
        """Нарисовать кнопку команды внизу по центру (в стиле yolomedia)"""
        try:
            H, W = image.shape[:2]
            full_text = f"当前指令：{text if text else '—'}"
            
            # Параметры кнопки
            font_px = 14
            pad_x, pad_y = 14, 8
            bottom_margin = 28
            
            # Вычислить размер текста
            if PIL_AVAILABLE:
                try:
                    from PIL import Image as PILImage, ImageDraw, ImageFont
                    # Попытаться загрузить китайский шрифт
                    font = None
                    for font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
                        if os.path.exists(font_path):
                            try:
                                font = ImageFont.truetype(font_path, font_px)
                                break
                            except:
                                continue
                    if font:
                        bbox = ImageDraw.Draw(PILImage.new('RGB', (1, 1))).textbbox((0, 0), full_text, font=font)
                        tw = max(1, bbox[2] - bbox[0])
                        th = max(1, bbox[3] - bbox[1])
                    else:
                        scale = font_px / 24.0
                        (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                except:
                    scale = font_px / 24.0
                    (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            else:
                scale = font_px / 24.0
                (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            
            # Вычислить положение кнопки (внизу по центру)
            bw = tw + pad_x * 2
            bh = th + pad_y * 2
            radius = max(10, bh // 2)
            
            cx = W // 2
            left = max(8, cx - bw // 2)
            top = H - bottom_margin - bh
            right = min(W - 8, left + bw)
            bottom = top + bh
            
            # Нарисовать полупрозрачный фон со скруглёнными углами
            overlay = image.copy()
            bg_color = (26, 32, 41)  # тёмный фон
            border_color = (60, 76, 102)  # рамка
            
            # Скруглённый прямоугольник (середина + два круга)
            cv2.rectangle(overlay, (left + radius, top), (right - radius, bottom), bg_color, -1)
            cv2.circle(overlay, (left + radius, (top + bottom) // 2), radius, bg_color, -1)
            cv2.circle(overlay, (right - radius, (top + bottom) // 2), radius, bg_color, -1)
            
            # Полупрозрачное смешивание
            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
            
            # Нарисовать рамку
            cv2.rectangle(image, (left + radius, top), (right - radius, bottom), border_color, 1)
            cv2.circle(image, (left + radius, (top + bottom) // 2), radius, border_color, 1)
            cv2.circle(image, (right - radius, (top + bottom) // 2), radius, border_color, 1)
            
            # Нарисовать текст
            text_x = left + pad_x
            text_y = top + pad_y + th
            
            if PIL_AVAILABLE and font:
                # Рисуем китайский текст через PIL
                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((text_x, top + pad_y), full_text, font=font, fill=(255, 255, 255))
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                # Рисуем через OpenCV
                cv2.putText(image, full_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
            
            return image
        except Exception as e:
            logger.error(f"Не удалось нарисовать кнопку команды: {e}")
            return image
    
    def _draw_data_panel_no_bg(self, image, data, position=(15, 15)):
        """Нарисовать панель данных (без чёрного фона, текст с обводкой), как в blindpath"""
        if not PIL_AVAILABLE:
            return image
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")
            env_scale = float(os.getenv("AIGLASS_PANEL_SCALE", "0.7"))
            base_font_size = max(10, int(round(14 * env_scale)))
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/System/Library/Fonts/PingFang.ttc",
            ]
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()

            y_offset = position[1]
            for key, value in data.items():
                text = f"{key}: {value}"
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((position[0] + dx, y_offset + dy), text,
                                      font=font, fill=(0, 0, 0, 255))
                draw.text((position[0], y_offset), text, font=font, fill=(255, 255, 255, 255))
                y_offset += base_font_size + 5
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"Не удалось нарисовать панель данных: {e}")
            return image

    def _draw_visualizations(self, image, viz_elements):
        """Расширенный метод отрисовки визуализаций (как в blindpath)"""
        if not viz_elements:
            return image
        current_time = time.time()
        panel_elements = [v for v in viz_elements if v.get("type") == "data_panel"]
        standard_elements = [v for v in viz_elements if v.get("type") != "data_panel"]

        # Первый проход: полупрозрачная заливка
        for element in standard_elements:
            elem_type = element.get("type")
            if elem_type in ['blind_path_mask', 'obstacle_mask', 'crosswalk_mask']:
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.5)"))
                    if element.get("effect") == "pulse":
                        pulse_speed = element.get("pulse_speed", 1.0)
                        alpha = 0.3 + 0.3 * np.sin(current_time * pulse_speed * 2 * np.pi)
                    else:
                        alpha = 0.4
                    x, y, w, h = cv2.boundingRect(points)
                    x = max(0, x); y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    if w > 0 and h > 0:
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        local_points = points - np.array([x, y])
                        cv2.fillPoly(binary_mask, [local_points], 255)
                        local_region = image[y:y+h, x:x+w].copy()
                        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                        color_overlay[:] = color
                        for c in range(3):
                            local_region[:, :, c] = np.where(
                                binary_mask > 0,
                                (1 - alpha) * local_region[:, :, c] + alpha * color_overlay[:, :, c],
                                local_region[:, :, c]
                            )
                        image[y:y+h, x:x+w] = local_region

        # Второй проход: контуры и элементы
        for element in standard_elements:
            elem_type = element.get("type")
            if elem_type == 'outline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                    thickness = element.get("thickness", 3)
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
            elif elem_type == 'polyline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 0, 1.0)"))
                    thickness = element.get("width", 2)
                    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)
            elif elem_type == 'circle':
                center = tuple(element.get("center", (0, 0)))
                radius = element.get("radius", 10)
                color = self._parse_color(element.get("color", "rgba(255, 0, 0, 1.0)"))
                thickness = -1 if element.get("filled", True) else 2
                cv2.circle(image, center, radius, color, thickness)
            elif elem_type == 'arrow':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.3)
                cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip_length)
            elif elem_type == 'text_with_bg':
                text = element.get("text", "")
                pos = element.get("position", [10, 30])
                font_scale = element.get("font_scale", 0.6)
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            cv2.putText(image, text, (pos[0] + dx, pos[1] + dy),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
                cv2.putText(image, text, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            elif elem_type == 'warning_icon':
                pos = element.get("position", (100, 100))
                level = element.get("level", "info")
                text = element.get("text", "")
                flash = element.get("flash", False)
                if level == "danger":
                    icon_color = (0, 0, 255)
                    text_color = (255, 255, 255)
                elif level == "warning":
                    icon_color = (0, 165, 255)
                    text_color = (255, 255, 255)
                else:
                    icon_color = (0, 255, 255)
                    text_color = (0, 0, 0)
                if flash:
                    alpha = 0.5 + 0.5 * np.sin(current_time * 4 * np.pi)
                    icon_color = tuple(int(c * alpha) for c in icon_color)
                triangle = np.array([
                    [pos[0], pos[1] - 20],
                    [pos[0] - 15, pos[1]],
                    [pos[0] + 15, pos[1]]
                ], np.int32)
                cv2.fillPoly(image, [triangle], icon_color)
                cv2.polylines(image, [triangle], True, (255, 255, 255), 2)
                cv2.putText(image, "!", (pos[0] - 5, pos[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if text:
                    font_scale = 0.5
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    text_pos = (pos[0] - tw // 2, pos[1] + 20)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(image, text, (text_pos[0] + dx, text_pos[1] + dy),
                                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
                    cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
            elif elem_type == 'text':
                text = element.get("text", "")
                pos = tuple(element.get("pos", (10, 30)))
                cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Панель данных
        if PIL_AVAILABLE:
            for panel in panel_elements:
                image = self._draw_data_panel_no_bg(image, panel["data"], panel["position"])
        else:
            for panel in panel_elements:
                y_offset = panel["position"][1]
                for key, value in panel["data"].items():
                    text = f"{key}: {value}"
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(image, text, (panel["position"][0] + dx, y_offset + dy),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(image, text, (panel["position"][0], y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25
        return image

    def _speech_for_obstacle(self, name: str) -> str:
        """Сформировать голосовую подсказку о препятствии"""
        k = (name or '').strip().lower()
        if k == 'person': return "前方有人，注意避让。"
        if k == 'car': return "前方有车，注意避让。"
        if k == 'bicycle': return "前方有自行车，停一下。"
        if k == 'motorcycle': return "前方有摩托车，停一下。"
        if k == 'bus': return "前方有公交车，停一下。"
        if k == 'truck': return "前方有卡车，停一下。"
        if k == 'scooter': return "前方有电瓶车，停一下。"
        if k == 'stroller': return "前方有婴儿车，停一下。"
        if k == 'dog': return "前方有狗，停一下。"
        if k == 'animal': return "前方有动物，停一下。"
        return "前方有障碍物，注意避让。"

    def process_frame(self, bgr_image: np.ndarray) -> CrossStreetResult:
        """Обработать один кадр (сегментация на каждом кадре; при неудаче — трекинг маски предыдущего кадра оптическим потоком для сохранения визуализации и навигации)"""
        self.frame_counter += 1
        current_time = time.time()

        try:
            annotated = bgr_image.copy()
            h, w = bgr_image.shape[:2]
            frame_visualizations = []

            # Текущее полутоновое изображение для LK и стабилизации обхода препятствий
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            # ========== 1) Сегментация с интервалом (детекция раз в 4 кадра) ==========
            crosswalk_mask = None
            blindpath_mask = None
            det_area = 0

            # 【Новое】Логика интервала детекции
            if self.seg_model and self.frame_counter % self.CROSSWALK_DETECTION_INTERVAL == 0:
                # Выполнить новую детекцию
                # Получить всех кандидатов с пониженным базовым порогом
                base_thr = min(CROSSWALK_MIN_CONF, BLIND_MIN_CONF)
                detections = self.seg_model.detect(bgr_image, confidence_threshold=base_thr) or []
                
                # Разобрать по ID класса и имени
                raw_cw, raw_bp = [], []
                for det in detections:
                    if not hasattr(det, 'mask') or det.mask is None:
                        continue
                    
                    cid = _cls_of(det)
                    name = str(getattr(det, "name", "")).lower()
                    
                    # Пешеходный переход: совпадение по ID или по имени
                    if (cid == CW_ID) or _in_set(name, _CW):
                        raw_cw.append(det)
                    # Тактильная дорожка: совпадение по ID или по имени
                    elif (cid == BP_ID) or _in_set(name, _BP):
                        raw_bp.append(det)
                
                # Вторичная пороговая фильтрация
                cw_list = [d for d in raw_cw if _score_of(d) >= CROSSWALK_MIN_CONF]
                bp_list = [d for d in raw_bp if _score_of(d) >= BLIND_MIN_CONF]
                
                # Объединить маски пешеходного перехода
                if cw_list:
                    cw_masks = []
                    for det in cw_list:
                        mask = det.mask
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bin = (mask > 0.5).astype(np.uint8)
                        cw_masks.append(mask_bin)
                    if cw_masks:
                        crosswalk_mask = np.maximum.reduce(cw_masks)
                        det_area = int(crosswalk_mask.sum())
                        if det_area < CROSSWALK_MIN_AREA:
                            crosswalk_mask = None
                            det_area = 0
                
                # Объединить маски тактильной дорожки
                if bp_list:
                    bp_masks = []
                    for det in bp_list:
                        mask = det.mask
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bin = (mask > 0.5).astype(np.uint8)
                        bp_masks.append(mask_bin)
                    if bp_masks:
                        blindpath_mask = np.maximum.reduce(bp_masks)
                
                # Устранение перекрытия: убрать область тактильной дорожки из маски перехода
                if crosswalk_mask is not None and blindpath_mask is not None:
                    crosswalk_mask = crosswalk_mask.copy()
                    crosswalk_mask[blindpath_mask > 0] = 0
                
                # Проверка подлинности тактильной дорожки
                if blindpath_mask is not None:
                    if not _looks_like_blind_path(blindpath_mask, crosswalk_mask, h, w):
                        blindpath_mask = None
                
                # 【Новое】Сохранить результаты детекции в кэш
                self.last_detected_crosswalk_mask = crosswalk_mask
                self.last_detected_blindpath_mask = blindpath_mask
                self.last_crosswalk_detection_frame = self.frame_counter
            
            else:
                # 【Новое】Использовать кэшированные результаты детекции
                crosswalk_mask = self.last_detected_crosswalk_mask
                blindpath_mask = self.last_detected_blindpath_mask

            # ========== 2) Сегментация не удалась → восстановление трекингом особых точек предыдущего кадра оптическим потоком ==========
            used_tracking = False
            if crosswalk_mask is None:
                if self.old_gray is not None and self.p0 is not None and len(self.p0) >= TRACK_MIN_POINTS:
                    try:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **LK_PARAMS)
                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            if good_new is not None and len(good_new) >= TRACK_MIN_POINTS:
                                tracked_mask = self._hull_mask_from_points(good_new, (h, w))
                                if tracked_mask is not None and int(tracked_mask.sum()) >= (0.3 * (self.prev_mask.sum() if self.prev_mask is not None else 1)):
                                    crosswalk_mask = tracked_mask
                                    used_tracking = True
                                    self.p0 = good_new.reshape(-1, 1, 2)
                                else:
                                    self.p0 = None
                                    self.old_gray = None
                    except Exception as e:
                        logger.warning(f"[CROSS_STREET] Ошибка оптического потока LK: {e}")
                        self.p0 = None
                        self.old_gray = None

            # ========== 3) EMA-сглаживание (меньше дрожания) + морфологическая очистка ==========
            if crosswalk_mask is not None:
                m = crosswalk_mask.astype(np.float32)
                if self.prev_mask_float is not None and self.prev_mask_float.shape == m.shape:
                    self.prev_mask_float = MASK_EMA_ALPHA * m + (1.0 - MASK_EMA_ALPHA) * self.prev_mask_float
                else:
                    self.prev_mask_float = m
                crosswalk_mask = (self.prev_mask_float > 0.5).astype(np.uint8)
                crosswalk_mask = self._postprocess_mask(crosswalk_mask)
                self.prev_mask = crosswalk_mask
                self.prev_mask_ts = current_time

            # ========== 4) Если сегментация (или трекинг) успешна → расставить/обновить особые точки ==========
            if crosswalk_mask is not None:
                need_seed = (self.p0 is None or len(self.p0) < TRACK_MIN_POINTS or
                             (self.frame_counter - self.last_seed_frame) >= TRACK_RESEED_EVERY)
                if need_seed:
                    pts = self._seed_points_from_mask(gray, crosswalk_mask)
                    if pts is not None and len(pts) >= TRACK_MIN_POINTS:
                        self.p0 = pts
                        self.old_gray = gray.copy()
                        self.last_seed_frame = self.frame_counter
                else:
                    self.old_gray = gray.copy()
            else:
                self.crosswalk_detected = False
                self.p0 = None
                self.old_gray = None

            # ========== 4.5) Детекция и визуализация препятствий (как в blindpath) ==========
            # Использовать crosswalk_mask как path_mask, если её нет — детекция по всему кадру
            detected_obstacles = []
            if self.obstacle_detector is not None:
                if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
                    detected_obstacles = self._detect_obstacles(bgr_image, path_mask=crosswalk_mask)
                    # Стабилизация
                    if self.prev_gray is not None:
                        detected_obstacles = self._stabilize_obstacle_list(
                            detected_obstacles,
                            self.last_detected_obstacles,
                            self.prev_gray,
                            gray,
                            bgr_image.shape[:2]
                        )
                    self.last_detected_obstacles = detected_obstacles
                    self.last_obstacle_detection_frame = self.frame_counter
                else:
                    if self.frame_counter - self.last_obstacle_detection_frame < self.OBSTACLE_CACHE_DURATION_FRAMES:
                        detected_obstacles = self.last_detected_obstacles
                    else:
                        detected_obstacles = []
                # Визуализировать все препятствия
                for obs in detected_obstacles:
                    self._add_obstacle_visualization(obs, frame_visualizations)

            # ========== 5) Конечный автомат + визуализация и навигационные команды ==========
            guidance_text = ""
            
            # Сначала рисуем тактильную дорожку (зелёная маска, без чёрного фона)
            if blindpath_mask is not None:
                # Смешиваем зелёный только в области маски, чтобы не было чёрного фона
                mask_area = (blindpath_mask > 0).astype(bool)
                green_color = np.array([0, 255, 0], dtype=np.float32)  # BGR
                # Смешиваем цвет внутри области маски
                for c in range(3):
                    annotated[:, :, c] = np.where(
                        mask_area,
                        (annotated[:, :, c] * 0.7 + green_color[c] * 0.3).astype(np.uint8),
                        annotated[:, :, c]
                    )
                # Нарисовать контур тактильной дорожки
                bp_ct = self._largest_contour(blindpath_mask)
                if bp_ct is not None:
                    cv2.drawContours(annotated, [bp_ct], -1, (0, 255, 0), 2)
            
            # Нарисовать пешеходный переход (оранжевая маска, без обводки, цвет как в режиме тактильной дорожки)
            if crosswalk_mask is not None:
                self.crosswalk_detected = True
                # Тот же оранжевый, что в режиме тактильной дорожки: BGR(0, 165, 255), смешивание только в области маски
                mask_area = (crosswalk_mask > 0).astype(bool)
                orange_color = np.array([0, 165, 255], dtype=np.float32)  # BGR
                # Смешиваем цвет внутри области маски
                for c in range(3):
                    annotated[:, :, c] = np.where(
                        mask_area,
                        (annotated[:, :, c] * 0.7 + orange_color[c] * 0.3).astype(np.uint8),
                        annotated[:, :, c]
                    )
            
            # ===== Логика конечного автомата =====
            if self.state == STATE_SEEKING:
                # Этап 1: поиск и выравнивание по удалённому пешеходному переходу
                if crosswalk_mask is not None:
                    is_near = self._is_crosswalk_near(crosswalk_mask, h, w)
                    
                    if is_near:
                        # Пешеходный переход уже рядом — переходим к определению сигнала светофора
                        self.state = STATE_WAIT_LIGHT
                        guidance_text = "斑马线已在跟前，进入红绿灯判定模式"
                        self.last_seeking_guidance = ""  # Сбросить состояние ограничения частоты
                    else:
                        # Подсказки выравнивания на дальней дистанции (с более мягкими порогами)
                        angle, offset = self._compute_远_distance_alignment(crosswalk_mask, h, w)
                        
                        # Сначала угол, затем положение (мягкие пороги, специально для SEEKING)
                        if abs(angle) >= SEEKING_ANGLE_THRESH_DEG:
                            direction = "左转一点" if angle > 0 else "右转一点"
                        elif abs(offset) >= SEEKING_OFFSET_THRESH:
                            direction = "向右平移" if offset > 0 else "向左平移"
                        else:
                            direction = "保持直行"
                        
                        # 【Текст в левом верхнем углу убран, вместо него панель данных в правом верхнем углу】
                        # Добавить панель данных в правом верхнем углу
                        frame_visualizations.append({
                            "type": "data_panel",
                            "data": {
                                "状态": "对准斑马线",
                                "角度": f"{angle:.1f}°",
                                "偏移": f"{offset:.2f}"
                            },
                            "position": (w - 180, 20)
                        })
                        
                        # Ограничение частоты: озвучивать только при смене текста подсказки или по истечении интервала
                        if current_time - self.last_guide_time > self.guide_interval:
                            if direction != self.last_seeking_guidance:
                                guidance_text = direction
                                self.last_seeking_guidance = direction
                            elif current_time - self.last_guide_time > self.guide_interval * 2:
                                # Прошло более 2 интервалов — повторное озвучивание
                                guidance_text = direction
                else:
                    # 【Текст в левом верхнем углу убран, вместо него панель данных в правом верхнем углу】
                    frame_visualizations.append({
                        "type": "data_panel",
                        "data": {
                            "状态": "寻找斑马线"
                        },
                        "position": (w - 180, 20)
                    })
                    self.last_seeking_guidance = ""  # Сброс, когда пешеходного перехода нет
            
            elif self.state == STATE_WAIT_LIGHT:
                # Этап 2: определение сигнала светофора
                # 【Текст в левом верхнем углу убран, панель данных в правом верхнем углу добавляется позже】
                
                if TRAFFIC_LIGHT_AVAILABLE and trafficlight_detection:
                    try:
                        # Передаём annotated (уже с переходом и тактильной дорожкой), детекция светофора добавляет поверх рамки
                        result = trafficlight_detection.process_single_frame(annotated)
                        
                        # Визуализировать результаты детекции светофора (нарисовать рамки)
                        if result and 'vis_image' in result:
                            vis_img = result['vis_image']
                            if vis_img is not None:
                                # Записать в annotated визуализацию детекции светофора (с переходом, тактильной дорожкой и рамками)
                                annotated = vis_img
                        
                        if result and 'stable_light' in result:
                            stable_light = result['stable_light']
                            
                            if stable_light == 'go':
                                self.green_light_counter += 1
                                # 【Текст в левом верхнем углу убран, вместо него панель данных в правом верхнем углу】
                                frame_visualizations.append({
                                    "type": "data_panel",
                                    "data": {
                                        "状态": "红绿灯判定",
                                        "检测": f"绿灯 {self.green_light_counter}/{GREEN_LIGHT_STABLE_FRAMES}"
                                    },
                                    "position": (w - 180, 20)
                                })
                                
                                if self.green_light_counter >= GREEN_LIGHT_STABLE_FRAMES:
                                    self.state = STATE_CROSSING
                                    guidance_text = "绿灯稳定，开始通行。"
                                    self.green_light_counter = 0
                                    self.crossing_end_announced = False      # Сбросить флаг завершения перехода
                                    self.last_crosswalk_seen_time = current_time  # Инициализировать время детекции пешеходного перехода
                                    self.last_blindpath_announce_time = 0    # Сбросить время озвучивания тактильной дорожки
                                else:
                                    # Зелёный обнаружен, но ещё нестабилен — озвучивание с ограничением частоты
                                    if current_time - self.last_waiting_light_time > 3.0:
                                        guidance_text = "正在等待绿灯…"
                                        self.last_waiting_light_time = current_time
                            else:
                                self.green_light_counter = 0
                                if stable_light in ['stop', 'countdown_stop']:
                                    # 【Текст в левом верхнем углу убран, вместо него панель данных в правом верхнем углу】
                                    frame_visualizations.append({
                                        "type": "data_panel",
                                        "data": {
                                            "状态": "红绿灯判定",
                                            "检测": "红灯，请等待"
                                        },
                                        "position": (w - 180, 20)
                                    })
                                    # Озвучивание красного сигнала (с ограничением частоты)
                                    if current_time - self.last_waiting_light_time > 3.0:
                                        guidance_text = "正在等待绿灯…"
                                        self.last_waiting_light_time = current_time
                                else:
                                    # Прочие состояния (жёлтый или не обнаружен) — озвучивание с ограничением частоты
                                    if current_time - self.last_waiting_light_time > 3.0:
                                        guidance_text = "正在等待绿灯…"
                                        self.last_waiting_light_time = current_time
                        else:
                            # Стабильный светофор не обнаружен — озвучивание с ограничением частоты
                            if current_time - self.last_waiting_light_time > 3.0:
                                guidance_text = "正在等待绿灯…"
                                self.last_waiting_light_time = current_time
                    except Exception as e:
                        logger.warning(f"[CROSS_STREET] Ошибка детекции светофора: {e}")
                        if current_time - self.last_waiting_light_time > 3.0:
                            guidance_text = "正在等待绿灯…"
                            self.last_waiting_light_time = current_time
                else:
                    # Модуля светофора нет — переключаемся сразу
                    # 【Текст в левом верхнем углу убран, вместо него панель данных в правом верхнем углу】
                    frame_visualizations.append({
                        "type": "data_panel",
                        "data": {
                            "状态": "红绿灯判定",
                            "检测": "模块未加载"
                        },
                        "position": (w - 180, 20)
                    })
                    if current_time - self.last_guide_time > 2.0:
                        self.state = STATE_CROSSING
                        guidance_text = "开始通行"
                        self.crossing_end_announced = False          # Сбросить флаг завершения перехода
                        self.last_crosswalk_seen_time = current_time # Инициализировать время детекции пешеходного перехода
                        self.last_blindpath_announce_time = 0        # Сбросить время озвучивания тактильной дорожки
            
            elif self.state == STATE_CROSSING:
                # Этап 3: подсказки при переходе дороги (исходная логика)
                
                # 【Новое】Детекция светофора в реальном времени (в состоянии CROSSING)
                traffic_light_warning = None  # Для хранения предупреждения о светофоре
                if TRAFFIC_LIGHT_AVAILABLE and trafficlight_detection:
                    try:
                        # Передаём annotated (уже с переходом и тактильной дорожкой), детекция светофора добавляет поверх рамки
                        result = trafficlight_detection.process_single_frame(annotated)
                        
                        # Записать в annotated визуализацию детекции светофора (с переходом, тактильной дорожкой и рамками)
                        if result and 'vis_image' in result:
                            vis_img = result['vis_image']
                            if vis_img is not None:
                                # Наложить рамки детекции светофора на annotated (сохраняя переход и тактильную дорожку)
                                annotated = vis_img
                        
                        # Проверить стабильное состояние; если идёт обратный отсчёт зелёного — озвучить предупреждение
                        if result and 'stable_light' in result:
                            stable_light = result['stable_light']
                            if stable_light == 'countdown_go':
                                # Обратный отсчёт зелёного — озвучить предупреждение (с ограничением частоты)
                                if current_time - self.last_guide_time > 2.0:
                                    traffic_light_warning = "绿灯快没了"
                    except Exception as e:
                        logger.warning(f"[CROSS_STREET] Ошибка детекции светофора в состоянии CROSSING: {e}")
                
                if crosswalk_mask is not None:
                    # Обновить время детекции пешеходного перехода
                    self.last_crosswalk_seen_time = current_time
                    
                    # Пешеходный переход обнаружен: если ранее ошибочно озвучено завершение — сбросить флаг и вернуться к обычному процессу
                    area = int(crosswalk_mask.sum())
                    area_ratio = float(area) / float(h * w)
                    # Если площадь перехода ещё довольно большая (>0.1), переход продолжается — сбросить флаг завершения
                    if area_ratio > 0.1 and self.crossing_end_announced:
                        self.crossing_end_announced = False
                        self.blindpath_announced = False
                        logger.info("[CROSS_STREET] Обнаружен пешеходный переход, флаг завершения сброшен, возврат к обычному процессу перехода")
                    
                    # 【Текст в левом верхнем углу убран, вместо него панель данных в правом верхнем углу】
                    panel_data = {
                        "状态": "正在过马路",
                        "面积": f"{area_ratio:.2f}"
                    }
                    if self.crossing_end_announced:
                        panel_data["提示"] = "已播报结束"
                    frame_visualizations.append({
                        "type": "data_panel",
                        "data": panel_data,
                        "position": (w - 180, 20)
                    })
                    
                    # Смещение выводится по «центральной линии-нормали к поперечным полосам перехода» (offset изначально 0, затем обновляется по голубой нормали)
                    angle_deg, offset = 0.0, 0.0

                    # Угол: сначала оценка по линиям Хафа на полосах; при неудаче — запасной вариант PCA
                    angle_source = "条纹"
                    stripes = self._estimate_angle_by_stripes(crosswalk_mask, gray)
                    if stripes and ("angle_deg" in stripes):
                        angle_deg = -float(stripes["angle_deg"])
                        for (x1, y1, x2, y2) in stripes.get("lines", []):
                            cv2.line(annotated, (x1, y1), (x2, y2), VIS_COLORS["stripes"], 2)
                        # Визуализировать стрелку направления (внизу по центру, отклонение от вертикали)
                        cx, cy = int(w * 0.5), int(h * 0.85)
                        length = int(60)
                        rad = np.radians(angle_deg)
                        dx = int(length * np.sin(rad))
                        dy = int(length * np.cos(rad))
                        cv2.arrowedLine(annotated, (cx, cy), (cx + dx, cy - dy), VIS_COLORS["heading"], 3, tipLength=0.25)
                    else:
                        angle_source = "PCA"
                        angle_deg, _ = self._compute_angle_and_offset(crosswalk_mask)


                    # === По центроиду маски + нормали к полосам рисуем «голубую центральную линию-нормаль» и «белый пунктир (сонаправлен с красной стрелкой)» ===
                    # === Две опорные линии через центр: голубая = нормаль, белый пунктир = сонаправлен с красной стрелкой ===
                    center_pt = self._mask_center(crosswalk_mask)
                    if center_pt is not None and stripes and ("angle_deg" in stripes):
                        # 1) Голубая нормаль: «средний угол полос» берётся как 【угол нормали относительно вертикали】, гарантируя перпендикулярность оранжевым полосам
                        angle_blue = float(stripes["angle_deg"])  # ← Важно: не менять знак и не прибавлять/вычитать 90°
                        self._draw_line_vertical_angle(annotated, center_pt, angle_blue,
                                                       length_ratio=0.7,
                                                       color=VIS_COLORS["centerline"], thickness=3)

                        # 2) Белый пунктир: «вертикаль кадра (0°)» через центроид — предполагаемое направление движения пользователя
                        angle_white = 0.0
                        self._draw_dashed_line_vertical_angle(annotated, center_pt, angle_white,
                                                              length_ratio=0.7,
                                                              dash=12, gap=8, color=(255, 255, 255), thickness=2)

                        # 3) Отображение разницы углов (опционально): голубая vs белый пунктир
                        diff = angle_blue - 0.0  # = angle_blue
                        diff = (diff + 180.0) % 360.0 - 180.0  # привести к [-180,180]
                        cv2.putText(annotated, f"{abs(diff):.1f}°",
                                    (min(center_pt[0] + 12, w - 110), max(center_pt[1] - 12, 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # === Вычисление «смещения влево/вправо» по голубой центральной линии-нормали ===
                        try:
                            # Внимание: система углов _offset_from_centerline совпадает с _draw_line_vertical_angle (вертикаль = 0°)
                            offset_new = self._offset_from_centerline(center_pt, angle_blue, w, h, y_ratio=0.75)
                            offset = float(offset_new)
                        except Exception:
                            # Запасной вариант: при ошибке вычисления оставить прежний offset (по умолчанию 0)
                            pass

                    # Направление навигации (базовое)
                    if abs(angle_deg) >= ANGLE_THRESH_DEG:
                        direction = "左转一点" if angle_deg > 0 else "右转一点"
                    elif abs(offset) >= OFFSET_THRESH:
                        direction = "向右平移" if offset > 0 else "向左平移"
                    else:
                        direction = "保持直行"

                    # Приоритет подсказок о препятствиях (близкие препятствия перекрывают подсказку направления)
                    obstacle_override = None
                    if detected_obstacles:
                        NEAR_Y = 0.7
                        NEAR_AREA = 0.1
                        near_list = [o for o in detected_obstacles if (o.get('bottom_y_ratio', 0) > NEAR_Y or o.get('area_ratio', 0) > NEAR_AREA)]
                        if near_list:
                            name = (near_list[0].get('name') or '障碍物')
                            obstacle_override = self._speech_for_obstacle(name)

                    # 【Отладочная информация в левом верхнем углу убрана, вместо неё панель данных в правом верхнем углу】
                    # Обновить панель данных в правом верхнем углу (объединить с уже имеющимися данными панели)
                    src_text = "分割" if not used_tracking else "追踪"
                    # Панель данных уже добавлена выше, здесь только записываются отладочные данные
                    # Полная панель данных будет добавлена позже единым образом

                    # Голосовой вывод (с ограничением частоты)
                    if current_time - self.last_guide_time > self.guide_interval:
                        # Проверить, почти ли пройден пешеходный переход
                        is_almost_done = self._is_crosswalk_almost_done(crosswalk_mask, h, w)
                        
                        # Отладочная информация: показать условия определения
                        if self.frame_counter % 30 == 0:
                            ys = np.where(crosswalk_mask > 0)[0]
                            if ys.size > 0:
                                top_y, bottom_y = int(ys.min()), int(ys.max())
                                logger.info(f"[CROSS_STREET] area_ratio={area_ratio:.3f}, top_ratio={top_y/h:.3f}, bottom_ratio={bottom_y/h:.3f}, almost_done={is_almost_done}")
                        
                        # Приоритет 1: предупреждение светофора (обратный отсчёт зелёного)
                        if traffic_light_warning:
                            guidance_text = traffic_light_warning
                            self.last_guide_time = current_time
                        # Приоритет 2: подсказка о завершении перехода (переход почти исчез)
                        elif is_almost_done and not self.crossing_end_announced:
                            guidance_text = "过马路结束，准备上人行道。"
                            self.crossing_end_announced = True
                            self.last_guide_time = current_time
                        # Приоритет 3: подсказка о тактильной дорожке (обнаружена после перехода; можно повторять, но не чаще раза в 4 с)
                        elif self.crossing_end_announced and blindpath_mask is not None:
                            if current_time - self.last_blindpath_announce_time > 4.0:
                                guidance_text = "远处有盲道，继续前行。"
                                self.last_blindpath_announce_time = current_time
                                self.last_guide_time = current_time
                        # Приоритет 4: препятствия
                        elif obstacle_override:
                            guidance_text = obstacle_override
                            self.last_guide_time = current_time
                        # Приоритет 5: подсказка направления
                        else:
                            guidance_text = direction
                            self.last_guide_time = current_time
                else:
                    # Этап CROSSING, но пешеходный переход не обнаружен
                    no_crosswalk_duration = current_time - self.last_crosswalk_seen_time
                    # 【Текст в левом верхнем углу убран, вместо него панель данных в правом верхнем углу】
                    frame_visualizations.append({
                        "type": "data_panel",
                        "data": {
                            "状态": "正在过马路",
                            "斑马线": f"未检测到 ({no_crosswalk_duration:.1f}s)"
                        },
                        "position": (w - 180, 20)
                    })
                    
                    # Только если перехода нет более 10 с подряд — озвучить «переход завершён»
                    if no_crosswalk_duration > 10.0:
                        if not self.crossing_end_announced:
                            if current_time - self.last_guide_time > self.guide_interval:
                                # Приоритет 1: предупреждение светофора
                                if traffic_light_warning:
                                    guidance_text = traffic_light_warning
                                    self.last_guide_time = current_time
                                # Приоритет 2: завершение перехода
                                else:
                                    guidance_text = "过马路结束，准备上人行道。"
                                    self.crossing_end_announced = True
                                    self.last_guide_time = current_time
                        # После озвучивания завершения, если обнаружена тактильная дорожка — повторять (не чаще раза в 4 с)
                        elif blindpath_mask is not None:
                            if current_time - self.last_blindpath_announce_time > 4.0:
                                guidance_text = "远处有盲道，继续前行。"
                                self.last_blindpath_announce_time = current_time
                                self.last_guide_time = current_time

            # 【Информация о кадре убрана】
            # Добавить кнопку команды внизу (показывает текущее состояние или подсказку)
            if guidance_text:
                current_instruction = guidance_text
            elif self.state == STATE_SEEKING:
                current_instruction = self.last_seeking_guidance if self.last_seeking_guidance else "寻找斑马线..."
            elif self.state == STATE_WAIT_LIGHT:
                current_instruction = "等待绿灯..."
            elif self.state == STATE_CROSSING:
                current_instruction = "过马路中..."
            else:
                current_instruction = "等待中..."
            annotated = self._draw_command_button(annotated, current_instruction)

            # Единая отрисовка слоёв визуализации препятствий и пр. (в стиле blindpath)
            if frame_visualizations:
                annotated = self._draw_visualizations(annotated, frame_visualizations)

            # 【Изменено】аудио не воспроизводится внутри рабочего процесса, этим централизованно занимается app_main
            # guidance_text напрямую возвращается вызывающему коду (app_main) для воспроизведения

            # Обновить prev_gray (для стабилизации препятствий)
            self.prev_gray = gray

            return CrossStreetResult(
                annotated_image=annotated,
                guidance_text=guidance_text,
                visualizations=frame_visualizations,
                should_switch_to_blindpath=False
            )

        except Exception as e:
            logger.error(f"[CROSS_STREET] Ошибка при обработке кадра: {e}", exc_info=True)
            return CrossStreetResult(
                annotated_image=bgr_image,
                guidance_text="",
                visualizations=[],
                should_switch_to_blindpath=False
            )

class YOLOModelWrapper:
    """Обёртка модели YOLO, адаптирующая метод predict под detect"""

    def __init__(self, yolo_model):
        self.model = yolo_model

    def detect(self, image, confidence_threshold=0.25):
        """Вызвать predict и преобразовать результат в формат detect"""
        try:
            results = self.model.predict(image, conf=confidence_threshold, verbose=False)
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    for i, mask in enumerate(result.masks.data):
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            cls = int(result.boxes.cls[i].cpu().numpy())
                            conf = float(result.boxes.conf[i].cpu().numpy())
                            class Detection:
                                def __init__(self):
                                    self.cls = cls
                                    self.conf = conf
                                    self.mask = mask.cpu().numpy()
                            detections.append(Detection())
            return detections
        except Exception as e:
            logger.error(f"[YOLO Wrapper] Ошибка детекции: {e}")
            return []