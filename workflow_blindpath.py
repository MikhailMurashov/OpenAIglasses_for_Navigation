# workflow_blindpath.py
# -*- coding: utf-8 -*-
"""
Рабочий процесс навигации по тактильной дорожке — чистая версия
Удалены все зависимости от Redis и Celery, можно напрямую интегрировать в любое Python-приложение
"""
import os
import time
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import torch  # Добавить эту строку
from obstacle_detector_client import ObstacleDetectorClient
from audio_player import play_voice_text  # Новое
from crosswalk_awareness import CrosswalkAwarenessMonitor, split_combined_voice  # Восприятие пешеходного перехода
# Пытаемся импортировать Pillow для отображения китайского текста
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageDraw, ImageFont = None, None, None

logger = logging.getLogger(__name__)

# ========== Определение констант состояний ==========
STATE_ONBOARDING = "ONBOARDING"
STATE_NAVIGATING = "NAVIGATING"
STATE_MANEUVERING_TURN = "MANEUVERING_TURN"
STATE_AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
STATE_LOCKING_ON = "LOCKING_ON"

# Подшаги ONBOARDING
ONBOARDING_STEP_ROTATION = "ROTATION"
ONBOARDING_STEP_TRANSLATION = "TRANSLATION"

# Подшаги поворота
MANEUVER_STEP_1_ISSUE_COMMAND = "ISSUE_COMMAND"
MANEUVER_STEP_2_WAIT_FOR_SHIFT = "WAIT_FOR_SHIFT"
MANEUVER_STEP_3_ALIGN_ON_NEW_PATH = "ALIGN_ON_NEW_PATH"

# Определение цветов (формат BGR)
VIS_COLORS = {
    "blind_path": (0, 255, 0),      # Зелёный
    "obstacle": (0, 0, 255),        # Красный
    "crosswalk": (0, 165, 255),     # Оранжевый
    "centerline": (0, 255, 255),     # Жёлтый
    "target_point": (255, 0, 0),     # Синий
    "turn_point": (128, 0, 128),     # Фиолетовый
    "pulse_effect": (100, 100, 255)  # Светло-красный
}

# Соответствие названий препятствий
_OBSTACLE_NAME_CN = {
    'person': '人',
    'bicycle': '自行车',
    'car': '车',
    'motorcycle': '摩托车',
    'bus': '公交车',
    'truck': '卡车',
    'animal': '动物',
    'scooter': '电瓶车',
    'stroller': '婴儿车',
    'dog': '狗',
}

# Список названий динамических классов
DYNAMIC_CLASS_NAMES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'animal', 'dog'}

@dataclass
class ProcessingResult:
    """Класс данных результата обработки"""
    guidance_text: str  # Текст голосовой подсказки
    visualizations: List[Dict[str, Any]]  # Список элементов визуализации
    annotated_image: Optional[np.ndarray] = None  # Аннотированное изображение
    state_info: Dict[str, Any] = None  # Информация о состоянии
    
    def __post_init__(self):
        if self.state_info is None:
            self.state_info = {}


class BlindPathNavigator:
    """Обработчик навигации по тактильной дорожке — версия без внешних зависимостей"""
    
    def __init__(self, yolo_model=None, obstacle_detector=None):
        """
        Инициализация навигатора
        :param yolo_model: модель сегментации YOLO (необязательно)
        :param obstacle_detector: детектор препятствий (необязательно)
        """
        self.yolo_model = yolo_model
        self.obstacle_detector = obstacle_detector
        
        # Переменные состояния
        self.current_state = STATE_ONBOARDING
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None
        

        # Параметры отслеживания оптического потока
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Параметры детекции признаков
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.05,
            minDistance=10,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04
        )
        
        # Кэш точек отслеживания оптического потока
        self.flow_points = {}  # {mask_type: points}
        self.flow_grace = {}   # {mask_type: grace_count}
        self.FLOW_GRACE_MAX = 3  # 【Изменено】Снижено с 8 до 3 кадров для быстрой очистки остатков оптического потока
        
        # Кэш сглаживания центральной линии
        self.centerline_history = []  # Исторические данные центральной линии
        self.centerline_history_max = 5  # Храним последние 5 кадров для сглаживания
        
        # Кэш сглаживания полиномиальных коэффициентов
        self.poly_coeffs_history = []  # Исторические полиномиальные коэффициенты
        self.poly_coeffs_history_max = 8  # Храним коэффициенты последних 8 кадров для сглаживания

        # Трекер обнаружения поворотов
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
        
        # Кулдаун поворота
        self.turn_cooldown_frames = 0
        self.TURN_COOLDOWN_DURATION = 50
        
        # Обход препятствий
        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None
        
        # Отслеживание пешеходного перехода
        self.crosswalk_tracker = {
            'stage': 'not_detected',
            'consecutive_frames': 0,
            'last_area_ratio': 0.0,
            'last_bottom_y_ratio': 0.0,
            'last_center_x_ratio': 0.5,
            'position_announced': False,
            'alignment_status': 'not_aligned',
            'last_seen_frame': 0,
            'last_angle': 0.0
        }
        
        # Счётчик кадров
        self.frame_counter = 0
        
        # Настройка подсказок «прямо» — поддерживаются переменные окружения
        self.guide_interval = float(os.getenv("AIGLASS_STRAIGHT_INTERVAL", "4.0"))  # Интервал озвучивания (сек)
        self.last_guide_time = 0.0
        self.straight_continuous_mode = os.getenv("AIGLASS_STRAIGHT_CONTINUOUS", "1") == "1"  # Режим непрерывного озвучивания
        self.straight_repeat_limit = int(os.getenv("AIGLASS_STRAIGHT_LIMIT", "2"))  # Максимальное число повторов в ограниченном режиме
        self.straight_repeat_count = 0
        
        # 【Новое】Настройка непрерывного озвучивания команд направления
        self.direction_interval = float(os.getenv("AIGLASS_DIRECTION_INTERVAL", "3.0"))  # Интервал команд направления (сек)
        self.last_direction_time = 0.0
        self.last_direction_message = ""
        
        # Вывод информации о конфигурации
        logger.info(f"[BlindPath] Настройка озвучивания «прямо»: интервал={self.guide_interval} с, "
                   f"непрерывный режим={self.straight_continuous_mode}, "
                   f"лимит повторов={self.straight_repeat_limit}")
        logger.info(f"[BlindPath] Настройка озвучивания направления: интервал={self.direction_interval} с")

        # Кэшируемые переменные
        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_any_speech_time = 0
        
        # Флаг состояния подготовки к пешеходному переходу
        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0
        
        # Голосовое сообщение о препятствии, ожидающее озвучивания
        self.pending_obstacle_voice = None
        
        # Детекция светофора
        self.traffic_light_detector = None
        self.init_traffic_light_detector()
        self.traffic_light_history = deque(maxlen=8)  # Для голосования большинством
        self.last_traffic_light_state = "unknown"
        self.green_light_announced = False
        
        # Настройка порогов
        self.CLASS_CONF_THRESHOLDS = {
            1: 0.20,  # blind_path
            0: 0.30   # crosswalk
        }
        
        # Пороги навигации
        # Пороги навигации
        self.ONBOARDING_ALIGN_THRESHOLD_RATIO = 0.1
        self.VP_FIT_ERROR_THRESHOLD = 8.0

        self.ONBOARDING_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.NAV_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.NAV_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.CURVATURE_PROXY_THRESHOLD = 5e-5
        
        # Пороги переключения на пешеходный переход
        self.CROSSWALK_SWITCH_AREA_RATIO = 0.22
        self.CROSSWALK_SWITCH_BOTTOM_RATIO = 0.9
        self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES = 10
        
        # Интервал детекции препятствий
        # Параметры оптимизации детекции препятствий — читаются из переменных окружения для настройки производительности
        self.OBSTACLE_DETECTION_INTERVAL = int(os.getenv("AIGLASS_OBS_INTERVAL", "15"))  # По умолчанию детекция раз в 5 кадров
        self.OBSTACLE_CACHE_DURATION_FRAMES = int(os.getenv("AIGLASS_OBS_CACHE_FRAMES", "10"))  # Кэш на 10 кадров
        
        # Управление озвучиванием препятствий
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.obstacle_speech_cooldown = 5.0  # Одно и то же препятствие не озвучивается повторно в течение 3 секунд
        
        # Параметры стабилизации маски (экстраполяция оптическим потоком отключена, эти параметры больше не используются)
        self.MASK_STAB_MIN_AREA = int(os.getenv("AIGLASS_MASK_MIN_AREA", "1500"))
        self.MASK_STAB_KERNEL = int(os.getenv("AIGLASS_MASK_MORPH", "3"))
        self.MASK_MISS_TTL = 0  # 【Изменено на 0】Экстраполяция оптическим потоком отключена, полностью в реальном времени
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0
        
        # Параметры отслеживания оптическим потоком
        self.flow_iou_threshold = 0.3  # При IoU ниже этого значения точки оптического потока переинициализируются
        
        # 【Новое】Интервал YOLO-детекции тактильной дорожки
        self.BLINDPATH_DETECTION_INTERVAL = int(os.getenv("AIGLASS_BLINDPATH_INTERVAL", "8"))  # Детекция раз в 2 кадра
        self.last_blindpath_detection_frame = 0
        self.last_blindpath_mask = None
        self.last_crosswalk_mask = None
        
        # 【Новое】Монитор восприятия пешеходного перехода
        self.crosswalk_monitor = CrosswalkAwarenessMonitor()
        logger.info("[BlindPath] Монитор восприятия пешеходного перехода инициализирован")
        logger.info(f"[BlindPath] Интервал детекции тактильной дорожки: каждые {self.BLINDPATH_DETECTION_INTERVAL} кадров")
    
    def init_traffic_light_detector(self):
        """Инициализация детектора светофора"""
        try:
            # Сначала пытаемся использовать модель YOLO для детекции светофора
            self.traffic_light_yolo = None
            # Если у вас есть специальная модель светофора, загрузите её здесь
            # self.traffic_light_yolo = YOLO('path/to/traffic_light_model.pt')
        except Exception as e:
            logger.info(f"Модель YOLO для светофора не загружена: {e}")
    
    def detect_traffic_light(self, image: np.ndarray) -> str:
        """Определение состояния светофора
        Возвращает: 'red', 'green', 'yellow', 'unknown'
        """
        # Режим симуляции (для тестирования)
        if os.getenv("AIGLASS_SIMULATE_TRAFFIC_LIGHT", "0") == "1":
            # Симулируем смену сигналов светофора по номеру кадра
            cycle = (self.frame_counter // 100) % 3
            if cycle == 0:
                return "red"
            elif cycle == 1:
                return "yellow"
            else:
                return "green"
        
        # Если есть модель YOLO, используем её в первую очередь
        if self.traffic_light_yolo:
            try:
                results = self.traffic_light_yolo.predict(image, verbose=False, conf=0.3)
                # TODO: разобрать результаты YOLO и определить цвет светофора
                pass
            except:
                pass
        
        # В качестве запасного варианта используем детекцию цвета в HSV
        return self._detect_traffic_light_by_color(image)
    
    def _detect_traffic_light_by_color(self, image: np.ndarray) -> str:
        """Детекция светофора на основе цветового пространства HSV"""
        h, w = image.shape[:2]
        # Проверяем верхнюю и среднюю часть изображения (светофор может быть на разной высоте)
        roi = image[:int(h * 0.7), :]  # Расширяем область детекции до 70%
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Изображение с повышенной яркостью для детекции (помогает находить тусклые светофоры)
        hsv_bright = hsv.copy()
        hsv_bright[:, :, 2] = cv2.add(hsv_bright[:, :, 2], 30)  # Увеличиваем яркость
        
        # Задаём диапазоны цветов (оптимизированные параметры)
        # Красный (два диапазона, так как красный в HSV пересекает 0 градусов)
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Зелёный (расширенный диапазон для разного освещения)
        lower_green = np.array([40, 60, 60])
        upper_green = np.array([90, 255, 255])
        
        # Жёлтый
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # Создаём маски (детекция одновременно на исходном и осветлённом изображении)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red1_bright = cv2.inRange(hsv_bright, lower_red1, upper_red1)
        mask_red2_bright = cv2.inRange(hsv_bright, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), 
                                 cv2.bitwise_or(mask_red1_bright, mask_red2_bright))
        
        mask_green = cv2.bitwise_or(cv2.inRange(hsv, lower_green, upper_green),
                                   cv2.inRange(hsv_bright, lower_green, upper_green))
        mask_yellow = cv2.bitwise_or(cv2.inRange(hsv, lower_yellow, upper_yellow),
                                    cv2.inRange(hsv_bright, lower_yellow, upper_yellow))
        
        # Морфологические операции для удаления шума
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        
        # Вычисляем площадь каждого цвета
        area_red = cv2.countNonZero(mask_red)
        area_green = cv2.countNonZero(mask_green)
        area_yellow = cv2.countNonZero(mask_yellow)
        
        # Задаём порог минимальной площади (снижаем порог для большей чувствительности)
        min_area = 30  # Ещё сильнее снижаем порог
        
        # Добавляем более подробную отладочную информацию
        if hasattr(self, 'frame_counter') and self.frame_counter % 30 == 0:
            logger.info(f"[HSV-детекция] красный:{area_red}, зелёный:{area_green}, жёлтый:{area_yellow}")
            # Сохраняем отладочное изображение
            if os.getenv("AIGLASS_DEBUG_TRAFFIC_LIGHT", "0") == "1":
                debug_dir = "traffic_light_debug"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_roi.jpg", roi)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_red.jpg", mask_red)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_green.jpg", mask_green)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_yellow.jpg", mask_yellow)
        
        # Определяем цвет (приоритет: зелёный > красный > жёлтый)
        if area_green > min_area and area_green > area_red * 0.8:  # Зелёный имеет приоритет
            return "green"
        elif area_red > min_area and area_red > area_green:
            return "red"
        elif area_yellow > min_area:
            return "yellow"
        else:
            return "unknown"
    
    def _get_voice_priority(self, guidance_text):
        """Получить приоритет голосовой команды
        Приоритет: препятствие (100) > поворот/смещение (50) > движение прямо (10)
        """
        if not guidance_text:
            return 0
        
        # Озвучивание препятствий — наивысший приоритет
        obstacle_keywords = ['前方有', '左侧有', '右侧有', '停一下', '注意避让', '障碍物']
        for keyword in obstacle_keywords:
            if keyword in guidance_text:
                return 100
        
        # Повороты и смещения — средний приоритет  
        direction_keywords = ['左转', '右转', '左移', '右移', '向左', '向右', '平移', '微调']
        for keyword in direction_keywords:
            if keyword in guidance_text:
                return 50
        
        # Движение прямо — наименьший приоритет
        if '保持直行' in guidance_text or '继续前进' in guidance_text or '方向正确' in guidance_text:
            return 10
        
        # Прочие команды — по умолчанию средний приоритет
        return 30

    def process_frame(self, image: np.ndarray) -> ProcessingResult:
        """
        Обработка одного кадра
        :param image: изображение в формате BGR
        :return: результат обработки
        """
        self.frame_counter += 1
        
        # Обновление периода кулдауна
        if self.turn_cooldown_frames > 0:
            self.turn_cooldown_frames -= 1
        
        image_height, image_width = image.shape[:2]
        image_center_x = image_width / 2
        
        # Преобразование в оттенки серого
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Список элементов визуализации
        frame_visualizations = []
        guidance_text = ""
        
        # 1. 【Изменено на детекцию в реальном времени】YOLO-детекция на каждом кадре, без кэша
        blind_path_mask, crosswalk_mask = self._detect_path_and_crosswalk(image)
        
        # 【Отладка】Проверка результатов YOLO-детекции
        if self.frame_counter % 30 == 0:  # Вывод раз в 30 кадров
            has_blind = blind_path_mask is not None and np.sum(blind_path_mask > 0) > 0
            has_cross = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(f"[YOLO-детекция] Frame={self.frame_counter}, дорожка={'есть' if has_blind else 'нет'}, "
                       f"переход={'есть' if has_cross else 'нет'}")
            if has_cross:
                cross_area = np.sum(crosswalk_mask > 0) / crosswalk_mask.size
                logger.info(f"[YOLO-детекция] Исходная площадь перехода: {cross_area*100:.2f}%")
        
        # 【Изменено】Сохраняем результат детекции перехода для восприятия пешеходного перехода
        # crosswalk_mask = None  # Больше не обнуляем принудительно
        
        # 2. 【Стабилизация маски и экстраполяция оптическим потоком отключены】Используем результаты детекции в реальном времени напрямую
        # Экстраполяция оптическим потоком больше не используется, полностью в реальном времени: есть — значит есть, нет — значит нет
        # blind_path_mask и crosswalk_mask напрямую берутся из результатов детекции выше
        crosswalk_mask_before_stabilize = crosswalk_mask
        
        # 【Отладка】Проверка результатов после стабилизации
        if self.frame_counter % 30 == 0 and crosswalk_mask_before_stabilize is not None:
            after_stab = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(f"[Стабилизация маски] Переход после стабилизации: {'есть' if after_stab else 'нет (отфильтрован)'}")
        
        # 【Новое】3. Детекция препятствий на всём протяжении
        # Детекция препятствий выполняется в любом состоянии
        logger.info(f"[Frame {self.frame_counter}] Начало детекции препятствий...")
        
        # Используем стратегию кэширования, но гарантируем визуализацию всех препятствий
        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image, blind_path_mask)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
            logger.info(f"[Frame {self.frame_counter}] Выполнена новая детекция препятствий, обнаружено препятствий: {len(detected_obstacles)}")
        else:
            if self.frame_counter - self.last_obstacle_detection_frame < self.OBSTACLE_CACHE_DURATION_FRAMES:
                detected_obstacles = self.last_detected_obstacles
                logger.info(f"[Frame {self.frame_counter}] Используются кэшированные данные о препятствиях, всего: {len(detected_obstacles)}")
            else:
                detected_obstacles = []
                logger.info(f"[Frame {self.frame_counter}] Кэш устарел, данных о препятствиях нет")
        
        # Добавляем визуализацию всех препятствий (не только близких)
        for i, obs in enumerate(detected_obstacles):
            logger.info(f"  Препятствие {i+1}: {obs.get('name', 'unknown')}, "
                    f"bottom_y_ratio={obs.get('bottom_y_ratio', 0):.2f}, "
                    f"area_ratio={obs.get('area_ratio', 0):.3f}, "
                    f"позиция=({obs.get('center_x', 0):.0f}, {obs.get('center_y', 0):.0f})")
            self._add_obstacle_visualization(obs, frame_visualizations)
        
        # 【Новое】Проверяем близкие препятствия и задаём голосовое сообщение
        self._check_and_set_obstacle_voice(detected_obstacles)
        
        # 【Новое】Обработка восприятия пешеходного перехода
        # Сначала проверяем состояние crosswalk_mask
        if crosswalk_mask is not None:
            cross_pixels = np.sum(crosswalk_mask > 0)
            if cross_pixels > 0:
                logger.info(f"[Переход] Передано в monitor: pixels={cross_pixels}, area={cross_pixels/crosswalk_mask.size*100:.2f}%")
            else:
                logger.info(f"[Переход] crosswalk_mask целиком нулевая, перехода нет")
        else:
            if self.frame_counter % 30 == 0:
                logger.info(f"[Переход] crosswalk_mask равна None")
        
        crosswalk_guidance = self.crosswalk_monitor.process_frame(crosswalk_mask, blind_path_mask)
        if crosswalk_guidance:
            logger.info(f"[Восприятие перехода] Результат детекции: area={crosswalk_guidance.get('area', 0):.3f}, "
                       f"should_broadcast={crosswalk_guidance.get('should_broadcast', False)}, "
                       f"voice={crosswalk_guidance.get('voice_text', 'None')}")
        if crosswalk_guidance and crosswalk_guidance['should_broadcast']:
            # Добавляем голосовое сообщение о переходе в список ожидания (через механизм pending)
            if not hasattr(self, 'pending_crosswalk_voice'):
                self.pending_crosswalk_voice = None
            self.pending_crosswalk_voice = crosswalk_guidance
            logger.info(f"[Голос перехода] Установлено сообщение для озвучивания: {crosswalk_guidance['voice_text']}, приоритет {crosswalk_guidance['priority']}")
        
        # 【Новое】Добавляем визуализацию пешеходного перехода
        if crosswalk_mask is not None:
            # Вычисляем данные визуализации
            total_pixels = crosswalk_mask.size
            crosswalk_pixels = np.sum(crosswalk_mask > 0)
            area_ratio = crosswalk_pixels / total_pixels
            
            y_coords, x_coords = np.where(crosswalk_mask > 0)
            if len(y_coords) > 0:
                center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
                center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]
                has_occlusion = self.crosswalk_monitor._check_occlusion(crosswalk_mask, blind_path_mask)
                
                # Получаем данные визуализации
                viz_data = self.crosswalk_monitor.get_visualization_data(
                    crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion
                )
                
                # Добавляем визуализацию маски перехода
                self._add_mask_visualization(crosswalk_mask, frame_visualizations, 
                                            "crosswalk_mask", viz_data['stage_color'])
                
                # Добавляем визуализацию информации о детекции перехода
                self._add_crosswalk_info_visualization(viz_data, image_height, image_width, 
                                                      frame_visualizations)
        
        # 【Отключено】4. Обновление трекера перехода — навигация по дорожке больше не переключается на переход
        # self._update_crosswalk_tracker(crosswalk_mask, image_height, image_width)
        
        # 5. Добавляем визуализацию пути
        # 【Восстановлено】Визуализация маски тактильной дорожки
        self._add_mask_visualization(blind_path_mask, frame_visualizations, "blind_path_mask", "rgba(0, 255, 0, 0.4)")
        # 【Визуализацию перехода выполняет crosswalk_monitor, здесь не добавляем】
        

        # 【Отключено】5. Разная логика навигации в зависимости от состояния — навигация по дорожке больше не обрабатывает переход
        current_stage = 'not_detected'  # Фиксированно: переход не детектируется
        # current_stage = self.crosswalk_tracker['stage']  # Отключено
        
        # Сразу выполняем навигацию по дорожке, не проверяя состояние перехода
        if False:  # current_stage == 'ready':
            # Проверяем, была ли уже озвучена подсказка о подготовке
            if not hasattr(self, 'crosswalk_ready_announced'):
                self.crosswalk_ready_announced = False
                self.crosswalk_ready_time = 0
            
            current_time = time.time()
            
            # Детекция светофора
            traffic_light_color = self.detect_traffic_light(image)
            self.traffic_light_history.append(traffic_light_color)
            
            # Отладочная информация
            if self.frame_counter % 30 == 0:  # Вывод раз в 30 кадров
                logger.info(f"[Детекция светофора] Текущий цвет: {traffic_light_color}, история: {list(self.traffic_light_history)}")
            
            # Голосование большинством для получения стабильного состояния светофора
            if len(self.traffic_light_history) >= 3:
                color_counts = {}
                for color in self.traffic_light_history:
                    color_counts[color] = color_counts.get(color, 0) + 1
                # Берём наиболее часто встречающийся цвет
                stable_color = max(color_counts.items(), key=lambda x: x[1])[0]
            else:
                stable_color = "unknown"
            
            # Добавляем визуализацию состояния светофора
            self._add_traffic_light_visualization(
                stable_color, frame_visualizations, image_height, image_width
            )
            
            # Решаем, что озвучить
            if not self.crosswalk_ready_announced:
                guidance_text = "已对准, 准备切换过马路模式。"
                self.crosswalk_ready_announced = True
                self.crosswalk_ready_time = current_time
            elif stable_color == "green" and not self.green_light_announced:
                guidance_text = "绿灯稳定，开始通行。"
                self.green_light_announced = True
            elif stable_color == "red":
                # При красном свете периодически напоминаем
                if current_time - self.crosswalk_ready_time > 5.0:
                    guidance_text = "正在等待绿灯…"
                    self.crosswalk_ready_time = current_time
                else:
                    guidance_text = ""
            else:
                guidance_text = ""
            
            # Добавляем информацию о состоянии
            frame_visualizations.append({
                "type": "data_panel",
                "data": {
                    "Состояние": "Ожидание перехода",
                    "Светофор": stable_color,
                    "История детекции": len(self.traffic_light_history)
                },
                "position": (25, image_height - 120)
            })
            
        elif False:  # current_stage == 'approaching':
            guidance_text = self._handle_crosswalk_approaching(
                frame_visualizations, image_height, image_width, image
            )
            
        # elif current_stage in ['far', 'not_detected']:
        else:  # Всегда выполняем навигацию по дорожке
            # 【Отключено】Подсказка о переходе
            # if current_stage == 'far' and not self.crosswalk_tracker['position_announced']:
            #     guidance_text = "远处发现斑马线，继续直行。"
            #     self.crosswalk_tracker['position_announced'] = True
            
            if blind_path_mask is None:
                guidance_text = ""
                # 【Убран текст в левом верхнем углу, заменён на панель данных в правом верхнем】
                frame_visualizations.append({
                    "type": "data_panel",
                    "data": {
                        "Состояние": "Ожидание распознавания дорожки"
                    },
                    "position": (image_width - 180, 20)
                })
            else:
                guidance_text = self._execute_state_machine(
                    blind_path_mask, image, frame_visualizations,
                    image_height, image_width, curr_gray
                )
        
        # 6. Обновление кэша
        self.prev_gray = curr_gray
        if blind_path_mask is not None:
            self.prev_blind_path_mask = blind_path_mask.copy()
        if crosswalk_mask is not None:
            self.prev_crosswalk_mask = crosswalk_mask.copy()
        
        # 【Улучшено】Система управления приоритетами голосовых сообщений
        current_time = time.time()
        
        # Собираем все возможные голосовые команды
        voice_candidates = []
        
        # 1. Добавляем основное навигационное сообщение
        if guidance_text:
            voice_candidates.append({
                'text': guidance_text,
                'priority': self._get_voice_priority(guidance_text),
                'source': 'navigation'
            })
        
        # 2. Проверяем, есть ли сообщение о препятствии (отдельная проверка, гарантирующая наивысший приоритет)
        if hasattr(self, 'pending_obstacle_voice'):
            if self.pending_obstacle_voice:
                voice_candidates.append({
                    'text': self.pending_obstacle_voice,
                    'priority': 100,  # Препятствие всегда имеет наивысший приоритет
                    'source': 'obstacle'
                })
                self.pending_obstacle_voice = None  # Очищаем обработанное сообщение о препятствии
        
        # 【Новое】Проверяем, есть ли сообщение о переходе
        if hasattr(self, 'pending_crosswalk_voice'):
            if self.pending_crosswalk_voice:
                voice_candidates.append({
                    'text': self.pending_crosswalk_voice['voice_text'],
                    'priority': self.pending_crosswalk_voice['priority'],
                    'source': 'crosswalk'
                })
                self.pending_crosswalk_voice = None  # Очищаем обработанное сообщение о переходе
        
        # 3. Выбираем сообщение с наивысшим приоритетом
        if voice_candidates:
            # Сортируем по приоритету и берём наивысший
            voice_candidates.sort(key=lambda x: x['priority'], reverse=True)
            selected_voice = voice_candidates[0]
            final_guidance_text = selected_voice['text']
            
            # Глобальный кулдаун озвучивания (чтобы сообщения не накладывались)
            MIN_SPEECH_INTERVAL = 1.2  # Интервал между любыми двумя сообщениями не менее 0.8 с
            if hasattr(self, 'last_any_speech_time'):
                if current_time - self.last_any_speech_time < MIN_SPEECH_INTERVAL:
                    final_guidance_text = ""  # Слишком быстро, пропускаем это озвучивание
            
            # Особое ограничение частоты для «прямо»
            if final_guidance_text == "保持直行":
                if self.straight_continuous_mode:
                    # Режим непрерывного озвучивания: проверяем только интервал времени
                    if current_time - self.last_guide_time >= self.guide_interval:
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
                else:
                    # Исходный ограниченный режим
                    if (current_time - self.last_guide_time >= self.guide_interval) and \
                       (self.straight_repeat_count < self.straight_repeat_limit):
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice['source'] != 'obstacle':
                # 【Изменено】Команды, не являющиеся «прямо» и препятствием — поддержка непрерывного озвучивания команд направления
                # Определяем, является ли это командой направления
                direction_keywords = ["左转", "右转", "左移", "右移", "向左", "向右", "平移", "微调"]
                is_direction = any(keyword in final_guidance_text for keyword in direction_keywords)
                
                if is_direction:
                    # Команда направления: поддерживается непрерывное озвучивание
                    if final_guidance_text == self.last_direction_message:
                        # Та же команда направления — проверяем интервал времени
                        if current_time - self.last_direction_time >= self.direction_interval:
                            self.last_direction_time = current_time
                            self.last_any_speech_time = current_time
                            self.straight_repeat_count = 0
                        else:
                            final_guidance_text = ""  # Интервал недостаточен, пропускаем
                    else:
                        # Новая команда направления — озвучиваем сразу
                        self.last_direction_message = final_guidance_text
                        self.last_direction_time = current_time
                        self.last_any_speech_time = current_time
                        self.straight_repeat_count = 0
                else:
                    # Прочие команды: озвучиваем только один раз
                    if final_guidance_text != self.last_guidance_message:
                        self.last_guidance_message = final_guidance_text
                        self.straight_repeat_count = 0
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice['source'] == 'obstacle':
                # Сообщение о препятствии озвучивается всегда
                self.last_any_speech_time = current_time
            elif final_guidance_text and selected_voice['source'] == 'crosswalk':
                # Сообщение о переходе озвучивается всегда (без проверки на повтор)
                self.last_any_speech_time = current_time
                
            # Озвучиваем выбранное сообщение
            if final_guidance_text:
                try:
                    # 【Оптимизация】У составного сообщения озвучиваем только первую часть, чтобы не копить очередь
                    if selected_voice.get('source') == 'crosswalk' and ',' in final_guidance_text:
                        voice_parts = split_combined_voice(final_guidance_text)
                        logger.info(f"[Голос перехода] В составном сообщении обнаружено частей: {len(voice_parts)}, озвучиваем только первую для работы в реальном времени")
                        # Озвучиваем только первую часть, остальные отбрасываем для сохранения работы в реальном времени
                        if voice_parts:
                            play_voice_text(voice_parts[0])
                            logger.info(f"[Озвучивание] Приоритет {selected_voice['priority']}: {voice_parts[0]}")
                    else:
                        play_voice_text(final_guidance_text)
                        logger.info(f"[Озвучивание] Приоритет {selected_voice['priority']}: {final_guidance_text}")
                except Exception as e:
                    logger.error(f"[Озвучивание] Ошибка воспроизведения: {e}")
        else:
            final_guidance_text = ""
        
        # 7. Формируем аннотированное изображение
        annotated_image = None

        if frame_visualizations:
            annotated_image = self._draw_visualizations(image.copy(), frame_visualizations)
        else:
            annotated_image = image.copy()
        
        # Добавляем кнопку команды внизу (показывает фактически озвученное сообщение)
        current_instruction = final_guidance_text if final_guidance_text else "Ожидание..."
        annotated_image = self._draw_command_button(annotated_image, current_instruction)
        
        # 8. Возвращаем результат
        return ProcessingResult(
            guidance_text=guidance_text,
            visualizations=frame_visualizations,
            annotated_image=annotated_image,
            state_info={
                "state": self.current_state,
                "crosswalk_stage": current_stage,
                "frame_count": self.frame_counter
            }
        )
    
    def _detect_path_and_crosswalk(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Детекция тактильной дорожки и пешеходного перехода"""
        if self.yolo_model is None:
            # 【Новое】При отсутствии модели возвращаем симулированные данные для тестирования
            logger.warning("Модель YOLO не загружена, возвращаются симулированные данные")
            h, w = image.shape[:2]
            # Создаём симулированную маску дорожки (вертикальная полоса по центру)
            blind_path_mask = np.zeros((h, w), dtype=np.uint8)
            # Создаём в центре изображения вертикальную полосу шириной 20% от ширины изображения
            strip_width = int(w * 0.2)
            strip_left = (w - strip_width) // 2
            blind_path_mask[int(h*0.3):, strip_left:strip_left+strip_width] = 255
            return blind_path_mask, None
        
        blind_path_mask = None
        crosswalk_mask = None
        
        try:
            min_conf = min(self.CLASS_CONF_THRESHOLDS.values())
            results = self.yolo_model.predict(image, verbose=False, conf=min_conf, classes=[0, 1])
            
            if (results and results[0] and results[0].masks is not None and 
                results[0].boxes is not None and len(results[0].masks.data) > 0):
                
                for mask_tensor, conf_tensor, cls_tensor in zip(
                    results[0].masks.data, results[0].boxes.conf, results[0].boxes.cls
                ):
                    class_id = int(cls_tensor.item())
                    confidence = float(conf_tensor.item())
                    threshold = self.CLASS_CONF_THRESHOLDS.get(class_id, 1.0)
                    
                    if confidence >= threshold:
                        current_mask = self._tensor_to_mask(mask_tensor, image.shape[1], image.shape[0])
                        
                        if class_id == 1:  # Тактильная дорожка
                            if blind_path_mask is None:
                                blind_path_mask = current_mask
                            else:
                                blind_path_mask = cv2.bitwise_or(blind_path_mask, current_mask)
                        elif class_id == 0:  # Пешеходный переход
                            if crosswalk_mask is None:
                                crosswalk_mask = current_mask
                            else:
                                crosswalk_mask = cv2.bitwise_or(crosswalk_mask, current_mask)
        except Exception as e:
            logger.error(f"Ошибка YOLO-детекции: {e}")
            # 【Новое】При ошибке детекции тоже возвращаем симулированные данные
            h, w = image.shape[:2]
            blind_path_mask = np.zeros((h, w), dtype=np.uint8)
            strip_width = int(w * 0.2)
            strip_left = (w - strip_width) // 2
            blind_path_mask[int(h*0.3):, strip_left:strip_left+strip_width] = 255
        
        return blind_path_mask, crosswalk_mask
    
    def _tensor_to_mask(self, mask_tensor, out_w: int, out_h: int, binarize: bool = True) -> np.ndarray:
        """Преобразование тензорной маски в массив numpy"""
        try:
            import torch
            
            if not isinstance(mask_tensor, torch.Tensor):
                arr = np.asarray(mask_tensor)
                if arr.dtype != np.uint8:
                    arr = (arr > 0.5).astype(np.uint8) * 255 if binarize else (arr * 255.0).astype(np.uint8)
                mask_u8 = arr
            else:
                if mask_tensor.dtype in (torch.bfloat16, torch.float16):
                    mask_tensor = mask_tensor.to(torch.float32)
                
                if mask_tensor.ndim > 2:
                    mask_tensor = mask_tensor.squeeze()
                
                if binarize:
                    mask_tensor = (mask_tensor > 0.5).to(torch.uint8).mul_(255)
                    mask_u8 = mask_tensor.cpu().numpy()
                else:
                    mask_u8 = (mask_tensor.mul(255).clamp_(0, 255).to(torch.uint8)).cpu().numpy()
            
            if mask_u8.ndim == 3:
                mask_u8 = mask_u8.squeeze(-1)
            
            if mask_u8.shape[1] != out_w or mask_u8.shape[0] != out_h:
                mask_u8 = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            
            return mask_u8
        except ImportError:
            # Если torch нет, возвращаем пустую маску
            return np.zeros((out_h, out_w), dtype=np.uint8)
    
    def _stabilize_mask(self, prev_gray, curr_gray, raw_mask, prev_stable_mask, mask_type):
        """Стабилизация маски — с помощью оптического потока Лукаса — Канаде"""
        if mask_type == 'blind_path':
            ttl = self.blind_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA
        else:  # crosswalk
            ttl = self.cross_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA
        
        # Вызываем новый метод стабилизации оптическим потоком
        stable_mask = self._stabilize_seg_mask(
            prev_gray, curr_gray, raw_mask, prev_stable_mask,
            (curr_gray.shape[1], curr_gray.shape[0]) if curr_gray is not None else (640, 480),
            min_area_px=min_area,
            morph_kernel=self.MASK_STAB_KERNEL,
            mask_type=mask_type
        )
        
        if stable_mask is not None:
            # Сбрасываем TTL
            if mask_type == 'blind_path':
                self.blind_miss_ttl = self.MASK_MISS_TTL
            else:
                self.cross_miss_ttl = self.MASK_MISS_TTL
            return stable_mask
        else:
            # Уменьшаем TTL
            if mask_type == 'blind_path':
                self.blind_miss_ttl = max(0, self.blind_miss_ttl - 1)
            else:
                self.cross_miss_ttl = max(0, self.cross_miss_ttl - 1)
            return None
    
    def _stabilize_seg_mask(self, prev_gray, curr_gray, curr_mask, prev_stable_mask, 
                          image_wh, min_area_px=1500, morph_kernel=3, iou_high_thr=0.4, mask_type='', 
                          fast_clear=True):
        """Реализация стабилизации маски с помощью оптического потока Лукаса — Канаде"""
        W, H = image_wh
        
        def _binarize(mask):
            if mask is None:
                return None
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        
        def _morph_smooth(mask, kernel_size):
            if mask is None:
                return None
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (max(1, kernel_size), max(1, kernel_size)))
            sm = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
            sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, k, iterations=1)
            return sm
        
        curr_mask_b = _binarize(curr_mask)
        prev_mask_b = _binarize(prev_stable_mask)
        
        # Если исторических данных нет, сразу возвращаем текущую маску
        if prev_mask_b is None or prev_gray is None or curr_gray is None:
            return _morph_smooth(curr_mask_b, morph_kernel) if curr_mask_b is not None else None
        
        # На текущем кадре есть результат детекции
        if curr_mask_b is not None and np.sum(curr_mask_b > 0) >= min_area_px:
            # Вычисляем IoU с предыдущим кадром
            if prev_mask_b is not None:
                inter = np.logical_and(curr_mask_b > 0, prev_mask_b > 0).sum()
                union = np.logical_or(curr_mask_b > 0, prev_mask_b > 0).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0
                
                # IoU достаточно высокий — детекция стабильна, используем текущий результат напрямую
                if iou >= iou_high_thr:
                    return _morph_smooth(curr_mask_b, morph_kernel)
                
                # IoU низкий, но перекрытие есть — выполняем взвешенное слияние
                elif iou > 0.1:
                    # Используем маску, предсказанную оптическим потоком
                    flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                    if flow_mask is not None:
                        # Динамически подстраиваем веса по IoU
                        # Чем ниже IoU, тем больше опора на оптический поток; чем выше IoU, тем больше опора на текущую детекцию
                        w_curr = min(0.9, 0.4 + iou)  # При IoU=0.1 w_curr=0.5, при IoU=0.5 w_curr=0.9
                        w_flow = 1.0 - w_curr
                        
                        fused = (w_curr * curr_mask_b.astype(np.float32) + 
                                w_flow * flow_mask.astype(np.float32))
                        fused_bin = (fused >= 128).astype(np.uint8) * 255
                        
                        # Переинициализируем точки оптического потока (если IoU слишком низкий)
                        if iou < self.flow_iou_threshold:
                            self.flow_points['blind_path'] = None
                        
                        return _morph_smooth(fused_bin, morph_kernel)
            
            # Нет истории или IoU слишком низкий — используем текущую детекцию
            return _morph_smooth(curr_mask_b, morph_kernel)
        
        # На текущем кадре нет результата детекции — пытаемся экстраполировать оптическим потоком
        else:
            # Получаем соответствующий TTL
            if mask_type == 'blind_path':
                ttl = self.blind_miss_ttl
            else:
                ttl = self.cross_miss_ttl
            
            # 【Изменено】На текущем кадре нет результата детекции — быстро очищаем
            if fast_clear and ttl <= 1:
                # TTL исчерпан, сразу возвращаем None без оптического потока
                return None
                
            if prev_mask_b is not None and np.sum(prev_mask_b > 0) >= min_area_px and ttl > 0:
                # Используем предсказание оптическим потоком
                flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                if flow_mask is not None and np.sum(flow_mask > 0) >= min_area_px * 0.5:
                    return _morph_smooth(flow_mask, morph_kernel)
            
            # Оптический поток не сработал или TTL превышен
            return None
    
    def _predict_mask_with_flow(self, prev_mask, prev_gray, curr_gray):
        """Предсказание положения маски оптическим потоком Лукаса — Канаде (улучшенная версия)"""
        try:
            # Способ 1: пробуем метод выпуклой оболочки (по образцу yolomedia)
            if hasattr(self, 'flow_points') and 'blind_path' in self.flow_points:
                p0 = self.flow_points['blind_path']
                if p0 is not None and len(p0) >= 5:
                    # Вычисляем оптический поток
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
                    
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:
                            # Обновляем точки оптического потока
                            self.flow_points['blind_path'] = good_new.reshape(-1, 1, 2)
                            
                            # Строим маску по выпуклой оболочке
                            hull = cv2.convexHull(good_new.reshape(-1, 1, 2))
                            poly = hull.reshape(-1, 2)
                            
                            if len(poly) >= 3:
                                H, W = curr_gray.shape[:2]
                                flow_mask = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(flow_mask, [poly.astype(np.int32)], 255)
                                return flow_mask
            
            # Способ 2: метод краевых точек-признаков (исходный, как запасной)
            edge_mask = self._get_edge_mask(prev_mask, offset=10)
            
            # Детекция точек-признаков
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=edge_mask, **self.feature_params)
            if p0 is None or len(p0) < 8:
                return None
            
            # Сохраняем точки-признаки для следующего раза
            self.flow_points['blind_path'] = p0
            
            # Вычисляем оптический поток
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
            
            if p1 is None or st is None:
                return None
            
            # Оставляем только успешно отслеженные точки
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) < 5:
                return None
            
            # Оцениваем матрицу преобразования (RANSAC для повышения устойчивости)
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            
            if M is None:
                return None
            
            # Применяем преобразование
            H, W = curr_gray.shape[:2]
            flow_mask = cv2.warpAffine(prev_mask, M, (W, H), 
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
            
            return flow_mask
            
        except Exception as e:
            logger.debug(f"Ошибка предсказания оптическим потоком: {e}")
            return None
            
    
    def _get_edge_mask(self, mask, offset=10):
        """Получение внутренней краевой области маски для детекции точек-признаков"""
        if mask is None:
            return None
        
        # Эрозией получаем внутреннюю маску
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset*2, offset*2))
        inner = cv2.erode(mask, kernel, iterations=1)
        
        # Край = исходная − внутренняя
        edge = cv2.subtract(mask, inner)
        
        # Слегка расширяем краевую область
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.dilate(edge, kernel_small, iterations=1)
        
        return edge

    def _smooth_centerline(self, centerline_data):
        """Сглаживание данных центральной линии для уменьшения дрожания"""
        if centerline_data is None or len(centerline_data) < 5:
            return centerline_data
        
        # Сохраняем в историю
        self.centerline_history.append(centerline_data.copy())
        if len(self.centerline_history) > self.centerline_history_max:
            self.centerline_history.pop(0)
        
        # Если истории недостаточно, возвращаем слегка сглаженные данные текущего кадра
        if len(self.centerline_history) < 3:
            # Пространственное сглаживание текущего кадра
            smoothed_data = centerline_data.copy()
            # Усреднение скользящим окном
            window_size = 5
            for i in range(len(smoothed_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(smoothed_data), i + window_size // 2 + 1)
                window = smoothed_data[start_idx:end_idx]
                if len(window) > 0:
                    smoothed_data[i, 1] = np.mean(window[:, 1])  # Сглаживаем координату x
                    smoothed_data[i, 2] = np.mean(window[:, 2])  # Сглаживаем ширину
            return smoothed_data
        
        # Временное сглаживание: взвешенное среднее по историческим кадрам
        smoothed_data = centerline_data.copy()
        
        # Для каждой координаты y находим соответствующие данные в исторических кадрах
        for i, (y, x, width) in enumerate(centerline_data):
            x_values = [x]
            width_values = [width]
            weights = [1.0]  # Текущий кадр имеет наибольший вес
            
            # Ищем в исторических кадрах данные с близкой координатой y
            for hist_idx, hist_data in enumerate(self.centerline_history[-3:-1]):  # Используем 2 последних исторических кадра
                # Находим ближайшую координату y
                y_diffs = np.abs(hist_data[:, 0] - y)
                if len(y_diffs) > 0:
                    closest_idx = np.argmin(y_diffs)
                    if y_diffs[closest_idx] < 10:  # Разница по y меньше 10 пикселей
                        x_values.append(hist_data[closest_idx, 1])
                        width_values.append(hist_data[closest_idx, 2])
                        # Вес исторических кадров убывает
                        weights.append(0.5 ** (len(self.centerline_history) - hist_idx - 1))
            
            # Взвешенное среднее
            if len(x_values) > 1:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                smoothed_data[i, 1] = np.sum(np.array(x_values) * weights)
                smoothed_data[i, 2] = np.sum(np.array(width_values) * weights)
        
        # Пространственное сглаживание: ещё раз усредняем результат скользящим окном
        window_size = 3
        final_data = smoothed_data.copy()
        for i in range(len(final_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(final_data), i + window_size // 2 + 1)
            window = smoothed_data[start_idx:end_idx]
            if len(window) > 0:
                final_data[i, 1] = np.mean(window[:, 1])
                final_data[i, 2] = np.mean(window[:, 2])
        
        return final_data

    def _estimate_affine(self, prev_gray, curr_gray, mask=None):
        """Оценка аффинного преобразования оптическим потоком (запасной метод)"""
        try:
            # Извлекаем точки-признаки
            if mask is not None:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
            else:
                p0 = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
            
            if p0 is None or len(p0) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # Вычисляем оптический поток
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
            
            if p1 is None or st is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # Оставляем только хорошие точки
            good_new = p1[st == 1].reshape(-1, 2)
            good_old = p0[st == 1].reshape(-1, 2)
            
            if len(good_new) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # Оцениваем аффинное преобразование
            M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
            
            if M is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            return M
            
        except Exception as e:
            logger.debug(f"Ошибка оценки аффинного преобразования: {e}")
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    def _warp_mask(self, mask, M, output_shape):
        """Применение аффинного преобразования"""
        try:
            W, H = output_shape
            warped = cv2.warpAffine(mask, M, (W, H), 
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            return warped
        except:
            return None
    
    def _add_mask_visualization(self, mask, visualizations, viz_type, color, add_outline=True):
        """Добавление визуализации маски (с обводкой)"""
        if mask is None:
            return
        
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                points = main_contour.squeeze(1)[::5].tolist()
                
                # Добавляем заливку
                visualizations.append({
                    "type": viz_type,
                    "points": points,
                    "color": color
                })
                
                # Добавляем обводку (для тактильной дорожки обводка не добавляется)
                if add_outline and viz_type != "blind_path_mask":
                    visualizations.append({
                        "type": "outline",
                        "points": points,
                        "color": "rgba(255, 255, 255, 0.8)",  # Белая обводка
                        "thickness": 3
                    })
        except:
            pass

    
    def _update_crosswalk_tracker(self, crosswalk_mask, image_height, image_width):
        """Обновление трекера пешеходного перехода"""
        if crosswalk_mask is not None:
            self.crosswalk_tracker['consecutive_frames'] += 1
            self.crosswalk_tracker['last_seen_frame'] = self.frame_counter
            
            # Вычисляем ключевые показатели
            total_area = image_height * image_width
            area_ratio = np.sum(crosswalk_mask > 0) / total_area
            y_coords, x_coords = np.where(crosswalk_mask > 0)
            
            if len(y_coords) > 0:
                bottom_y_ratio = np.max(y_coords) / image_height
                center_x_ratio = np.mean(x_coords) / image_width
                
                self.crosswalk_tracker['last_area_ratio'] = area_ratio
                self.crosswalk_tracker['last_bottom_y_ratio'] = bottom_y_ratio
                self.crosswalk_tracker['last_center_x_ratio'] = center_x_ratio
                
                # Вычисляем угол
                try:
                    contours, _ = cv2.findContours(crosswalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(main_contour)
                        angle = rect[-1]
                        w, h = rect[1]
                        if w < h:
                            angle += 90
                        self.crosswalk_tracker['last_angle'] = angle
                except:
                    self.crosswalk_tracker['last_angle'] = 0.0
                
                # Переключение состояния
                is_ready_to_switch = (
                    area_ratio >= self.CROSSWALK_SWITCH_AREA_RATIO and
                    bottom_y_ratio >= self.CROSSWALK_SWITCH_BOTTOM_RATIO or
                    (self.crosswalk_tracker['consecutive_frames'] >= self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES 
                     and area_ratio > 0.18)
                )
                
                if is_ready_to_switch and self.crosswalk_tracker['alignment_status'] == 'aligned':
                    if self.crosswalk_tracker['stage'] != 'ready':
                        self.crosswalk_tracker['stage'] = 'ready'
                elif area_ratio > 0.07 or bottom_y_ratio > 0.75:
                    if self.crosswalk_tracker['stage'] in ['far', 'not_detected']:
                        self.crosswalk_tracker['stage'] = 'approaching'
                elif area_ratio > 0.01:
                    if self.crosswalk_tracker['stage'] == 'not_detected':
                        self.crosswalk_tracker['stage'] = 'far'
        else:
            # Потеря детекции
            if self.frame_counter - self.crosswalk_tracker['last_seen_frame'] > 15:
                self.crosswalk_tracker['stage'] = 'not_detected'
                self.crosswalk_tracker['consecutive_frames'] = 0
                self.crosswalk_tracker['position_announced'] = False
                self.crosswalk_tracker['alignment_status'] = 'not_aligned'
                # Сбрасываем флаг состояния подготовки
                if hasattr(self, 'crosswalk_ready_announced'):
                    self.crosswalk_ready_announced = False
                    self.crosswalk_ready_time = 0
                if hasattr(self, 'traffic_light_history'):
                    self.traffic_light_history.clear()
                    self.green_light_announced = False
    
    def _handle_crosswalk_approaching(self, frame_visualizations, image_height, image_width, image):
        """Обработка ситуации приближения к пешеходному переходу"""
        # Детекция препятствий
        if self.obstacle_detector and self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
        
        # Добавляем визуализацию препятствий
        for obs in self.last_detected_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)
        
        # Сначала проверяем близкие препятствия (порог повышен, тревога только при очень близком)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75  # Повышено до 0.75
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12  # Повышено до 0.12
        near_obstacles = [
            obs for obs in self.last_detected_obstacles
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD)
        ]
        
        # Если есть близкие препятствия, применяем ту же логику озвучивания
        if near_obstacles:
            main_obstacle = near_obstacles[0]
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()
            
            # Проверяем, нужно ли озвучивать (избегаем повторов)
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                should_announce = True
                self.last_obstacle_speech_time = current_time
            
            if should_announce:
                return self._speech_for_obstacle(obstacle_name)
        else:
            # Если препятствий нет, очищаем запись
            self.last_obstacle_speech = ""
        
        # Логика выравнивания
        if self.crosswalk_tracker['alignment_status'] == 'not_aligned':
            guidance_text = "正在接近斑马线，为您对准方向。"
            self.crosswalk_tracker['alignment_status'] = 'aligning'
        else:
            angle = self.crosswalk_tracker['last_angle']
            center_x_ratio = self.crosswalk_tracker['last_center_x_ratio']
            
            ANGLE_ALIGN_THRESHOLD = 15
            POSITION_ALIGN_THRESHOLD = 0.25
            
            if abs(angle) > ANGLE_ALIGN_THRESHOLD:
                guidance_text = "右转" if angle < 0 else "左转"
            elif abs(center_x_ratio - 0.5) > (POSITION_ALIGN_THRESHOLD / 2):
                guidance_text = "右移" if center_x_ratio < 0.5 else "左移"
            else:
                self.crosswalk_tracker['alignment_status'] = 'aligned'
                guidance_text = "斑马线已对准，继续前行。"
        
        # Добавляем панель данных
        data_for_panel = {
            "Состояние": "Выравнивание по переходу",
            "Подсказка": guidance_text,
            "Угол": f"{self.crosswalk_tracker['last_angle']:.1f}°",
            "Смещение": f"{(self.crosswalk_tracker['last_center_x_ratio'] - 0.5):.2f}"
        }
        frame_visualizations.append({
            "type": "data_panel",
            "data": data_for_panel,
            "position": (25, image_height - 75)
        })
        
        return guidance_text
    
    def _execute_state_machine(self, mask, image, frame_visualizations, 
                              image_height, image_width, curr_gray):
        """Выполнение логики конечного автомата"""
        if self.current_state == STATE_ONBOARDING:
            return self._handle_onboarding(mask, image, frame_visualizations, 
                                         image_height, image_width)
        elif self.current_state == STATE_NAVIGATING:
            return self._handle_navigating(mask, image, frame_visualizations,
                                         image_height, image_width, curr_gray)
        elif self.current_state == STATE_MANEUVERING_TURN:
            return self._handle_maneuvering_turn(mask, image, frame_visualizations,
                                               image_height, image_width)
        elif self.current_state == STATE_LOCKING_ON:
            return self._handle_locking_on(frame_visualizations)
        elif self.current_state == STATE_AVOIDING_OBSTACLE:
            return self._handle_avoiding_obstacle(mask, image, frame_visualizations,
                                                image_height, image_width)
        
        return ""
    
    def _handle_onboarding(self, mask, image, frame_visualizations, image_height, image_width):
        """Обработка состояния выхода на тактильную дорожку"""
        image_center_x = image_width / 2
        vp_features = self._get_vanishing_point_features(mask)
        
        if vp_features and vp_features['fit_error'] < self.VP_FIT_ERROR_THRESHOLD:
            # Используем метод точки схода
            VP, L_center = vp_features["VP"], vp_features["L_center"]
            
            if self.onboarding_step == ONBOARDING_STEP_ROTATION:
                if abs(VP[0] - image_center_x) < (image_width * self.ONBOARDING_ALIGN_THRESHOLD_RATIO):
                    guidance_text = "方向已对正！现在校准位置。"
                    self.onboarding_step = ONBOARDING_STEP_TRANSLATION
                else:
                    guidance_text = "请向左转动。" if VP[0] < image_center_x else "请向右转动。"
                
                angle_error_px = VP[0] - image_center_x
                self._add_data_panel(frame_visualizations, {
                    "Состояние": "Выход на дорожку (направление)",
                    "Подсказка": guidance_text,
                    "Угол": f"{angle_error_px:.1f}px",
                    "Смещение": "Ожидает калибровки"
                }, (25, image_height - 75))
                
            elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
                L_center_bottom_x = self._calculate_line_x_at_y(L_center, image_height - 1)
                
                if L_center_bottom_x:
                    center_offset_pixels = L_center_bottom_x - image_center_x
                    center_offset_ratio = abs(center_offset_pixels) / image_width
                    
                    if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                        guidance_text = "校准完成！您已在盲道上，开始前行。"
                        self.current_state = STATE_NAVIGATING
                    else:
                        guidance_text = "请向左平移。" if L_center_bottom_x < image_center_x else "请向右平移。"
                    
                    self._add_data_panel(frame_visualizations, {
                        "Состояние": "Выход на дорожку (позиция)",
                        "Подсказка": guidance_text,
                        "Угол": "Выровнено",
                        "Смещение": f"{center_offset_ratio * 100:.1f}%"
                    }, (25, image_height - 75))
                else:
                    guidance_text = "请向前移动，让盲道更清晰。"
        else:
            # Используем метод в пиксельной области
            pixel_features = self._get_pixel_domain_features(mask, image.shape)
            if not pixel_features:
                return ""
            self._add_navigation_info_visualization(pixel_features, image_height, image_width, frame_visualizations)
            guidance_text = self._handle_pixel_domain_onboarding(
                pixel_features, image_height, image_width, frame_visualizations
            )
        
        return guidance_text
    
    def _handle_navigating(self, mask, image, frame_visualizations, 
                          image_height, image_width, curr_gray):
        """Обработка состояния обычной навигации"""
        image_center_x = image_width / 2
        
        # Извлекаем признаки пути
        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return "路径特征提取失败"
        self._add_navigation_info_visualization(features, image_height, image_width, frame_visualizations)
        
        # Обнаружение поворота
        if self.turn_cooldown_frames == 0:
            corner_info = self._detect_sharp_corner(features['centerline_data'])
            if corner_info:
                self._update_turn_tracker(corner_info)
                
                if self.turn_detection_tracker['consecutive_hits'] >= 3:
                    stable_corner_info = self.turn_detection_tracker['corner_info']
                    corner_y = stable_corner_info['corner_point_pixel'][1]
                    turn_trigger_y_threshold = image_height * 0.65
                    
                    if corner_y > turn_trigger_y_threshold:
                        # Запускаем поворот
                        direction_text = '右' if self.turn_detection_tracker['direction'] == 'right' else '左'
                        self.current_state = STATE_MANEUVERING_TURN
                        self.maneuver_target_info = stable_corner_info
                        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                        self._reset_turn_tracker()
                        # Больше не озвучиваем «дошли до поворота», сразу возвращаем пустую строку — дальше обработает последующая логика
                        return ""
                    else:
                        # Больше не предупреждаем о повороте заранее, продолжаем обычную навигацию
                        pass
        
        # Приоритет 1: детекция препятствий (наивысший приоритет)
        obstacles = self._check_obstacles(image, mask, frame_visualizations)
        if obstacles:
            # Получаем основное препятствие
            main_obstacle = obstacles[0]
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()
            
            # Проверяем, нужно ли озвучивать (избегаем повторов)
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                # Другое препятствие — озвучиваем сразу
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                # То же препятствие, но кулдаун истёк — озвучиваем снова
                should_announce = True
                self.last_obstacle_speech_time = current_time
            
            if should_announce:
                # Не входим в полный процесс обхода, только предупреждаем
                # Задаём сообщение о препятствии для озвучивания вместо прямого возврата
                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)
            # Если озвучивать не нужно, продолжаем обычную навигацию
        else:
            # Препятствий нет, очищаем запись
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None
        
        # Приоритет 2: обычная навигация (смещение влево/вправо, поворот влево/вправо > прямо)
        return self._generate_navigation_guidance(
            features, image_height, image_width, frame_visualizations
        )
    
    def _handle_maneuvering_turn(self, mask, image, frame_visualizations,
                                image_height, image_width):
        """Обработка состояния поворота"""
        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return "丢失路径，重新搜索。"
        self._add_navigation_info_visualization(features, image_height, image_width, frame_visualizations)
        if self.maneuver_step == MANEUVER_STEP_1_ISSUE_COMMAND:
            direction_text = '右' if self.maneuver_target_info['direction'] == 'right' else '左'
            guidance_text = f"请向{direction_text}平移。"
            
            poly_func = features['poly_func']
            y_check = image_height * 0.7
            self.maneuver_target_info['old_path_center_x'] = poly_func(y_check)
            
            self.maneuver_step = MANEUVER_STEP_2_WAIT_FOR_SHIFT
            
            self._add_data_panel(frame_visualizations, {
                "Состояние": "Обработка поворота",
                "Подсказка": guidance_text,
                "Шаг": "Выдача команды",
                "Направление": direction_text
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif self.maneuver_step == MANEUVER_STEP_2_WAIT_FOR_SHIFT:
            old_path_x = self.maneuver_target_info.get('old_path_center_x')
            if old_path_x is None:
                self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                return ""
            
            poly_func = features['poly_func']
            y_check = image_height * 0.7
            current_path_x = poly_func(y_check)
            shift_distance = abs(current_path_x - old_path_x)
            
            centerline_data = features['centerline_data']
            width_at_check_y = self._get_width_at_y(centerline_data, y_check)
            
            if shift_distance > (width_at_check_y * 0.5):
                guidance_text = "检测到已移动，开始对准新方向。"
                self.maneuver_step = MANEUVER_STEP_3_ALIGN_ON_NEW_PATH
            else:
                direction_text = '右' if self.maneuver_target_info['direction'] == 'right' else '左'
                guidance_text = f"请继续向{direction_text}平移。"
            
            self._add_data_panel(frame_visualizations, {
                "Состояние": "Обработка поворота",
                "Подсказка": guidance_text,
                "Шаг": "Ожидание смещения",
                "Величина смещения": f"{shift_distance:.1f}px"
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif self.maneuver_step == MANEUVER_STEP_3_ALIGN_ON_NEW_PATH:
            poly_func = features['poly_func']
            y_check = image_height * 0.5
            current_path_x_at_center = poly_func(y_check)
            
            pixel_error = current_path_x_at_center - image_width / 2
            center_offset_ratio = abs(pixel_error) / image_width
            
            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "已对准新路径，请向前直行。"
                self.current_state = STATE_NAVIGATING
                self.maneuver_target_info = None
                self.turn_cooldown_frames = self.TURN_COOLDOWN_DURATION
            else:
                move_direction = "右" if pixel_error > 0 else "左"
                guidance_text = f"请向{move_direction}微调，对准盲道。"
            
            self._add_data_panel(frame_visualizations, {
                "Состояние": "Обработка поворота",
                "Подсказка": guidance_text,
                "Шаг": "Выравнивание по новому пути",
                "Ошибка": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
            
            return guidance_text
    
    def _handle_locking_on(self, frame_visualizations):
        """Обработка состояния захвата"""
        if not self.lock_on_data:
            self.current_state = STATE_NAVIGATING
            return ""
        
        main_obstacle = self.lock_on_data['main_obstacle']
        
        # Добавляем эффект пульсации
        self._add_obstacle_visualization(main_obstacle, frame_visualizations, pulse_effect=True)
        
        # Проверяем время
        if time.time() - self.lock_on_data['start_time'] > 0.7:
            self.avoidance_plan = self.lock_on_data['avoidance_plan']
            self.avoidance_step_index = 0
            self.current_state = STATE_AVOIDING_OBSTACLE
            self.lock_on_data = None
        
        return ""
    
    def _handle_avoiding_obstacle(self, mask, image, frame_visualizations,
                                 image_height, image_width):
        """Обработка состояния обхода препятствия"""
        if not self.avoidance_plan or self.avoidance_step_index >= len(self.avoidance_plan):
            self.current_state = STATE_NAVIGATING
            self.avoidance_plan = None
            return "避让完成，已回到盲道。"
        
        step = self.avoidance_plan[self.avoidance_step_index]
        
        if step['type'] == 'sidestep_clear':
            direction = step['direction']
            
            if self.obstacle_detector:
                final_obstacles = self._detect_obstacles(image, mask)
            else:
                final_obstacles = []
            
            if final_obstacles:
                guidance_text = f"路径被挡住，请向{'右' if direction == 'right' else '左'}侧平移。"
            else:
                guidance_text = "好的，请停下侧移。"
                self.avoidance_step_index += 1
            
            self._add_data_panel(frame_visualizations, {
                "Состояние": "Обход препятствия",
                "Подсказка": guidance_text,
                "Шаг": "Боковой выход",
                "Направление": direction
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif step['type'] == 'forward_pass':
            # Упрощённая обработка, сразу переходим к следующему шагу
            self.avoidance_step_index += 1
            return "向前直行几步越过障碍物。然后说‘好了’。"
            
        elif step['type'] == 'sidestep_return':
            direction = step['direction']
            features = self._get_pixel_domain_features(mask, image.shape)
            
            if not features:
                return f"没看到盲道，请向{'右' if direction == 'right' else '左'}侧小幅移动。"
            
            poly_func = features['poly_func']
            y_target = image_height * 0.5
            x_target = poly_func(y_target)
            
            center_offset_pixels = x_target - image_width / 2
            center_offset_ratio = abs(center_offset_pixels) / image_width
            
            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "已回到盲道。"
                self.avoidance_step_index += 1
            else:
                guidance_text = "向右平移，对准盲道" if center_offset_pixels > 0 else "向左平移，对准盲道"
            
            self._add_data_panel(frame_visualizations, {
                "Состояние": "Обход препятствия",
                "Подсказка": guidance_text,
                "Шаг": "Возврат на дорожку",
                "Смещение": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
            
            return guidance_text
    
    # ========== Вспомогательные методы ==========
    
    def _get_vanishing_point_features(self, mask):
        """Извлечение признаков точки схода"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: 
                return None
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < 5000: 
                return None
            
            rect = cv2.minAreaRect(main_contour)
            center, _, angle = rect
            angle_rad = np.deg2rad(angle)
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                          [np.sin(angle_rad), np.cos(angle_rad)]])
            points_transformed = np.dot(main_contour.squeeze(1) - center, R)
            left_points = main_contour.squeeze(1)[points_transformed[:, 0] < 0]
            right_points = main_contour.squeeze(1)[points_transformed[:, 0] >= 0]
            
            if len(left_points) < 20 or len(right_points) < 20: 
                return None
            
            [vx_l, vy_l, x_l, y_l] = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)
            [vx_r, vy_r, x_r, y_r] = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            a1, b1, c1 = vy_l, -vx_l, vx_l * y_l - vy_l * x_l
            a2, b2, c2 = vy_r, -vx_r, vx_r * y_r - vy_r * x_r
            determinant = a1 * b2 - a2 * b1
            
            if abs(determinant) < 1e-6: 
                return None
            
            vp_x = (b1 * c2 - b2 * c1) / determinant
            vp_y = (a2 * c1 - a1 * c2) / determinant
            L_center = ((vx_l + vx_r) / 2, (vy_l + vy_r) / 2, (x_l + x_r) / 2, (y_l + y_r) / 2)
            
            total_dist = 0
            for pt in left_points: 
                total_dist += abs((pt[0] - x_l) * vy_l - (pt[1] - y_l) * vx_l)
            for pt in right_points: 
                total_dist += abs((pt[0] - x_r) * vx_r - (pt[1] - y_r) * vy_r)
            fit_error = total_dist / (len(left_points) + len(right_points))
            
            return {"VP": (vp_x, vp_y), "L_center": L_center, "fit_error": fit_error}
        except:
            return None
    
    def _get_pixel_domain_features(self, mask, image_shape):
        """Извлечение признаков в пиксельной области"""
        try:
            height, width = image_shape[:2]
            
            centerline_data = []
            for y in range(height - 1, int(height * 0.3), -5):
                row = mask[y, :]
                x_pixels = np.where(row > 0)[0]
                if x_pixels.size > 10:
                    x_min, x_max = x_pixels[0], x_pixels[-1]
                    path_width = x_max - x_min
                    center_x = (x_min + x_max) / 2
                    centerline_data.append([y, center_x, path_width])
            
            if len(centerline_data) < 20: 
                return None
            
            data = np.array(centerline_data)
            
            # Применяем сглаживание центральной линии
            data = self._smooth_centerline(data)
            
            # Обнаружение крутого поворота
            sharp_turn_index = self._find_sharp_turn(data)
            if sharp_turn_index is not None:
                cutoff_index = int(sharp_turn_index * 0.6)
                if cutoff_index >= 10:
                    data = data[:cutoff_index]
            
            y_coords, x_coords, widths = data[:, 0], data[:, 1], data[:, 2]
            weights = widths
            
            # Исходная полиномиальная аппроксимация
            coeffs_raw = np.polyfit(y_coords, x_coords, 2, w=weights)
            
            # 【Новое】Временное сглаживание полиномиальных коэффициентов
            self.poly_coeffs_history.append(coeffs_raw.copy())
            if len(self.poly_coeffs_history) > self.poly_coeffs_history_max:
                self.poly_coeffs_history.pop(0)
            
            # Сглаживаем коэффициенты экспоненциально взвешенным скользящим средним
            if len(self.poly_coeffs_history) >= 3:
                # Веса: у последних кадров вес выше
                weights_time = np.array([0.7 ** (len(self.poly_coeffs_history) - i - 1) 
                                        for i in range(len(self.poly_coeffs_history))])
                weights_time = weights_time / np.sum(weights_time)
                
                # Взвешенное среднее коэффициентов
                coeffs = np.zeros_like(coeffs_raw)
                for i, hist_coeffs in enumerate(self.poly_coeffs_history):
                    coeffs += hist_coeffs * weights_time[i]
            else:
                coeffs = coeffs_raw
            
            poly_func = np.poly1d(coeffs)
            
            curvature_proxy = abs(coeffs[0])
            tangent_slope = 2 * coeffs[0] * height + coeffs[1]
            tangent_angle_rad = np.arctan(tangent_slope)
            
            return {
                "poly_func": poly_func,
                "curvature_proxy": curvature_proxy,
                "tangent_angle_rad": tangent_angle_rad,
                "centerline_data": np.array(centerline_data)
            }
        except Exception as e:
            logger.warning(f"Pixel domain feature calculation failed: {e}")
            return None
    
    def _find_sharp_turn(self, data):
        """Поиск точки крутого поворота"""
        window_size = 5
        angle_threshold = 30
        
        for i in range(len(data) - 2 * window_size):
            front_window = data[i:i + window_size]
            back_window = data[i + window_size:i + 2 * window_size]
            
            front_dir = [front_window[-1, 1] - front_window[0, 1],
                        front_window[-1, 0] - front_window[0, 0]]
            back_dir = [back_window[-1, 1] - back_window[0, 1],
                       back_window[-1, 0] - back_window[0, 0]]
            
            angle1 = np.arctan2(front_dir[1], front_dir[0])
            angle2 = np.arctan2(back_dir[1], back_dir[0])
            angle_diff = abs(np.degrees(angle2 - angle1))
            
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > angle_threshold:
                return i + window_size
        
        return None
    
    def _detect_sharp_corner(self, centerline_data, angle_threshold_deg=45):
        """Обнаружение крутого поворота"""
        try:
            if len(centerline_data) < 15: 
                return None
            points_in_range = np.array(centerline_data)
            num_points = len(points_in_range)
            
            window_size = max(5, int(num_points * 0.15))
            best_turn_info = None
            max_angle_diff = 0
            
            for i in range(0, num_points - 2 * window_size, 2):
                front_segment = points_in_range[i:i + window_size]
                back_segment = points_in_range[i + window_size:i + 2 * window_size]
                
                if len(front_segment) < 3 or len(back_segment) < 3:
                    continue
                
                front_y = front_segment[:, 0]
                front_x = front_segment[:, 1]
                front_coeffs = np.polyfit(front_y, front_x, 1)
                front_slope = front_coeffs[0]
                
                back_y = back_segment[:, 0]
                back_x = back_segment[:, 1]
                back_coeffs = np.polyfit(back_y, back_x, 1)
                back_slope = back_coeffs[0]
                
                front_angle = np.arctan(front_slope)
                back_angle = np.arctan(back_slope)
                
                angle_diff_rad = back_angle - front_angle
                angle_diff_deg = abs(np.degrees(angle_diff_rad))
                
                if angle_diff_deg > max_angle_diff and angle_diff_deg > angle_threshold_deg:
                    max_angle_diff = angle_diff_deg
                    corner_point_idx = i + window_size
                    corner_point = points_in_range[corner_point_idx]
                    
                    direction = "right" if angle_diff_rad > 0 else "left"
                    
                    post_turn_segment = points_in_range[
                        corner_point_idx:min(corner_point_idx + window_size * 2, num_points)]
                    if len(post_turn_segment) > 0:
                        post_turn_center_x = np.mean(post_turn_segment[:, 1])
                    else:
                        post_turn_center_x = corner_point[1]
                    
                    best_turn_info = {
                        "corner_point_pixel": (corner_point[1], corner_point[0]),
                        "turn_angle": max_angle_diff,
                        "direction": direction,
                        "post_turn_center_x": post_turn_center_x,
                        "corner_point_idx": corner_point_idx
                    }
            
            return best_turn_info
        
        except Exception as e:
            logger.warning(f"Corner detection error: {e}")
            return None
    
    def _update_turn_tracker(self, corner_info):
        """Обновление трекера поворотов"""
        detected_direction = corner_info['direction']
        
        if detected_direction == self.turn_detection_tracker['direction']:
            self.turn_detection_tracker['consecutive_hits'] += 1
        else:
            self.turn_detection_tracker['direction'] = detected_direction
            self.turn_detection_tracker['consecutive_hits'] = 1
        
        self.turn_detection_tracker['last_seen_frame'] = self.frame_counter
        self.turn_detection_tracker['corner_info'] = corner_info
    
    def _reset_turn_tracker(self):
        """Сброс трекера поворотов"""
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
    
    def _calculate_line_x_at_y(self, line_params, y_target):
        """Вычисление значения x прямой при заданной координате y"""
        vx, vy, x0, y0 = line_params
        if abs(vy) < 1e-6:
            return None
        t = (y_target - y0) / vy
        x = x0 + t * vx
        return x
    
    def _get_width_at_y(self, centerline_data, y_target):
        """Получение ширины пути при заданной координате y"""
        ys = centerline_data[:, 0]
        ws = centerline_data[:, 2]
        idx = np.abs(ys - y_target).argmin()
        return ws[idx]
    
    def _detect_obstacles(self, image, path_mask=None):
        """Детекция препятствий"""
        logger.info(f"[_detect_obstacles] Начало выполнения, Frame={self.frame_counter}, obstacle_detector={'загружен' if self.obstacle_detector else 'не загружен'}")
        
        if self.obstacle_detector is None:
            logger.warning("[_detect_obstacles] Детектор препятствий не загружен!")
            return []
        
        # 【Новое】Выводим классы белого списка (только при первом вызове)
        if not hasattr(self, '_classes_printed'):
            self._classes_printed = True
            if hasattr(self.obstacle_detector, 'WHITELIST_CLASSES'):
                logger.info("[_detect_obstacles] ===== Классы белого списка детекции препятствий =====")
                for idx, name in enumerate(self.obstacle_detector.WHITELIST_CLASSES):
                    logger.info(f"  - Класс {idx}: {name}")
                logger.info(f"[_detect_obstacles] Всего классов: {len(self.obstacle_detector.WHITELIST_CLASSES)}")
        
        try:
            # 【Ключевое изменение】Используем метод detect() из ObstacleDetectorClient
            # Он автоматически выполняет настройку текстовых подсказок YOLO-E, детекцию и базовую фильтрацию
            logger.info(f"[_detect_obstacles] Вызов ObstacleDetectorClient.detect()... image.shape={image.shape}")
            detected_obstacles = self.obstacle_detector.detect(image, path_mask=path_mask)
            
            logger.info(f"[_detect_obstacles] ObstacleDetectorClient вернул объектов: {len(detected_obstacles)}")
            
            # ObstacleDetectorClient уже возвращает данные в правильном формате, включая:
            # - name: название объекта
            # - mask: маска сегментации
            # - area: площадь
            # - area_ratio: доля площади
            # - center_x, center_y: координаты центра
            # - bottom_y_ratio: доля Y нижней границы
            
            # Дополняем поля, которые могут отсутствовать, но нужны последующему коду
            H, W = image.shape[:2]
            for i, obj in enumerate(detected_obstacles):
                # Добавляем координаты ограничивающей рамки для визуализации
                if 'mask' in obj and obj['mask'] is not None:
                    y_coords, x_coords = np.where(obj['mask'] > 0)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                        obj['box_coords'] = (x1, y1, x2, y2)
                        
                        # Дополняем поля, которые могут отсутствовать
                        if 'y_position_ratio' not in obj:
                            obj['y_position_ratio'] = obj.get('center_y', 0) / H
                        if 'label' not in obj:
                            obj['label'] = obj.get('name', 'unknown')
                        if 'center' not in obj:
                            obj['center'] = (obj.get('center_x', 0), obj.get('center_y', 0))
                        # Добавляем фиктивную уверенность (если нужно)
                        if 'confidence' not in obj:
                            obj['confidence'] = 0.5  # ObstacleDetectorClient уже отфильтровал объекты с низкой уверенностью
                
                # Подробный вывод в лог
                logger.info(f"[_detect_obstacles] Объект {i+1}/{len(detected_obstacles)}: ")
                logger.info(f"  - Класс: {obj.get('name', 'unknown')}")
                logger.info(f"  - Площадь: {obj.get('area', 0)} pixels ({obj.get('area_ratio', 0):.3f} of image)")
                logger.info(f"  - Центр: ({obj.get('center_x', 0):.1f}, {obj.get('center_y', 0):.1f})")
                logger.info(f"  - bottom_y_ratio: {obj.get('bottom_y_ratio', 0):.3f} (близко: {'да' if obj.get('bottom_y_ratio', 0) > 0.7 else 'нет'})")
                logger.info(f"  - area_ratio: {obj.get('area_ratio', 0):.3f} (слишком большой: {'да' if obj.get('area_ratio', 0) > 0.1 else 'нет'})")
            
            # 【Внимание】ObstacleDetectorClient уже выполнил следующую фильтрацию:
            # 1. Фильтрация по размеру (слишком большие объекты с площадью более 70% отфильтровываются)
            # 2. Фильтрация по уверенности (по переменной окружения AIGLASS_OBS_CONF, по умолчанию 0.25)
            # 3. Если передан path_mask, выполняется пространственная фильтрация относительно пути:
            #    - требуется пересечение с путём не менее 100 пикселей
            #    - требуется, чтобы пересечение составляло не менее 1% площади объекта
            # Поэтому здесь дополнительная фильтрация не нужна
            
            logger.info(f"[_detect_obstacles] ===== Итоговый результат: возвращено препятствий: {len(detected_obstacles)} =====")
            for idx, obj in enumerate(detected_obstacles):
                logger.info(f"  {idx+1}. {obj.get('name', 'unknown')} - позиция:({obj.get('center_x', 0):.0f},{obj.get('center_y', 0):.0f}) "
                        f"bottom_y_ratio:{obj.get('bottom_y_ratio', 0):.2f} area_ratio:{obj.get('area_ratio', 0):.3f}")
            
            return detected_obstacles
            
        except Exception as e:
            logger.error(f"[_detect_obstacles] Ошибка детекции препятствий: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _check_and_set_obstacle_voice(self, obstacles):
        """Проверка препятствий и установка голосового сообщения для озвучивания"""
        if not obstacles:
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None
            return
        
        # Отбираем близкие препятствия (порог повышен, тревога только при очень близком)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75  # Повышено до 0.75: нижняя граница препятствия должна быть ниже 75% высоты кадра
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12  # Повышено до 0.12: препятствие должно занимать более 12% кадра
        
        near_obstacles = []
        for obs in obstacles:
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD):
                near_obstacles.append(obs)
        
        if near_obstacles:
            # Получаем главное препятствие (с наибольшей площадью)
            main_obstacle = max(near_obstacles, key=lambda x: x.get('area_ratio', 0))
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()
            
            # Проверяем, нужно ли озвучивать
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                # Другое препятствие — озвучиваем сразу
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                # То же препятствие, но кулдаун истёк — озвучиваем снова
                should_announce = True
                self.last_obstacle_speech_time = current_time
            
            if should_announce:
                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)
        else:
            # Близких препятствий нет
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None

    def _check_obstacles(self, image, mask, frame_visualizations):
        """Проверка и обработка препятствий"""
        # Используем стратегию кэширования
        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            final_obstacles = self._detect_obstacles(image, mask)
            # 【Новое】Стабилизируем препятствия, чтобы избежать повторного наложения
            if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                final_obstacles = self._stabilize_obstacle_list(
                    final_obstacles, 
                    self.last_detected_obstacles,
                    self.prev_gray,
                    curr_gray,
                    image.shape[:2]
                )
            self.last_detected_obstacles = final_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
        else:
            if self.frame_counter - self.last_obstacle_detection_frame < self.OBSTACLE_CACHE_DURATION_FRAMES:
                final_obstacles = self.last_detected_obstacles
            else:
                final_obstacles = []
        
        # Добавляем визуализацию
        for obs in final_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)
        
        # Отбираем близкие препятствия (порог повышен, тревога только при очень близком)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75  # Повышено до 0.75: нижняя граница препятствия должна быть ниже 75% высоты кадра
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12  # Повышено до 0.12: препятствие должно занимать более 12% кадра
        
        near_obstacles = [
            obs for obs in final_obstacles
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD)
        ]
        
        return near_obstacles
    
    def _plan_avoidance(self, obstacle_info, image_width):
        """Планирование пути обхода препятствия"""
        obstacle_center_x = obstacle_info['center_x']
        image_center_x = image_width / 2
        
        if obstacle_center_x < image_center_x:
            turn_direction = 'right'
        else:
            turn_direction = 'left'
        
        plan = [
            {'type': 'sidestep_clear', 'direction': turn_direction},
            {'type': 'forward_pass'},
            {'type': 'sidestep_return', 'direction': 'left' if turn_direction == 'right' else 'right'}
        ]
        return plan
    
    def _generate_navigation_guidance(self, features, image_height, image_width, frame_visualizations):
        """Формирование навигационной подсказки"""
        poly_func = features['poly_func']
        is_curve = features['curvature_proxy'] > self.CURVATURE_PROXY_THRESHOLD
        lookahead_ratio = 0.6 if is_curve else 0.4
        y_target = image_height * lookahead_ratio
        x_target = poly_func(y_target)
        
        # Добавляем визуализацию центральной линии
        plot_y = np.arange(int(image_height * 0.3), image_height, 5).astype(int)
        plot_x = poly_func(plot_y).astype(int)
        centerline_points = np.vstack((plot_x, plot_y)).T.tolist()
        frame_visualizations.append({
            "type": "polyline",
            "points": centerline_points,
            "color": "yellow",
            "width": 2
        })
        
        # Добавляем целевую точку
        frame_visualizations.append({
            "type": "circle",
            "center": [int(x_target), int(y_target)],
            "radius": 10,
            "color": "red"
        })
        
        # Вычисляем навигационную команду (приоритет: поворот/смещение > прямо)
        center_offset_pixels = x_target - image_width / 2
        center_offset_ratio = abs(center_offset_pixels) / image_width
        orientation_error_rad = features['tangent_angle_rad']
        
        # Сначала проверяем, нужен ли поворот (влево/вправо)
        if orientation_error_rad > self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = "左转"
        elif orientation_error_rad < -self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = "右转"
        # Затем проверяем, нужно ли смещение (влево/вправо)
        elif center_offset_ratio > self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
            guidance_text = "右移" if center_offset_pixels > 0 else "左移"
        # И только в последнюю очередь — прямо
        else:
            guidance_text = "保持直行"
        
        # Добавляем панель данных
        self._add_data_panel(frame_visualizations, {
            "Состояние": "Обычная навигация",
            "Подсказка": guidance_text,
            "Ориентация": f"{np.degrees(orientation_error_rad):.1f}°",
            "Смещение": f"{center_offset_ratio * 100:.1f}%"
        }, (25, image_height - 75))
        
        return guidance_text
    
    def _handle_pixel_domain_onboarding(self, pixel_features, image_height, image_width, frame_visualizations):
        """Обработка подсказок выхода на дорожку в пиксельной области"""
        image_center_x = image_width / 2
        orientation_error_rad = pixel_features['tangent_angle_rad']
        poly_func = pixel_features['poly_func']
        
        y_bottom = image_height - 1
        x_target_bottom = poly_func(y_bottom)
        center_offset_pixels = x_target_bottom - image_center_x
        center_offset_ratio = abs(center_offset_pixels) / image_width
        
        if self.onboarding_step == ONBOARDING_STEP_ROTATION:
            if abs(orientation_error_rad) < self.ONBOARDING_ORIENTATION_THRESHOLD_RAD:
                guidance_text = "方向已对正！现在校准位置。"
                self.onboarding_step = ONBOARDING_STEP_TRANSLATION
            else:
                guidance_text = "请向左转动。" if orientation_error_rad > 0.1 else "请向右转动。"
            
            self._add_data_panel(frame_visualizations, {
                "Состояние": "Выход на дорожку (направление)",
                "Подсказка": guidance_text,
                "Угол": f"{np.degrees(orientation_error_rad):.1f}°",
                "Смещение": "Ожидает калибровки"
            }, (25, image_height - 75))
            self._add_navigation_info_visualization(pixel_features, image_height, image_width, frame_visualizations)
    
            return guidance_text
            
        elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
            if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "校准完成！您已在盲道上，开始前行。"
                self.current_state = STATE_NAVIGATING
            else:
                guidance_text = "请向右平移。" if center_offset_pixels > 0 else "请向左平移。"
            
            self._add_data_panel(frame_visualizations, {
                "Состояние": "Выход на дорожку (позиция)",
                "Подсказка": guidance_text,
                "Угол": "Выровнено",
                "Смещение": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
        
        return guidance_text
    
    def _add_obstacle_visualization(self, obstacle, visualizations, pulse_effect=False):
        """Добавление визуализации препятствия (упрощённо: только рамка, близкие — красные, дальние — жёлтые)"""
        try:
            # Вычисляем уровень опасности препятствия
            bottom_y_ratio = obstacle.get('bottom_y_ratio', 0)
            area_ratio = obstacle.get('area_ratio', 0)
            
            # Определяем, является ли препятствие близким
            is_near = bottom_y_ratio > 0.7 or area_ratio > 0.1  # Близкое препятствие
            
            # Добавляем визуализацию рамки по маске (если есть)
            if 'mask' in obstacle and obstacle['mask'] is not None:
                mask = obstacle['mask']
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Находим наибольший контур
                    max_contour = max(contours, key=cv2.contourArea)
                    points = max_contour.squeeze(1)[::5].tolist()
                    
                    # Выбираем цвет рамки по расстоянию: близкие — красный, дальние — жёлтый
                    if is_near:
                        outline_color = "rgba(255, 0, 0, 1.0)"  # Красный
                        thickness = 3
                    else:
                        outline_color = "rgba(255, 255, 0, 0.8)"  # Жёлтый
                        thickness = 2
                    
                    # Добавляем только рамку, без заливки и текста
                    visualizations.append({
                        "type": "outline",
                        "points": points,
                        "color": outline_color,
                        "thickness": thickness
                    })
        except Exception as e:
            logger.error(f"[_add_obstacle_visualization] Не удалось добавить визуализацию препятствия: {e}")

    def _add_navigation_info_visualization(self, features, image_height, image_width, frame_visualizations):
        """Добавление визуализации навигационных расчётов"""
        if not features:
            return
        
        try:
            # Получаем результаты расчётов
            poly_func = features.get('poly_func')
            curvature_proxy = features.get('curvature_proxy', 0)
            tangent_angle_rad = features.get('tangent_angle_rad', 0)
            tangent_angle_deg = np.degrees(tangent_angle_rad)
            
            # Рисуем направление касательной
            if poly_func:
                # Вычисляем касательную в нижней части кадра
                y_bottom = image_height - 50
                x_bottom = poly_func(y_bottom)
                
                # Вычисляем конечную точку касательной
                tangent_length = 100
                dx = tangent_length * np.cos(tangent_angle_rad)
                dy = tangent_length * np.sin(tangent_angle_rad)
                
                # 【Новое】Рисуем опорную пунктирную линию (вертикально вверх)
                baseline_length = 80
                frame_visualizations.append({
                    "type": "dashed_line",
                    "start": [int(x_bottom), int(y_bottom)],
                    "end": [int(x_bottom), int(y_bottom - baseline_length)],
                    "color": "rgba(255, 255, 255, 0.6)",  # Белая пунктирная линия
                    "thickness": 2
                })
                
                # Добавляем визуализацию касательной
                frame_visualizations.append({
                    "type": "arrow",
                    "start": [int(x_bottom), int(y_bottom)],
                    "end": [int(x_bottom + dx), int(y_bottom - dy)],  # Обратите внимание на направление оси Y
                    "color": "rgba(0, 255, 255, 0.8)",  # Голубой
                    "thickness": 3,
                    "tip_length": 0.3
                })
                
                # 【Новое】Рисуем дугу, обозначающую угол
                arc_radius = 40
                # Угол опорной линии -90 градусов (вверх), угол касательной — tangent_angle_deg
                # В OpenCV угол отсчитывается от правой горизонтали против часовой стрелки
                start_angle = -90  # Опорная линия (вертикально вверх)
                end_angle = -90 + tangent_angle_deg  # Угол касательной
                frame_visualizations.append({
                    "type": "angle_arc",
                    "center": [int(x_bottom), int(y_bottom)],
                    "radius": arc_radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "color": "rgba(255, 200, 0, 0.8)",  # Оранжево-жёлтый
                    "thickness": 2
                })
                
                # Добавляем текст угла (размер шрифта уменьшен вдвое)
                frame_visualizations.append({
                    "type": "text_with_bg",
                    "text": f"Угол: {tangent_angle_deg:.1f}°",
                    "position": [int(x_bottom + 10), int(y_bottom - 30)],
                    "font_scale": 0.3,  # Уменьшено вдвое с 0.6 до 0.3
                    "color": "rgba(255, 255, 255, 1.0)",
                    "bg_color": "rgba(0, 0, 0, 0.7)"
                })
            
            # Добавляем информацию о кривизне (размер шрифта уменьшен вдвое)
            if curvature_proxy > 0.00001:
                curve_text = "Поворот" if curvature_proxy > 0.00005 else "Плавный изгиб"
                frame_visualizations.append({
                    "type": "text_with_bg",
                    "text": f"{curve_text}: {curvature_proxy:.2e}",
                    "position": [20, 100],
                    "font_scale": 0.25,  # Уменьшено вдвое с 0.5 до 0.25
                    "color": "rgba(255, 255, 0, 1.0)",
                    "bg_color": "rgba(0, 0, 0, 0.7)"
                })
                
            # Показываем точки данных центральной линии
            if 'centerline_data' in features:
                centerline_data = features['centerline_data']
                # Показываем ширину пути в средней части кадра
                mid_idx = len(centerline_data) // 2
                if mid_idx < len(centerline_data):
                    y, x, width = centerline_data[mid_idx]
                    # Рисуем линию-индикатор ширины (заменена на двустороннюю стрелку)
                    frame_visualizations.append({
                        "type": "double_arrow",  # Новый тип — двусторонняя стрелка
                        "start": [int(x - width/2), int(y)],
                        "end": [int(x + width/2), int(y)],
                        "color": "rgba(0, 255, 0, 0.8)",
                        "thickness": 2,
                        "tip_length": 0.15
                    })
                    # Добавляем текст ширины (размер шрифта уменьшен вдвое)
                    frame_visualizations.append({
                        "type": "text_with_bg",
                        "text": f"Ширина: {width:.0f}px",
                        "position": [int(x - 30), int(y - 10)],
                        "font_scale": 0.25,  # Уменьшено вдвое с 0.5 до 0.25
                        "color": "rgba(255, 255, 255, 1.0)",
                        "bg_color": "rgba(0, 0, 0, 0.7)"
                    })
        except Exception as e:
            logger.error(f"Не удалось добавить визуализацию навигационной информации: {e}")

    def _add_data_panel(self, visualizations, data, position):
        """Добавление панели данных"""
        visualizations.append({
            "type": "data_panel",
            "data": data,
            "position": position
        })
    
    def _add_crosswalk_info_visualization(self, viz_data, image_height, image_width, visualizations):
        """Добавление наглядной визуализации информации о детекции перехода"""
        try:
            # 1. Рисуем метку центра перехода (большой крест)
            center_x = int(viz_data['center_x_ratio'] * image_width)
            center_y = int(viz_data['center_y_ratio'] * image_height)
            
            cross_size = 20 if viz_data['in_arrival'] else 15  # Уменьшенный размер
            cross_color = "rgba(255, 100, 0, 1.0)" if viz_data['in_arrival'] else "rgba(0, 200, 255, 0.8)"
            
            # Горизонтальная линия
            visualizations.append({
                "type": "line",
                "start": [center_x - cross_size, center_y],
                "end": [center_x + cross_size, center_y],
                "color": cross_color,
                "thickness": 2  # Тоньше
            })
            # Вертикальная линия
            visualizations.append({
                "type": "line",
                "start": [center_x, center_y - cross_size],
                "end": [center_x, center_y + cross_size],
                "color": cross_color,
                "thickness": 2  # Тоньше
            })
            
            # 2. Рисуем стрелку к переходу (от центра кадра к центру перехода)
            screen_center_x = image_width // 2
            screen_center_y = image_height // 2
            
            # Рисуем стрелку, только если переход не в центре кадра
            distance = np.sqrt((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)
            if distance > 80:  # Стрелка рисуется только от 80 пикселей (меньше помех)
                visualizations.append({
                    "type": "arrow",
                    "start": [screen_center_x, screen_center_y],
                    "end": [center_x, center_y],
                    "color": "rgba(255, 150, 0, 0.6)",  # Пониженная непрозрачность
                    "thickness": 2,  # Тоньше
                    "tip_length": 0.15  # Уменьшенная стрелка
                })
            
            # 3. Добавляем информационную панель (правый верхний угол)
            panel_x = image_width - 180
            panel_y = 20
            
            # Подготавливаем данные панели
            panel_data = {
                "Переход": viz_data['stage'],
                "Площадь": f"{viz_data['area_ratio']*100:.1f}%",
                "Положение": viz_data['position'],
            }
            
            if viz_data['has_occlusion']:
                panel_data["Состояние"] = "Перекрыт"
            elif viz_data['in_arrival']:
                panel_data["Состояние"] = "Можно переходить"
            
            visualizations.append({
                "type": "data_panel",
                "data": panel_data,
                "position": (panel_x, panel_y)
            })
            
            # 4. Добавляем полосу прогресса площади (визуализация размера площади)
            bar_width = 150
            bar_height = 20
            bar_x = image_width - bar_width - 20
            bar_y = panel_y + 90
            
            # Фоновая рамка
            visualizations.append({
                "type": "rectangle",
                "top_left": (bar_x, bar_y),
                "bottom_right": (bar_x + bar_width, bar_y + bar_height),
                "color": "rgba(50, 50, 50, 0.7)",
                "filled": True
            })
            
            # Заполнение прогресса (0–100%, но 100% соответствует порогу arrival 0.25)
            progress = min(viz_data['area_ratio'] / 0.25, 1.0)
            fill_width = int(bar_width * progress)
            
            # Выбираем цвет в зависимости от этапа
            if viz_data['in_arrival']:
                fill_color = "rgba(0, 255, 100, 0.8)"  # Зелёный (можно переходить)
            elif viz_data['area_ratio'] >= 0.18:
                fill_color = "rgba(255, 200, 0, 0.8)"  # Жёлтый (приближение)
            elif viz_data['area_ratio'] >= 0.08:
                fill_color = "rgba(0, 200, 255, 0.8)"  # Голубой (рядом)
            else:
                fill_color = "rgba(100, 150, 255, 0.8)"  # Синий (обнаружен)
            
            visualizations.append({
                "type": "rectangle",
                "top_left": (bar_x + 2, bar_y + 2),
                "bottom_right": (bar_x + fill_width - 2, bar_y + bar_height - 2),
                "color": fill_color,
                "filled": True
            })
            
            # Подпись полосы прогресса (текст с поддержкой китайского, уменьшенный шрифт)
            visualizations.append({
                "type": "text_with_bg",
                "text": f"Близость: {int(progress * 100)}%",
                "position": [bar_x, bar_y - 18],
                "font_scale": 0.25,  # Уменьшенный шрифт
                "color": "rgba(255, 255, 255, 1.0)",
                "bg_color": "rgba(0, 0, 0, 0.7)"
            })
            
        except Exception as e:
            logger.error(f"Не удалось добавить визуализацию перехода: {e}")
    
    def _add_traffic_light_visualization(self, color, visualizations, image_height, image_width):
        """Добавление визуализации состояния светофора"""
        # Рисуем индикатор светофора в правом верхнем углу
        x = image_width - 100
        y = 50
        
        # Фоновая рамка
        visualizations.append({
            "type": "rectangle",
            "top_left": (x - 40, y - 40),
            "bottom_right": (x + 40, y + 100),
            "color": "rgba(0, 0, 0, 0.5)",
            "filled": True
        })
        
        # Три круглых сигнала
        colors = {
            "red": [(255, 0, 0), (50, 0, 0), (50, 0, 0)],
            "yellow": [(50, 50, 0), (255, 255, 0), (50, 50, 0)],
            "green": [(0, 50, 0), (0, 50, 0), (0, 255, 0)],
            "unknown": [(50, 50, 50), (50, 50, 50), (50, 50, 50)]
        }
        
        light_colors = colors.get(color, colors["unknown"])
        positions = [y - 20, y + 20, y + 60]
        
        for i, (pos_y, light_color) in enumerate(zip(positions, light_colors)):
            # Внешний круг
            visualizations.append({
                "type": "circle",
                "center": [x, pos_y],
                "radius": 18,
                "color": f"rgba(100, 100, 100, 1.0)",
                "thickness": 2
            })
            # Внутренний круг (цвет сигнала)
            visualizations.append({
                "type": "circle",
                "center": [x, pos_y],
                "radius": 15,
                "color": f"rgba({light_color[0]}, {light_color[1]}, {light_color[2]}, 1.0)",
                "filled": True
            })
        
        # Подпись
        visualizations.append({
            "type": "text_with_bg",
            "text": f"Светофор: {color}",
            "position": [x - 35, y + 90],
            "font_scale": 0.5,
            "color": "rgba(255, 255, 255, 1.0)",
            "bg_color": "rgba(0, 0, 0, 0.7)"
        })
    
    def _to_cn_obstacle(self, name: str) -> str:
        """Преобразование названия препятствия на китайский"""
        try:
            key = (name or '').strip().lower()
            return _OBSTACLE_NAME_CN.get(key, '障碍物')
        except:
            return '障碍物'

    def _speech_for_obstacle(self, name: str) -> str:
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

    def _draw_command_button(self, image, text):
        """Отрисовка кнопки команды внизу по центру (единообразно с режимом перехода)"""
        try:
            H, W = image.shape[:2]
            full_text = f"Текущая команда: {text if text else '—'}"
            
            # Параметры кнопки
            font_px = 14
            pad_x, pad_y = 14, 8
            bottom_margin = 28
            
            # Вычисляем размер текста
            if PIL_AVAILABLE:
                try:
                    from PIL import Image as PILImage, ImageDraw, ImageFont
                    # Пытаемся загрузить китайский шрифт
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
            
            # Вычисляем положение кнопки (внизу по центру)
            bw = tw + pad_x * 2
            bh = th + pad_y * 2
            radius = max(10, bh // 2)
            
            cx = W // 2
            left = max(8, cx - bw // 2)
            top = H - bottom_margin - bh
            right = min(W - 8, left + bw)
            bottom = top + bh
            
            # Рисуем полупрозрачный фон со скруглёнными углами
            overlay = image.copy()
            bg_color = (26, 32, 41)  # Тёмный фон
            border_color = (60, 76, 102)  # Рамка
            
            # Скруглённый прямоугольник (середина + два круга)
            cv2.rectangle(overlay, (left + radius, top), (right - radius, bottom), bg_color, -1)
            cv2.circle(overlay, (left + radius, (top + bottom) // 2), radius, bg_color, -1)
            cv2.circle(overlay, (right - radius, (top + bottom) // 2), radius, bg_color, -1)
            
            # Полупрозрачное смешивание
            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
            
            # Рисуем рамку
            cv2.rectangle(image, (left + radius, top), (right - radius, bottom), border_color, 1)
            cv2.circle(image, (left + radius, (top + bottom) // 2), radius, border_color, 1)
            cv2.circle(image, (right - radius, (top + bottom) // 2), radius, border_color, 1)
            
            # Рисуем текст
            text_x = left + pad_x
            text_y = top + pad_y + th
            
            if PIL_AVAILABLE and 'font' in locals() and font:
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
            logger.error(f"Не удалось отрисовать кнопку команды: {e}")
            return image
    
    def _parse_color(self, color_str):
        """Разбор строки цвета, возвращает формат BGR"""
        try:
            if color_str.startswith('rgba('):
                values = color_str[5:-1].split(',')
                r, g, b = int(values[0]), int(values[1]), int(values[2])
                return (b, g, r)  # OpenCV использует формат BGR
            elif color_str == 'yellow':
                return (0, 255, 255)
            elif color_str == 'red':
                return (0, 0, 255)
            else:
                return (0, 0, 255)  # По умолчанию красный
        except:
            return (0, 0, 255)

    def _draw_data_panel_no_bg(self, image, data, position=(15, 15)):
        """Отрисовка панели данных (версия без чёрного фона)"""
        if not PIL_AVAILABLE:
            return image
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")
            
            env_scale = float(os.getenv("AIGLASS_PANEL_SCALE", "0.7"))
            base_font_size = max(10, int(round(14 * env_scale)))
            
            # Пробуем несколько шрифтов, чтобы обеспечить отображение китайского текста
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
                "C:/Windows/Fonts/simsun.ttc",    # SimSun
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
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
            
            # Рисуем текст с эффектом обводки
            y_offset = position[1]
            for key, value in data.items():
                text = f"{key}: {value}"
                
                # Рисуем чёрную обводку (8 направлений)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((position[0] + dx, y_offset + dy), text, 
                                    font=font, fill=(0, 0, 0, 255))
                
                # Рисуем белый текст
                draw.text((position[0], y_offset), text, font=font, fill=(255, 255, 255, 255))
                y_offset += base_font_size + 5
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        except Exception as e:
            logger.warning(f"Не удалось отрисовать панель данных: {e}")
            return image


    def _draw_visualizations(self, image, viz_elements):
        """Улучшенный метод отрисовки визуализации"""
        if not viz_elements:
            return image
        
        # Получаем текущее время для эффектов анимации
        current_time = time.time()
        
        # Разделяем элементы разных типов
        panel_elements = [v for v in viz_elements if v.get("type") == "data_panel"]
        standard_elements = [v for v in viz_elements if v.get("type") != "data_panel"]
        
        # Первый проход: рисуем заливку (полупрозрачный эффект)
        for element in standard_elements:
            elem_type = element.get("type")
            
            if elem_type in ['blind_path_mask', 'obstacle_mask', 'crosswalk_mask']:
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.5)"))
                    
                    # Обработка эффекта пульсации
                    if element.get("effect") == "pulse":
                        pulse_speed = element.get("pulse_speed", 1.0)
                        alpha = 0.3 + 0.3 * np.sin(current_time * pulse_speed * 2 * np.pi)
                    else:
                        alpha = 0.4
                    
                    # Получаем ограничивающую рамку многоугольника
                    x, y, w, h = cv2.boundingRect(points)
                    
                    # Убеждаемся, что рамка в пределах изображения
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    if w > 0 and h > 0:
                        # Создаём бинарную маску, равную 1 только внутри многоугольника
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        local_points = points - np.array([x, y])
                        cv2.fillPoly(binary_mask, [local_points], 255)
                        
                        # Смешиваем цвета только внутри многоугольника
                        local_region = image[y:y+h, x:x+w].copy()
                        
                        # Создаём цветной слой наложения
                        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                        color_overlay[:] = color
                        
                        # Смешиваем с помощью бинарной маски
                        for c in range(3):
                            local_region[:, :, c] = np.where(
                                binary_mask > 0,
                                (1 - alpha) * local_region[:, :, c] + alpha * color_overlay[:, :, c],
                                local_region[:, :, c]
                            )
                        
                        # Возвращаем смешанную локальную область в исходное изображение
                        image[y:y+h, x:x+w] = local_region
        
        # Второй проход: рисуем контуры и прочие элементы
        for element in standard_elements:
            elem_type = element.get("type")
            
            # 【Новое】Рисуем прямую линию
            if elem_type == 'line':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                cv2.line(image, start, end, color, thickness)
            
            # Рисуем контур/обводку
            elif elem_type == 'outline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                    thickness = element.get("thickness", 3)
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
            
            # Рисуем ломаную
            elif elem_type == 'polyline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 0, 1.0)"))
                    thickness = element.get("width", 2)
                    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)
            
            # Рисуем круг
            elif elem_type == 'circle':
                center = tuple(element.get("center", (0, 0)))
                radius = element.get("radius", 10)
                color = self._parse_color(element.get("color", "rgba(255, 0, 0, 1.0)"))
                thickness = element.get("thickness", -1 if element.get("filled", True) else 2)
                cv2.circle(image, center, radius, color, thickness)
            
            # Рисуем прямоугольник
            elif elem_type == 'rectangle':
                top_left = tuple(element.get("top_left", (0, 0)))
                bottom_right = tuple(element.get("bottom_right", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 0, 0, 0.5)"))
                thickness = -1 if element.get("filled", True) else 2
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            
            # Рисуем стрелку
            elif elem_type == 'arrow':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.3)
                cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip_length)
            
            # 【Новое】Рисуем двустороннюю стрелку
            elif elem_type == 'double_arrow':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.15)
                # Рисуем среднюю линию
                cv2.line(image, start, end, color, thickness)
                # Рисуем стрелки на обоих концах
                # Вычисляем вектор направления стрелки
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Единичный вектор направления
                    ux, uy = dx/length, dy/length
                    # Длина стрелки
                    arrow_len = length * tip_length
                    # Левая стрелка
                    tip1_x = int(start[0] + arrow_len * ux)
                    tip1_y = int(start[1] + arrow_len * uy)
                    # Рисуем левую стрелку (указывает влево)
                    angle = np.arctan2(dy, dx)
                    arrow_angle = 30 * np.pi / 180  # Угол стрелки
                    p1 = (int(start[0] + arrow_len * np.cos(angle - arrow_angle)),
                          int(start[1] + arrow_len * np.sin(angle - arrow_angle)))
                    p2 = (int(start[0] + arrow_len * np.cos(angle + arrow_angle)),
                          int(start[1] + arrow_len * np.sin(angle + arrow_angle)))
                    cv2.line(image, start, p1, color, thickness)
                    cv2.line(image, start, p2, color, thickness)
                    # Правая стрелка (указывает вправо)
                    p3 = (int(end[0] - arrow_len * np.cos(angle - arrow_angle)),
                          int(end[1] - arrow_len * np.sin(angle - arrow_angle)))
                    p4 = (int(end[0] - arrow_len * np.cos(angle + arrow_angle)),
                          int(end[1] - arrow_len * np.sin(angle + arrow_angle)))
                    cv2.line(image, end, p3, color, thickness)
                    cv2.line(image, end, p4, color, thickness)
            
            # 【Новое】Рисуем пунктирную линию
            elif elem_type == 'dashed_line':
                start = np.array(element.get("start", (0, 0)))
                end = np.array(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.6)"))
                thickness = element.get("thickness", 2)
                dash_length = 10
                gap_length = 5
                # Вычисляем общую длину и направление
                total_vec = end - start
                total_len = np.linalg.norm(total_vec)
                if total_len > 0:
                    unit_vec = total_vec / total_len
                    # Рисуем отрезки пунктира
                    current_len = 0
                    while current_len < total_len:
                        seg_start = start + unit_vec * current_len
                        seg_end = start + unit_vec * min(current_len + dash_length, total_len)
                        cv2.line(image, tuple(seg_start.astype(int)), tuple(seg_end.astype(int)), color, thickness)
                        current_len += dash_length + gap_length
            
            # 【Новое】Рисуем дугу угла
            elif elem_type == 'angle_arc':
                center = tuple(element.get("center", (100, 100)))
                radius = element.get("radius", 40)
                start_angle = element.get("start_angle", -90)
                end_angle = element.get("end_angle", 0)
                color = self._parse_color(element.get("color", "rgba(255, 200, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                # Функция ellipse в OpenCV: startAngle и endAngle отсчитываются от правой горизонтали по часовой стрелке
                # Нужно преобразование: наши углы отсчитываются от правой горизонтали против часовой стрелки (математический стандарт)
                # OpenCV требуется отсчёт от правой горизонтали по часовой стрелке
                cv2_start = -end_angle  # Преобразуем в формат OpenCV
                cv2_end = -start_angle
                # Убеждаемся, что диапазон углов корректен
                if cv2_start > cv2_end:
                    cv2_start, cv2_end = cv2_end, cv2_start
                cv2.ellipse(image, center, (radius, radius), 0, cv2_start, cv2_end, color, thickness)
            
            # 【Изменено】Рисуем текст с фоном (с поддержкой китайского)
            elif elem_type == 'text_with_bg':
                text = element.get("text", "")
                pos = element.get("position", [10, 30])
                font_scale = element.get("font_scale", 0.6)
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                
                # Используем новую функцию отрисовки китайского текста
                image = self._draw_chinese_text(image, text, tuple(pos), 
                                              font_scale=font_scale, 
                                              color=color,
                                              stroke_color=(0, 0, 0),
                                              stroke_width=1)
            
            # Рисуем значок предупреждения
            elif elem_type == 'warning_icon':
                pos = element.get("position", (100, 100))
                level = element.get("level", "info")
                text = element.get("text", "")
                flash = element.get("flash", False)
                
                # Выбираем цвет в зависимости от уровня
                if level == "danger":
                    icon_color = (0, 0, 255)  # Красный
                    text_color = (255, 255, 255)
                elif level == "warning":
                    icon_color = (0, 165, 255)  # Оранжевый
                    text_color = (255, 255, 255)
                else:
                    icon_color = (0, 255, 255)  # Жёлтый
                    text_color = (0, 0, 0)
                
                # Эффект мигания
                if flash:
                    alpha = 0.5 + 0.5 * np.sin(current_time * 4 * np.pi)
                    icon_color = tuple(int(c * alpha) for c in icon_color)
                
                # Рисуем треугольный значок предупреждения
                triangle = np.array([
                    [pos[0], pos[1] - 20],
                    [pos[0] - 15, pos[1]],
                    [pos[0] + 15, pos[1]]
                ], np.int32)
                cv2.fillPoly(image, [triangle], icon_color)
                cv2.polylines(image, [triangle], True, (255, 255, 255), 2)
                
                # Рисуем восклицательный знак
                cv2.putText(image, "!", (pos[0] - 5, pos[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Рисуем текстовую подпись (с поддержкой китайского)
                if text:
                    font_scale = 0.5
                    # Используем новую функцию отрисовки китайского текста
                    text_pos = (pos[0] - 20, pos[1] + 20)  # Упрощённый расчёт положения
                    image = self._draw_chinese_text(image, text, text_pos,
                                                  font_scale=font_scale,
                                                  color=text_color,
                                                  stroke_color=(0, 0, 0),
                                                  stroke_width=1)
            
            # Обычный текст
            elif elem_type == 'text':
                text = element.get("text", "")
                pos = tuple(element.get("pos", (10, 30)))
                # Используем функцию отрисовки китайского текста
                image = self._draw_chinese_text(image, text, pos,
                                              font_scale=0.7,
                                              color=(255, 255, 255),
                                              stroke_color=(0, 0, 0),
                                              stroke_width=1)
        
        # 【Изменено】Рисуем панель данных (версия без чёрного фона)
        if PIL_AVAILABLE:
            for panel in panel_elements:
                image = self._draw_data_panel_no_bg(image, panel["data"], panel["position"])
        else:
            # Если PIL нет, тоже используем эффект обводки
            for panel in panel_elements:
                y_offset = panel["position"][1]
                for key, value in panel["data"].items():
                    text = f"{key}: {value}"
                    # Рисуем обводку текста
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(image, text, (panel["position"][0] + dx, y_offset + dy), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    # Рисуем белый текст
                    cv2.putText(image, text, (panel["position"][0], y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25
        
        return image


    
    def _draw_chinese_text(self, image, text, position, font_scale=0.6, color=(255, 255, 255), 
                         stroke_color=(0, 0, 0), stroke_width=1):
        """Отрисовка китайского текста шрифтом Microsoft YaHei, белый текст с чёрной обводкой"""
        if not PIL_AVAILABLE:
            # Если PIL нет, откатываемся на cv2.putText (будут отображаться вопросительные знаки)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            return image
        
        try:
            # Преобразуем в изображение PIL
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Вычисляем размер шрифта (на основе font_scale)
            base_size = 24  # Базовый размер шрифта
            font_size = int(base_size * font_scale / 0.6)
            
            # Пытаемся загрузить шрифт Microsoft YaHei
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/msyh.ttf",      # Microsoft YaHei (старая версия)
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Преобразуем цвет OpenCV из BGR в RGB
            rgb_color = (color[2], color[1], color[0])
            rgb_stroke = (stroke_color[2], stroke_color[1], stroke_color[0])
            
            # Рисуем текст (с эффектом обводки)
            x, y = position
            # Рисуем обводку
            draw.text((x, y), text, font=font, fill=rgb_stroke, 
                     stroke_width=stroke_width, stroke_fill=rgb_stroke)
            # Рисуем основной текст
            draw.text((x, y), text, font=font, fill=rgb_color)
            
            # Преобразуем обратно в формат OpenCV
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.warning(f"Не удалось отрисовать китайский текст: {e}")
            # Откатываемся на cv2.putText
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            return image

    def _draw_data_panel(self, image, data, position=(15, 15)):
        """Отрисовка панели данных (требуется Pillow)"""
        if not PIL_AVAILABLE:
            return image
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")
            
            env_scale = float(os.getenv("AIGLASS_PANEL_SCALE", "0.65"))
            base_font_size = max(8, int(round(16 * env_scale)))
            padding = max(4, int(round(8 * env_scale)))
            
            # Пытаемся загрузить шрифт Microsoft YaHei
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/msyh.ttf",      # Microsoft YaHei (старая версия)
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            text_lines = [f"{key}: {value}" for key, value in data.items()]
            text_to_draw = "\n".join(text_lines)
            
            bbox = draw.textbbox(position, text_to_draw, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            bg_rect = [
                (position[0] - padding, position[1] - padding),
                (position[0] + text_w + padding, position[1] + text_h + padding)
            ]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 128))
            draw.text(position, text_to_draw, font=font, fill=(255, 255, 255, 255))
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        except Exception:
            return image
    
    def reset(self):
        """Сброс состояния навигатора"""
        self.current_state = STATE_ONBOARDING
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
        self.turn_cooldown_frames = 0
        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None
        
        # Сброс оптического потока и связанного со сглаживанием
        self.flow_points = {}
        self.flow_grace = {}
        self.centerline_history = []
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0
        
        # Сброс связанного с голосом
        self.pending_obstacle_voice = None
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        
        # Сброс истории полиномиальных коэффициентов
        self.poly_coeffs_history = []
        self.crosswalk_tracker = {
            'stage': 'not_detected',
            'consecutive_frames': 0,
            'last_area_ratio': 0.0,
            'last_bottom_y_ratio': 0.0,
            'last_center_x_ratio': 0.5,
            'position_announced': False,
            'alignment_status': 'not_aligned',
            'last_seen_frame': 0,
            'last_angle': 0.0
        }
        self.frame_counter = 0
        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.last_any_speech_time = 0
        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0
        self.traffic_light_history.clear()
        self.last_traffic_light_state = "unknown"
        self.green_light_announced = False
    
    def _stabilize_obstacle_list(self, obstacles, prev_obstacles, prev_gray, curr_gray, 
                                image_shape, threshold=0.5):
        """Стабилизация результатов детекции препятствий во избежание повторного наложения"""
        if not obstacles or prev_gray is None or curr_gray is None:
            return obstacles
        
        H, W = image_shape
        stabilized = []
        used_prev = set()  # Запоминаем использованные исторические препятствия
        
        # Для каждого обнаруженного на текущем кадре препятствия
        for curr_obs in obstacles:
            if 'mask' not in curr_obs or curr_obs['mask'] is None:
                stabilized.append(curr_obs)
                continue
                
            curr_mask = curr_obs['mask']
            best_match = None
            best_iou = 0
            best_idx = -1
            
            # Ищем наилучшее совпадение среди исторических препятствий
            if prev_obstacles:
                for idx, prev_obs in enumerate(prev_obstacles):
                    if idx in used_prev or 'mask' not in prev_obs:
                        continue
                    
                    # Предсказываем новое положение исторического препятствия оптическим потоком
                    flow_mask = self._predict_mask_with_flow(prev_obs['mask'], prev_gray, curr_gray)
                    if flow_mask is None:
                        flow_mask = prev_obs['mask']
                    
                    # Вычисляем IoU
                    inter = np.logical_and(curr_mask > 0, flow_mask > 0).sum()
                    union = np.logical_or(curr_mask > 0, flow_mask > 0).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    
                    if iou > best_iou and iou > threshold:
                        best_iou = iou
                        best_match = flow_mask
                        best_idx = idx
            
            # Если совпадение найдено, сливаем результаты
            if best_match is not None and best_idx >= 0:
                used_prev.add(best_idx)
                # Сливаем текущую детекцию и предсказание оптическим потоком для повышения стабильности
                fused_mask = ((0.8 * curr_mask + 0.2 * best_match) > 128).astype(np.uint8) * 255
                curr_obs['mask'] = fused_mask
                # Обновляем производные атрибуты
                self._update_obstacle_properties(curr_obs, H, W)
            
            stabilized.append(curr_obs)
        
        return stabilized
  
    def _speech_for_obstacle(self, name: str) -> str:
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

    def _update_obstacle_properties(self, obs, H, W):
        """Обновление производных атрибутов препятствия"""
        if 'mask' not in obs or obs['mask'] is None:
            return
        
        mask = obs['mask']
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) > 0:
            obs['area'] = len(y_coords)
            obs['center_x'] = float(np.mean(x_coords))
            obs['center_y'] = float(np.mean(y_coords))
            obs['y_position_ratio'] = obs['center_y'] / H
            obs['area_ratio'] = obs['area'] / (H * W)
            obs['bottom_y_ratio'] = np.max(y_coords) / H
            
            # Обновляем ограничивающую рамку
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            obs['box_coords'] = (x1, y1, x2, y2)