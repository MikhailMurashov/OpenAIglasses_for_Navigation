# app/models.py
import os
import logging
import torch
from threading import Semaphore
from contextlib import contextmanager
from typing import List
from app.cloud.obstacle_detector_client import ObstacleDetectorClient
# ==========================================================
# 0. Импорт всех нужных классов-обёрток моделей (Clients) и базовых классов Ultralytics
# ==========================================================
# Это классы-обёртки, используемые сценарием перехода дороги
from app.cloud.crosswalk_detector_client import CrosswalkDetector
from app.cloud.coco_perception_client import COCOClient
from obstacle_detector_client import ObstacleDetectorClient

# Это классы Ultralytics, напрямую используемые сценарием тактильной дорожки
from ultralytics import YOLO, YOLOE

logger = logging.getLogger(__name__)

# ==========================================================
# 1. Глобальное устройство и управление параллелизмом (централизованно)
# ==========================================================
DEVICE = os.getenv("AIGLASS_DEVICE", "cuda:0")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    logger.warning(f"AIGLASS_DEVICE={DEVICE}, но CUDA не обнаружена, откат на CPU")
    DEVICE = "cpu"
IS_CUDA = DEVICE.startswith("cuda")

# Настройка AMP (автоматической смешанной точности)
AMP_POLICY = os.getenv("AIGLASS_AMP", "bf16").lower()
AMP_DTYPE = torch.bfloat16 if AMP_POLICY == "bf16" else (
    torch.float16 if AMP_POLICY == "fp16" else None) if IS_CUDA else None

# 🔥 Ключевое: единственный глобальный семафор параллелизма GPU, общий для всех сценариев
GPU_SLOTS = int(os.getenv("AIGLASS_GPU_SLOTS", "2"))
gpu_semaphore = Semaphore(GPU_SLOTS)


# Единый контекстный менеджер инференса; все сценарии должны вызывать модели через него
@contextmanager
def gpu_infer_slot():
    """
    Единое управление: ограничение параллелизма GPU + torch.inference_mode() + AMP autocast
    """
    with gpu_semaphore:
        if IS_CUDA and AMP_POLICY != "off" and AMP_DTYPE is not None:
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                yield
        else:
            with torch.inference_mode():
                yield


# Оптимизация ускорения cuDNN
try:
    if IS_CUDA:
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ==========================================================
# 2. Определение глобальных экземпляров моделей (все инициализированы как None)
# ==========================================================

# --- Модели сценария перехода дороги (обёрнуты в классы Client) ---
crosswalk_detector_client: CrosswalkDetector = None
coco_client: COCOClient = None
# ObstacleDetectorClient будет общим детектором препятствий для всех сценариев
obstacle_detector_client: ObstacleDetectorClient = None

# --- Модели сценария тактильной дорожки (напрямую классы Ultralytics) ---
# Они в основном нужны для сегментации и планирования пути; логика детекции отличается от сценария перехода дороги
blindpath_seg_model: YOLO = None
# Детекция препятствий переиспользует obstacle_detector_client, но текстовые признаки YOLOE нужно хранить отдельно
blindpath_whitelist_embeddings = None

# Глобальный флаг состояния загрузки
models_are_loaded = False


# ==========================================================
# 3. Единая функция загрузки моделей (вызывается из celery.py при запуске)
# ==========================================================
def init_all_models():
    """
    Вызывается один раз при запуске процесса Celery Worker.
    Загружает модели, нужные всем сценариям, в глобальные переменные.
    """
    global models_are_loaded
    if models_are_loaded:
        return

    logger.info(f"========= 🚀 Начало глобальной предзагрузки моделей (целевое устройство: {DEVICE}) =========")

    try:
        # --- [1] Загрузка общего детектора препятствий (ObstacleDetectorClient) ---
        global obstacle_detector_client
        logger.info("[1/4] Загрузка общей модели детекции препятствий (ObstacleDetectorClient)...")
        obstacle_detector_client = ObstacleDetectorClient(model_path='models/yoloe-11l-seg.pt')

        # 🔥🔥🔥 [Ключевое исправление] добавлен недостающий код переноса на устройство 🔥🔥🔥
        if hasattr(obstacle_detector_client, 'model') and obstacle_detector_client.model is not None:
            obstacle_detector_client.model.to(DEVICE)

        logger.info("...Общая модель детекции препятствий успешно загружена.")

        # --- [2] Загрузка моделей для перехода дороги (Clients) ---
        global crosswalk_detector_client, coco_client
        logger.info("[2/4] Загрузка модели сегментации перехода (CrosswalkDetector)...")
        crosswalk_detector_client = CrosswalkDetector(model_path='models/yolo-seg.pt')
        # Перенос внутренней модели YOLO на указанное устройство
        if hasattr(crosswalk_detector_client, 'model') and crosswalk_detector_client.model is not None:
            crosswalk_detector_client.model.to(DEVICE)
        logger.info("...Модель сегментации перехода успешно загружена.")

        logger.info("[3/4] Загрузка общей модели восприятия (COCOClient)...")
        coco_client = COCOClient(model_path='models/yolov8l-world.pt')
        # Перенос внутренней модели YOLO на указанное устройство
        if hasattr(coco_client, 'model') and coco_client.model is not None:
            coco_client.model.to(DEVICE)
        logger.info("...Общая модель восприятия успешно загружена.")

        # --- [4] Загрузка моделей для тактильной дорожки ---
        global blindpath_seg_model, blindpath_whitelist_embeddings
        logger.info("[4/4] Загрузка модели сегментации тактильной дорожки (YOLO)...")
        blindpath_seg_model = YOLO('models/yolo-seg.pt')
        blindpath_seg_model.to(DEVICE)
        blindpath_seg_model.fuse()
        logger.info("...Модель сегментации тактильной дорожки успешно загружена.")

        # Сохраняем для сценария тактильной дорожки ссылку на нужные ему текстовые признаки YOLOE
        if obstacle_detector_client:
            blindpath_whitelist_embeddings = obstacle_detector_client.whitelist_embeddings
            logger.info("...Признаки модели препятствий привязаны к сценарию тактильной дорожки.")

        # Все модели загружены
        models_are_loaded = True
        logger.info("========= ✅ Все модели успешно предзагружены. Worker готов! =========")

    except Exception as e:
        logger.error(f"Критическая ошибка при предзагрузке моделей: {e}", exc_info=True)
        # Пробрасываем исключение — это приведёт к сбою запуска Celery Worker, и это разумно,
        # так как Worker без моделей бесполезен; лучше выявить проблему заранее.
        raise