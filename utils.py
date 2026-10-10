# utils.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Сопоставление названий предметов
ITEM_TO_CLASS_MAP = {
    "红牛": "Red_Bull",
    "AD钙奶": "AD_milk",
    "ad钙奶": "AD_milk",
    "钙奶": "AD_milk",
}

# Сопоставление английских названий классов с китайскими
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

def extract_english_label(item_cn: str) -> tuple:
    """
    Получает английскую метку для китайского названия предмета
    :param item_cn: китайское название предмета
    :return: (английская метка, источник)
    """
    # Сначала ищем в локальном сопоставлении
    if item_cn in ITEM_TO_CLASS_MAP:
        return ITEM_TO_CLASS_MAP[item_cn], "local"
    
    # Если не найдено — возвращаем исходное название
    return item_cn, "direct"

def _to_cn_obstacle(name: str) -> str:
    """
    Преобразует английское название препятствия в китайское
    :param name: английское название
    :return: китайское название
    """
    try:
        key = (name or '').strip().lower()
        return _OBSTACLE_NAME_CN.get(key, '障碍物')
    except Exception:
        return '障碍物'

def estimate_global_affine(prev_gray, curr_gray, mask=None):
    """
    Оценивает глобальное аффинное преобразование между двумя кадрами
    :param prev_gray: предыдущий кадр в оттенках серого
    :param curr_gray: текущий кадр в оттенках серого
    :param mask: необязательная маска, расчёт только внутри неё
    :return: (аффинная матрица, число инлайеров)
    """
    try:
        # Извлечение ключевых точек
        detector = cv2.ORB_create(nfeatures=500)
        kp1, des1 = detector.detectAndCompute(prev_gray, mask)
        kp2, des2 = detector.detectAndCompute(curr_gray, mask)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0
        
        # Сопоставление ключевых точек
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        if len(matches) < 4:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0
        
        # Извлечение сопоставленных пар точек
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Оценка аффинного преобразования методом RANSAC
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, 
                                                 ransacReprojThreshold=3.0)
        
        if M is None:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0
        
        inlier_count = np.sum(inliers) if inliers is not None else 0
        return M, inlier_count
        
    except Exception as e:
        logger.warning(f"estimate_global_affine failed: {e}")
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), 0

def warp_mask(mask, M, output_shape):
    """
    Применяет аффинное преобразование к маске
    :param mask: входная маска
    :param M: аффинная матрица 2x3
    :param output_shape: размер на выходе (width, height)
    :return: преобразованная маска
    """
    try:
        if mask is None or M is None:
            return None
        
        W, H = output_shape
        warped = cv2.warpAffine(mask, M, (W, H), 
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        return warped
        
    except Exception as e:
        logger.warning(f"warp_mask failed: {e}")
        return None

def estimate_translation_flow(prev_gray, curr_gray, mask=None):
    """
    Оценивает сдвиг (оптический поток) между двумя кадрами
    :param prev_gray: предыдущий кадр в оттенках серого
    :param curr_gray: текущий кадр в оттенках серого
    :param mask: необязательная маска
    :return: (медианная величина потока, матрица сдвига)
    """
    try:
        # Расчёт разреженного оптического потока
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, 
                                         qualityLevel=0.3, minDistance=7, 
                                         mask=mask)
        
        if corners is None or len(corners) < 10:
            return 0.0, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        # Расчёт оптического потока
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, 
                                                       corners, None)
        
        # Отбор валидных точек
        valid_old = corners[status == 1]
        valid_new = next_pts[status == 1]
        
        if len(valid_old) < 5:
            return 0.0, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        # Расчёт смещения
        flow_vectors = valid_new - valid_old
        flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
        median_flow = np.median(flow_magnitudes)
        
        # Оценка среднего сдвига
        mean_translation = np.mean(flow_vectors, axis=0)
        M = np.array([[1, 0, mean_translation[0]], 
                      [0, 1, mean_translation[1]]], dtype=np.float32)
        
        return median_flow, M
        
    except Exception as e:
        logger.warning(f"estimate_translation_flow failed: {e}")
        return 0.0, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

def is_stationary_frame(prev_gray, curr_gray, mask=None, threshold=0.35):
    """
    Определяет, неподвижен ли пользователь
    :param prev_gray: предыдущий кадр в оттенках серого
    :param curr_gray: текущий кадр в оттенках серого
    :param mask: необязательная маска
    :param threshold: порог неподвижности
    :return: True — неподвижен, False — движется
    """
    try:
        median_flow, _ = estimate_translation_flow(prev_gray, curr_gray, mask)
        return median_flow < threshold
    except:
        return False

def compute_approach_metrics(prev_obstacles, curr_obstacles, M, H, W):
    """
    Вычисляет метрики приближения препятствий
    :param prev_obstacles: список препятствий предыдущего кадра
    :param curr_obstacles: список препятствий текущего кадра
    :param M: аффинная матрица
    :param H: высота изображения
    :param W: ширина изображения
    :return: список метрик приближения
    """
    metrics = []
    
    for curr_obs in curr_obstacles:
        # Поиск наилучшего совпадения среди препятствий предыдущего кадра
        best_match = None
        best_iou = 0.0
        
        curr_mask = curr_obs.get('mask')
        if curr_mask is None:
            metrics.append(None)
            continue
        
        for prev_obs in prev_obstacles:
            prev_mask = prev_obs.get('mask')
            if prev_mask is None:
                continue
            
            # Перенос маски предыдущего кадра в текущий кадр
            warped_prev = warp_mask(prev_mask, M, (W, H))
            if warped_prev is None:
                continue
            
            # Расчёт IoU
            intersection = np.logical_and(curr_mask > 0, warped_prev > 0).sum()
            union = np.logical_or(curr_mask > 0, warped_prev > 0).sum()
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou:
                best_iou = iou
                best_match = prev_obs
        
        if best_match is None:
            metrics.append(None)
            continue
        
        # Расчёт метрик
        curr_area = curr_obs.get('area', 0)
        prev_area = best_match.get('area', 0)
        area_growth = (curr_area - prev_area) / prev_area if prev_area > 0 else 0.0
        
        curr_bottom_y = curr_obs.get('bottom_y_ratio', 0)
        prev_bottom_y = best_match.get('bottom_y_ratio', 0)
        v_forward = curr_bottom_y - prev_bottom_y
        
        metrics.append({
            'area_growth': area_growth,
            'v_forward': v_forward,
            'iou': best_iou
        })
    
    return metrics

def compute_risk_scores(obstacles, prev_obstacles, M, path_mask, image_shape,
                       stop_th=0.6, avoid_th=0.56):
    """
    Вычисляет оценку риска для препятствий
    :param obstacles: текущий список препятствий
    :param prev_obstacles: список препятствий предыдущего кадра
    :param M: аффинная матрица
    :param path_mask: маска пути
    :param image_shape: размер изображения
    :param stop_th: порог остановки
    :param avoid_th: порог обхода
    :return: (оценённые препятствия, нужна ли остановка, нужен ли обход, элементы визуализации)
    """
    H, W = image_shape[:2]
    has_stop = False
    has_avoid = False
    risk_vis = []
    
    # Расчёт метрик приближения
    metrics = compute_approach_metrics(prev_obstacles, obstacles, M, H, W)
    
    for obs, met in zip(obstacles, metrics):
        risk_score = 0.0
        
        if met is not None:
            # Риск на основе скорости приближения и роста площади
            if met['v_forward'] > 0.004:  # движение вниз
                risk_score += 0.3
            if met['area_growth'] > 0.01:  # рост площади
                risk_score += 0.3
        
        # Риск на основе расстояния
        bottom_y = obs.get('bottom_y_ratio', 0)
        area_ratio = obs.get('area_ratio', 0)
        
        if bottom_y > 0.8 or area_ratio > 0.15:
            risk_score += 0.3
        
        # Дополнительный риск для динамических объектов
        name_lower = str(obs.get('name', '')).lower()
        if name_lower in DYNAMIC_CLASS_NAMES:
            risk_score *= 1.2
        
        obs['risk_score'] = risk_score
        
        # Обновление флагов
        if risk_score >= stop_th:
            has_stop = True
        elif risk_score >= avoid_th:
            has_avoid = True
        
        # Добавление визуализации риска
        if risk_score > 0.3:
            risk_color = "rgba(255, 0, 0, 0.3)" if risk_score >= stop_th else "rgba(255, 165, 0, 0.3)"
            risk_vis.append({
                "type": "risk_indicator",
                "score": risk_score,
                "color": risk_color,
                "position": [int(obs.get('center_x', W/2)), int(obs.get('center_y', H/2))]
            })
    
    return obstacles, has_stop, has_avoid, risk_vis

