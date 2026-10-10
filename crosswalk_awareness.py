# -*- coding: utf-8 -*-
"""
Монитор восприятия пешеходного перехода
Обнаружение пешеходного перехода по изменению площади и голосовые подсказки
Не переключает состояния, только выдаёт голосовые подсказки
"""
import time
import numpy as np
from collections import deque
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrosswalkAwarenessMonitor:
    """Монитор восприятия пешеходного перехода — модуль только голосовых подсказок"""
    
    def __init__(self):
        # Пороги площади (фиксированные опорные точки)
        self.THRESHOLDS = {
            'discover': 0.01,      # 1% — обнаружение
            'approaching': 0.08,   # 8% — приближение
            'near': 0.18,          # 18% — очень близко
            'arrival': 0.25,       # 25% — прибытие (можно переходить дорогу)
        }
        
        # Уже озвученные пороги (чтобы избежать повторов)
        self.broadcasted_thresholds = set()
        
        # История значений площади
        self.area_history = deque(maxlen=30)  # Хранить последние 30 кадров
        
        # Отметки времени
        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0
        
        # Флаги состояния
        self.in_arrival_state = False  # Находимся ли в состоянии «можно переходить дорогу»
        self.last_position_zone = None  # Направление, озвученное в прошлый раз
        
        # Настройка интервалов озвучивания (настраиваемые параметры — чем меньше значение, тем чаще озвучивание)
        # 【Настройка параметров】все интервалы разделены на 1.5, частота озвучивания увеличена в 1.5 раза
        self.REPEAT_INTERVALS = {
            'approaching': 6.7,   # Этап приближения: повтор каждые 6.7 с (было 10 с ÷ 1.5)
            'near': 3.3,          # Этап «очень близко»: повтор каждые 3.3 с (было 5 с ÷ 1.5)
            'arrival': 5.3,       # Этап прибытия: повтор каждые 5.3 с (было 8 с ÷ 1.5)
        }
        # Подсказка: чтобы изменить частоту, достаточно поменять эти значения
        # - меньше значение = чаще озвучивание
        # - больше значение = реже озвучивание
        
        # Порог определения отсутствия перекрытия
        self.OCCLUSION_THRESHOLD = 0.30  # Перекрытие >30% считается загораживанием
    
    def process_frame(self, crosswalk_mask, blind_path_mask=None) -> Optional[Dict[str, Any]]:
        """
        Обработка обнаружения пешеходного перехода на каждом кадре
        
        Возвращает:
        {
            'voice_text': текст для озвучивания,
            'priority': приоритет,
            'should_broadcast': нужно ли озвучивать,
            'area': текущая площадь,
            'position': описание положения,
            'visualization': данные визуализации (для внешней отрисовки)
        }
        или None (озвучивание не требуется)
        """
        # Если пешеходного перехода нет — сбросить состояние
        if crosswalk_mask is None:
            self._reset_if_needed()
            return None
        
        # 1. Вычислить площадь
        total_pixels = crosswalk_mask.size
        crosswalk_pixels = np.sum(crosswalk_mask > 0)
        area_ratio = crosswalk_pixels / total_pixels
        
        # 2. Вычислить положение центра
        y_coords, x_coords = np.where(crosswalk_mask > 0)
        if len(y_coords) == 0:
            return None
        
        center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
        center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]
        
        # 3. Записать историю
        current_time = time.time()
        self.area_history.append({
            'area': area_ratio,
            'center_x': center_x_ratio,
            'center_y': center_y_ratio,
            'time': current_time
        })
        
        # 4. Проверить перекрытие
        has_occlusion = self._check_occlusion(crosswalk_mask, blind_path_mask)
        
        # 5. Определить текущий этап и сформировать голосовую подсказку
        return self._generate_guidance(area_ratio, center_x_ratio, center_y_ratio, 
                                       has_occlusion, current_time)
    
    def _check_occlusion(self, crosswalk_mask, blind_path_mask) -> bool:
        """Проверить, перекрыт ли пешеходный переход тактильной дорожкой"""
        if blind_path_mask is None:
            return False
        
        crosswalk_area = crosswalk_mask > 0
        blind_path_area = blind_path_mask > 0
        
        # Вычислить перекрытие
        overlap = np.logical_and(crosswalk_area, blind_path_area)
        overlap_ratio = np.sum(overlap) / max(np.sum(crosswalk_area), 1)
        
        # Перекрытие выше порога считается загораживанием
        return overlap_ratio > self.OCCLUSION_THRESHOLD
    
    def _get_position_description(self, center_x_ratio) -> str:
        """Получить описание положения (деление на 3 зоны)"""
        if center_x_ratio < 0.40:
            return "在画面左侧"
        elif center_x_ratio < 0.60:
            return "在画面中间"
        else:
            return "在画面右侧"
    
    def _generate_guidance(self, area_ratio, center_x_ratio, center_y_ratio, 
                          has_occlusion, current_time) -> Optional[Dict[str, Any]]:
        """Сформировать голосовую подсказку"""
        
        # Проверить стабильность площади (избежать дрожания)
        if not self._is_area_stable(area_ratio):
            return None
        
        position_desc = self._get_position_description(center_x_ratio)
        
        # Этап 1: обнаружение (0.01-0.08)
        if area_ratio >= self.THRESHOLDS['discover'] and area_ratio < self.THRESHOLDS['approaching']:
            if self.THRESHOLDS['discover'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['discover'])
                return {
                    'voice_text': f"远处发现斑马线,{position_desc}",
                    'priority': 55,  # Повышено до 55, выше команд направления по тактильной дорожке (50)
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
        
        # Этап 2: приближение (0.08-0.18)
        elif area_ratio >= self.THRESHOLDS['approaching'] and area_ratio < self.THRESHOLDS['near']:
            # Первое озвучивание
            if self.THRESHOLDS['approaching'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['approaching'])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"正在靠近斑马线,{position_desc}",
                    'priority': 55,  # Повышено до 55
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
            # Повторное озвучивание (каждые 10 с или при смене положения)
            elif (current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['approaching'] or
                  position_desc != self.last_position_zone):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"正在靠近斑马线,{position_desc}",
                    'priority': 55,  # Повышено до 55
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
        
        # Этап 3: очень близко (0.18-0.25)
        elif area_ratio >= self.THRESHOLDS['near'] and area_ratio < self.THRESHOLDS['arrival']:
            # Первое озвучивание
            if self.THRESHOLDS['near'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['near'])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"接近斑马线,{position_desc}",
                    'priority': 60,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
            # Повторное озвучивание (каждые 5 с или при смене положения)
            elif (current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['near'] or
                  position_desc != self.last_position_zone):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"接近斑马线,{position_desc}",
                    'priority': 60,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
        
        # Этап 4: прибытие (area ≥ 0.25, без перекрытия)
        elif area_ratio >= self.THRESHOLDS['arrival']:
            # Подсказывать переход дороги можно только без перекрытия
            if has_occlusion:
                # Есть перекрытие — пока не подсказываем переход, остаёмся на этапе 3
                logger.info(f"[Пешеходный переход] Площадь достигла {area_ratio:.2f}, но переход перекрыт — пока не подсказываем переход")
                return None
            
            # Первое прибытие
            if not self.in_arrival_state:
                self.in_arrival_state = True
                self.arrival_first_broadcast_time = current_time
                self.last_broadcast_time = current_time
                logger.info(f"[Пешеходный переход] Состояние прибытия: area={area_ratio:.2f}, без перекрытия")
                return {
                    'voice_text': "斑马线到了可以过马路",
                    'priority': 80,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': '到达'
                }
            # Повторное озвучивание (каждые 8 с)
            elif current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['arrival']:
                self.last_broadcast_time = current_time
                return {
                    'voice_text': "斑马线到了可以过马路",
                    'priority': 80,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': '到达'
                }
            # Обработка тайм-аута (через 30 с автоматический выход из состояния прибытия)
            elif current_time - self.arrival_first_broadcast_time > 30.0:
                logger.info("[Пешеходный переход] Тайм-аут состояния прибытия 30 с, автоматический выход")
                self.in_arrival_state = False
                return None
        
        # Понижение: если площадь уменьшилась после состояния прибытия
        if self.in_arrival_state and area_ratio < 0.20:
            logger.info(f"[Пешеходный переход] Площадь снизилась до {area_ratio:.2f}, выход из состояния прибытия")
            self.in_arrival_state = False
            # Сбросить часть отметок об озвучивании, чтобы разрешить повторное озвучивание
            self.broadcasted_thresholds.discard(self.THRESHOLDS['arrival'])
        
        return None
    
    def _is_area_stable(self, area_ratio, stability_frames=5) -> bool:
        """Проверить стабильность площади (избежать срабатывания из-за дрожания)"""
        if len(self.area_history) < stability_frames:
            return True  # Начальный этап — считаем стабильным
        
        recent_areas = [h['area'] for h in list(self.area_history)[-stability_frames:]]
        
        # Проверить, что последние N кадров близки к текущей площади (±20%)
        for recent_area in recent_areas:
            if abs(recent_area - area_ratio) / max(area_ratio, 0.001) > 0.20:
                return False
        
        return True
    
    def _reset_if_needed(self):
        """Сбросить состояние (при исчезновении пешеходного перехода)"""
        if len(self.area_history) > 0:
            logger.info("[Пешеходный переход] Пешеходный переход исчез, состояние сброшено")
        
        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_position_zone = None
    
    def reset(self):
        """Полный сброс"""
        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0
        self.last_position_zone = None
        logger.info("[Пешеходный переход] Монитор восприятия сброшен")
    
    def is_in_arrival_state(self) -> bool:
        """Находимся ли в состоянии прибытия (используется извне, чтобы решить, приостанавливать ли голос тактильной дорожки)"""
        return self.in_arrival_state
    
    def get_current_area(self) -> float:
        """Получить текущую площадь"""
        if len(self.area_history) > 0:
            return self.area_history[-1]['area']
        return 0.0
    
    def get_visualization_data(self, crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion) -> Dict[str, Any]:
        """
        Получить данные визуализации
        Возвращает словарь со всеми элементами визуализации
        """
        if crosswalk_mask is None:
            return {}
        
        # Определить текущий этап (везде используется оранжевый)
        if area_ratio >= self.THRESHOLDS['arrival']:
            stage = "到达"
            stage_color = "rgba(255, 165, 0, 0.5)"  # оранжевый
        elif area_ratio >= self.THRESHOLDS['near']:
            stage = "接近"
            stage_color = "rgba(255, 165, 0, 0.45)"  # оранжевый
        elif area_ratio >= self.THRESHOLDS['approaching']:
            stage = "靠近"
            stage_color = "rgba(255, 165, 0, 0.40)"  # оранжевый
        else:
            stage = "发现"
            stage_color = "rgba(255, 165, 0, 0.35)"  # оранжевый
        
        # Описание положения
        position = self._get_position_description(center_x_ratio)
        
        return {
            'area_ratio': area_ratio,
            'stage': stage,
            'stage_color': stage_color,
            'position': position.replace("在画面", ""),  # Убрать префикс "在画面" («в кадре»)
            'center_x_ratio': center_x_ratio,
            'center_y_ratio': center_y_ratio,
            'has_occlusion': has_occlusion,
            'in_arrival': self.in_arrival_state
        }


# Вспомогательные функции
def split_combined_voice(combined_text: str) -> list:
    """
    Разбить составную голосовую подсказку на несколько отдельных
    Например: "远处发现斑马线,在画面左侧" → ["远处发现斑马线", "在画面左侧"]
    """
    if ',' in combined_text:
        parts = combined_text.split(',')
        return [p.strip() for p in parts if p.strip()]
    return [combined_text]

