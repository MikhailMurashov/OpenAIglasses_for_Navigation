# sync_recorder.py
# Синхронная запись видеопотока ESP32 и голосовых подсказок
# Автоматически выравнивает временные шкалы видео и аудио

import os
import cv2
import wave
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque
import struct

class SyncRecorder:
    """Синхронный рекордер — выравнивание видео и аудио по времени"""
    
    def __init__(self, output_dir="recordings", fps=15.0):
        """
        Инициализация рекордера
        :param output_dir: каталог для вывода
        :param fps: частота кадров видео (по умолчанию 15 fps)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.frame_duration = 1.0 / fps  # Длительность кадра (сек)
        
        # Создаём каталог для вывода
        os.makedirs(output_dir, exist_ok=True)
        
        # Состояние записи
        self.is_recording = False
        self.start_time = None
        
        # Запись видео
        self.video_writer = None
        self.video_path = None
        self.last_frame = None
        self.frame_count = 0
        
        # Запись аудио
        self.audio_writer = None
        self.audio_path = None
        self.audio_buffer = bytearray()
        self.last_audio_time = 0.0
        
        # Параметры аудио (стандарт ESP32: 16 кГц, 16 бит, моно)
        self.sample_rate = 16000
        self.sample_width = 2  # 16bit = 2 bytes
        self.channels = 1
        
        # Потокобезопасность
        self.lock = threading.Lock()
        
        # Мониторинг производительности
        self.frames_written = 0
        self.audio_bytes_written = 0
        self.last_log_time = time.time()
        
        print(f"[RECORDER] Рекордер инициализирован - FPS={fps}, каталог вывода={output_dir}")
    
    def start_recording(self):
        """Начать новый сеанс записи"""
        if self.is_recording:
            print("[RECORDER] Предупреждение: запись уже идёт")
            return False
        
        # Формируем имя файла (метка времени)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"video_{timestamp}.avi")
        self.audio_path = os.path.join(self.output_dir, f"audio_{timestamp}.wav")
        
        # Сбрасываем состояние
        self.start_time = time.time()
        self.last_audio_time = 0.0
        self.frame_count = 0
        self.frames_written = 0
        self.audio_bytes_written = 0
        self.audio_buffer.clear()
        self.last_frame = None
        
        # Инициализируем аудиофайл
        try:
            self.audio_writer = wave.open(self.audio_path, 'wb')
            self.audio_writer.setnchannels(self.channels)
            self.audio_writer.setsampwidth(self.sample_width)
            self.audio_writer.setframerate(self.sample_rate)
        except Exception as e:
            print(f"[RECORDER] Не удалось инициализировать аудиофайл: {e}")
            return False
        
        self.is_recording = True
        print(f"[RECORDER] Запись начата")
        print(f"  Видео: {self.video_path}")
        print(f"  Аудио: {self.audio_path}")
        return True
    
    def add_frame(self, jpeg_data: bytes):
        """
        Добавить кадр видео (исходные данные JPEG)
        :param jpeg_data: данные изображения в формате JPEG
        """
        if not self.is_recording:
            return
        
        try:
            with self.lock:
                # Декодируем JPEG
                arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"[RECORDER] Предупреждение: не удалось декодировать кадр")
                    return
                
                # Первый кадр: инициализируем запись видео
                if self.video_writer is None:
                    height, width = frame.shape[:2]
                    # Используем кодек XVID (хорошая совместимость с Windows)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(
                        self.video_path, 
                        fourcc, 
                        self.fps, 
                        (width, height)
                    )
                    
                    if not self.video_writer.isOpened():
                        print(f"[RECORDER] Ошибка: не удалось инициализировать запись видео")
                        self.is_recording = False
                        return
                    
                    print(f"[RECORDER] Запись видео инициализирована: {width}x{height} @ {self.fps}fps")
                
                # Записываем кадр
                self.video_writer.write(frame)
                self.frame_count += 1
                self.frames_written += 1
                self.last_frame = frame
                
                # Вычисляем текущую длительность видео (сек)
                current_video_time = self.frame_count * self.frame_duration
                
                # Синхронизация аудио: дополняем тишиной до длительности видео
                self._sync_audio_to_video(current_video_time)
                
                # Лог производительности (каждые 10 секунд)
                now = time.time()
                if now - self.last_log_time > 10.0:
                    elapsed = now - self.start_time
                    avg_fps = self.frames_written / elapsed if elapsed > 0 else 0
                    audio_duration = self.audio_bytes_written / (self.sample_rate * self.sample_width)
                    print(f"[RECORDER] Идёт запись - кадров={self.frames_written}, "
                          f"фактический FPS={avg_fps:.1f}, "
                          f"длительность видео={current_video_time:.1f}s, "
                          f"длительность аудио={audio_duration:.1f}s")
                    self.last_log_time = now
                    
        except Exception as e:
            print(f"[RECORDER] Не удалось добавить кадр: {e}")
            import traceback
            traceback.print_exc()
    
    def add_audio(self, pcm_data: bytes, text: str = ""):
        """
        Добавить аудиоданные (PCM 16 бит)
        :param pcm_data: аудиоданные в формате PCM
        :param text: текст фразы (для лога)
        """
        if not self.is_recording:
            return
        
        try:
            with self.lock:
                # Текущая длительность видео
                current_video_time = self.frame_count * self.frame_duration
                
                # Перед добавлением аудио дополняем тишиной до длительности видео
                self._sync_audio_to_video(current_video_time)
                
                # Записываем само аудио
                self.audio_writer.writeframes(pcm_data)
                audio_duration = len(pcm_data) / (self.sample_rate * self.sample_width)
                self.last_audio_time = current_video_time + audio_duration
                self.audio_bytes_written += len(pcm_data)
                
                if text:
                    print(f"[RECORDER] Записана фраза: {text[:30]}... (время={current_video_time:.2f}s, длительность={audio_duration:.2f}s)")
                    
        except Exception as e:
            print(f"[RECORDER] Не удалось добавить аудио: {e}")
    
    def _sync_audio_to_video(self, video_time: float):
        """
        Синхронизировать аудио с длительностью видео (дополнение тишиной)
        :param video_time: текущая длительность видео (сек)
        """
        # Вычисляем длительность тишины для дополнения
        silence_duration = video_time - self.last_audio_time
        
        if silence_duration > 0.01:  # Дополняем, только если больше 10 мс
            # Генерируем тишину
            silence_samples = int(silence_duration * self.sample_rate)
            silence_bytes = silence_samples * self.sample_width
            silence_data = b'\x00' * silence_bytes
            
            # Записываем тишину
            self.audio_writer.writeframes(silence_data)
            self.audio_bytes_written += len(silence_data)
            self.last_audio_time = video_time
    
    def stop_recording(self):
        """Остановить запись и сохранить файлы"""
        if not self.is_recording:
            return
        
        print("[RECORDER] Сохранение файлов записи...")
        self.is_recording = False
        
        with self.lock:
            # Последняя синхронизация аудио
            try:
                if self.frame_count > 0:
                    final_video_time = self.frame_count * self.frame_duration
                    self._sync_audio_to_video(final_video_time)
            except Exception as e:
                print(f"[RECORDER] Ошибка финальной синхронизации аудио: {e}")
            
            # Закрываем запись видео (ключевой шаг)
            if self.video_writer is not None:
                try:
                    print("[RECORDER] Закрытие записи видео...")
                    self.video_writer.release()
                    print("[RECORDER] Запись видео закрыта")
                except Exception as e:
                    print(f"[RECORDER] Не удалось закрыть запись видео: {e}")
                finally:
                    self.video_writer = None
            
            # Закрываем запись аудио
            if self.audio_writer is not None:
                try:
                    print("[RECORDER] Закрытие записи аудио...")
                    self.audio_writer.close()
                    print("[RECORDER] Запись аудио закрыта")
                except Exception as e:
                    print(f"[RECORDER] Не удалось закрыть запись аудио: {e}")
                finally:
                    self.audio_writer = None
            
            # Статистика
            try:
                elapsed = time.time() - self.start_time if self.start_time else 0
                video_duration = self.frame_count * self.frame_duration
                audio_duration = self.audio_bytes_written / (self.sample_rate * self.sample_width)
                
                print(f"\n{'='*60}")
                print(f"[RECORDER] Запись завершена")
                print(f"{'='*60}")
                print(f"  Общее время: {elapsed:.1f} с")
                print(f"\n  Видео: {self.video_path}")
                print(f"    - Кадров: {self.frames_written}")
                print(f"    - Длительность: {video_duration:.2f} с")
                if elapsed > 0:
                    print(f"    - Средний FPS: {self.frames_written/elapsed:.1f}")
                print(f"\n  Аудио: {self.audio_path}")
                print(f"    - Объём данных: {self.audio_bytes_written/1024:.1f} KB")
                print(f"    - Длительность: {audio_duration:.2f} с")
                print(f"\n  Расхождение по времени: {abs(video_duration - audio_duration):.3f} с")
                
                # Проверяем файлы
                if os.path.exists(self.video_path):
                    video_size = os.path.getsize(self.video_path) / 1024 / 1024
                    print(f"  Размер видеофайла: {video_size:.2f} MB ✓")
                else:
                    print(f"  ⚠ Предупреждение: видеофайл не создан")
                
                if os.path.exists(self.audio_path):
                    audio_size = os.path.getsize(self.audio_path) / 1024
                    print(f"  Размер аудиофайла: {audio_size:.2f} KB ✓")
                else:
                    print(f"  ⚠ Предупреждение: аудиофайл не создан")
                
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"[RECORDER] Не удалось вывести статистику: {e}")


# Глобальный экземпляр рекордера
_global_recorder = None
_recorder_lock = threading.Lock()

def get_recorder():
    """Получить глобальный экземпляр рекордера"""
    global _global_recorder
    with _recorder_lock:
        if _global_recorder is None:
            _global_recorder = SyncRecorder()
        return _global_recorder

def start_recording():
    """Запустить запись"""
    recorder = get_recorder()
    return recorder.start_recording()

def stop_recording():
    """Остановить запись"""
    recorder = get_recorder()
    recorder.stop_recording()

def record_frame(jpeg_data: bytes):
    """Записать кадр (для внешнего вызова)"""
    recorder = get_recorder()
    if recorder.is_recording:
        recorder.add_frame(jpeg_data)

def record_audio(pcm_data: bytes, text: str = ""):
    """Записать аудио (для внешнего вызова)"""
    recorder = get_recorder()
    if recorder.is_recording:
        recorder.add_audio(pcm_data, text)

