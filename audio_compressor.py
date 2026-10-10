# audio_compressor.py
# -*- coding: utf-8 -*-
"""
Инструмент сжатия аудио — для снижения нагрузки на сеть
Поддерживает сжатие PCM 16 кГц 16 бит в более компактные форматы
"""
import os
import wave
import struct
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioCompressor:
    """Компрессор аудио — поддерживает несколько алгоритмов сжатия"""
    
    @staticmethod
    def pcm16_to_ulaw(pcm_data: bytes) -> bytes:
        """
        Преобразовать 16-битный PCM в 8-битный μ-law
        Степень сжатия: 50% (16 бит -> 8 бит)
        """
        # Разбираем 16-битный PCM
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Сжатие μ-law
        ulaw_data = bytearray()
        for sample in samples:
            ulaw_byte = AudioCompressor._linear_to_ulaw(sample)
            ulaw_data.append(ulaw_byte)
        
        return bytes(ulaw_data)
    
    @staticmethod
    def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
        """
        Преобразовать 8-битный μ-law обратно в 16-битный PCM
        """
        pcm_samples = []
        for ulaw_byte in ulaw_data:
            pcm_sample = AudioCompressor._ulaw_to_linear(ulaw_byte)
            pcm_samples.append(pcm_sample)
        
        return np.array(pcm_samples, dtype=np.int16).tobytes()
    
    @staticmethod
    def _linear_to_ulaw(sample: int) -> int:
        """
        16-битный линейный PCM в μ-law
        """
        # Таблица кодирования μ-law
        ULAW_MAX = 0x1FFF
        ULAW_BIAS = 0x84
        
        # Ограничиваем диапазон
        sample = max(-32768, min(32767, sample))
        
        # Получаем знаковый бит
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample
        
        # Добавляем смещение
        sample = sample + ULAW_BIAS
        
        # Ограничиваем максимальное значение
        if sample > ULAW_MAX:
            sample = ULAW_MAX
        
        # Находим экспоненту и мантиссу
        exponent = 7
        for exp in range(7, -1, -1):
            if sample & (0x4000 >> exp):
                exponent = exp
                break
        
        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulawbyte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        
        return ulawbyte
    
    @staticmethod
    def _ulaw_to_linear(ulawbyte: int) -> int:
        """
        μ-law в 16-битный линейный PCM
        """
        ULAW_BIAS = 0x84
        
        ulawbyte = ~ulawbyte & 0xFF
        sign = ulawbyte & 0x80
        exponent = (ulawbyte >> 4) & 0x07
        mantissa = ulawbyte & 0x0F
        
        sample = ((mantissa << 3) + ULAW_BIAS) << exponent
        
        if sign:
            sample = -sample
            
        return sample
    
    @staticmethod
    def pcm16_to_adpcm(pcm_data: bytes) -> bytes:
        """
        Преобразовать 16-битный PCM в 4-битный ADPCM
        Степень сжатия: 75% (16 бит -> 4 бита)
        Сохраняет хорошее качество речи
        """
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Таблица шагов IMA ADPCM
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]
        
        # Таблица корректировки индекса
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]
        
        # Инициализация
        adpcm_data = bytearray()
        predicted = 0
        step_index = 0
        
        # Каждые два отсчёта упаковываются в один байт
        for i in range(0, len(samples), 2):
            byte = 0
            
            for j in range(2):
                if i + j < len(samples):
                    sample = samples[i + j]
                    
                    # Вычисляем разность
                    diff = sample - predicted
                    
                    # Квантование
                    step = step_table[step_index]
                    adpcm_sample = 0
                    
                    if diff < 0:
                        adpcm_sample = 8
                        diff = -diff
                    
                    if diff >= step:
                        adpcm_sample |= 4
                        diff -= step
                        
                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 2
                        diff -= step
                        
                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 1
                    
                    # Обновляем предсказанное значение
                    step = step_table[step_index]
                    diff = 0
                    if adpcm_sample & 4:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 2:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 1:
                        diff += step
                    step >>= 1
                    diff += step
                    
                    if adpcm_sample & 8:
                        predicted -= diff
                    else:
                        predicted += diff
                    
                    # Ограничиваем диапазон предсказанного значения
                    if predicted > 32767:
                        predicted = 32767
                    elif predicted < -32768:
                        predicted = -32768
                    
                    # Обновляем индекс шага
                    step_index += index_table[adpcm_sample & 7]
                    if step_index < 0:
                        step_index = 0
                    elif step_index > 88:
                        step_index = 88
                    
                    # Упаковываем в байт
                    if j == 0:
                        byte = adpcm_sample
                    else:
                        byte |= (adpcm_sample << 4)
            
            adpcm_data.append(byte)
        
        # Добавляем заголовок: начальное предсказанное значение и индекс шага
        header = struct.pack('<hB', predicted, step_index)
        return header + bytes(adpcm_data)
    
    @staticmethod
    def adpcm_to_pcm16(adpcm_data: bytes) -> bytes:
        """
        Преобразовать 4-битный ADPCM обратно в 16-битный PCM
        """
        if len(adpcm_data) < 3:
            return b''
        
        # Читаем заголовок
        predicted, step_index = struct.unpack('<hB', adpcm_data[:3])
        adpcm_bytes = adpcm_data[3:]
        
        # Таблица шагов IMA ADPCM
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]
        
        # Таблица корректировки индекса
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]
        
        pcm_samples = []
        
        for byte in adpcm_bytes:
            # Декодируем два 4-битных отсчёта
            for shift in [0, 4]:
                adpcm_sample = (byte >> shift) & 0x0F
                
                # Вычисляем разность
                step = step_table[step_index]
                diff = 0
                
                if adpcm_sample & 4:
                    diff += step
                step >>= 1
                if adpcm_sample & 2:
                    diff += step
                step >>= 1
                if adpcm_sample & 1:
                    diff += step
                step >>= 1
                diff += step
                
                if adpcm_sample & 8:
                    predicted -= diff
                else:
                    predicted += diff
                
                # Ограничиваем диапазон
                if predicted > 32767:
                    predicted = 32767
                elif predicted < -32768:
                    predicted = -32768
                
                pcm_samples.append(predicted)
                
                # Обновляем индекс шага
                step_index += index_table[adpcm_sample & 7]
                if step_index < 0:
                    step_index = 0
                elif step_index > 88:
                    step_index = 88
        
        return np.array(pcm_samples, dtype=np.int16).tobytes()
    
    @staticmethod
    def downsample_pcm16(pcm_data: bytes, from_rate: int = 16000, to_rate: int = 8000) -> bytes:
        """
        Понижение частоты дискретизации (опционально)
        16 кГц -> 8 кГц позволяет сократить объём данных ещё на 50%
        """
        if from_rate == to_rate:
            return pcm_data
            
        # Разбираем PCM-данные
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Простое понижение частоты (берём каждый второй отсчёт)
        if from_rate == 16000 and to_rate == 8000:
            downsampled = samples[::2]
        else:
            # Для более сложной передискретизации нужен scipy
            ratio = to_rate / from_rate
            new_length = int(len(samples) * ratio)
            downsampled = np.interp(
                np.linspace(0, len(samples) - 1, new_length),
                np.arange(len(samples)),
                samples
            ).astype(np.int16)
        
        return downsampled.tobytes()


class CompressedAudioCache:
    """Кэш сжатого аудио"""
    
    def __init__(self, compression_type: str = "adpcm", use_downsample: bool = False):
        """
        compression_type: "none", "ulaw", "adpcm"
        """
        self.compression_type = compression_type
        self.use_downsample = use_downsample
        self._cache = {}  # {filepath: compressed_data}
        self._original_sizes = {}  # {filepath: original_size}
        
    def load_and_compress(self, filepath: str) -> Optional[bytes]:
        """Загрузить и сжать аудиофайл (с приведением к 8 кГц)"""
        if filepath in self._cache:
            return self._cache[filepath]
        
        try:
            with wave.open(filepath, 'rb') as wav:
                # Проверяем формат
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                
                if channels != 1:
                    logger.warning(f"{filepath} не моно")
                if sampwidth != 2:
                    logger.warning(f"{filepath} не 16-битное аудио")
                
                # Читаем все данные
                frames = wav.readframes(wav.getnframes())
                
                # Если стерео — преобразуем в моно
                if channels == 2:
                    import audioop
                    frames = audioop.tomono(frames, sampwidth, 1, 0)
                
                # 【Изменено】всегда приводим к 8 кГц (ratecv сохраняет высоту тона и скорость)
                if framerate != 8000:
                    import audioop
                    frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                    framerate = 8000
                
                # Запоминаем исходный размер (размер после преобразования)
                self._original_sizes[filepath] = len(frames)
                
                # Сжатие
                if self.compression_type == "ulaw":
                    compressed = AudioCompressor.pcm16_to_ulaw(frames)
                    # Добавляем простой заголовок (1 байт — тип + 4 байта — исходная длина)
                    header = struct.pack('!BI', 0x01, len(frames))  # 0x01 означает μ-law
                    compressed = header + compressed
                elif self.compression_type == "adpcm":
                    compressed = AudioCompressor.pcm16_to_adpcm(frames)
                    # Добавляем простой заголовок (1 байт — тип + 4 байта — исходная длина)
                    header = struct.pack('!BI', 0x02, len(frames))  # 0x02 означает ADPCM
                    compressed = header + compressed
                else:
                    compressed = frames
                
                self._cache[filepath] = compressed
                
                # Выводим степень сжатия
                compression_ratio = len(compressed) / self._original_sizes[filepath]
                logger.info(f"[Сжатие] {os.path.basename(filepath)}: "
                          f"{self._original_sizes[filepath]} -> {len(compressed)} bytes "
                          f"({compression_ratio:.1%})")
                
                return compressed
                
        except Exception as e:
            logger.error(f"Не удалось сжать аудио {filepath}: {e}")
            return None
    
    def decompress(self, compressed_data: bytes) -> Optional[bytes]:
        """Распаковать аудиоданные"""
        if not compressed_data or len(compressed_data) < 5:
            return compressed_data
        
        try:
            # Проверяем заголовок
            compression_type = compressed_data[0]
            if compression_type == 0x01:  # Признак μ-law
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                ulaw_data = compressed_data[header_size:]
                
                # Распаковка μ-law
                pcm_data = AudioCompressor.ulaw_to_pcm16(ulaw_data)
                
                return pcm_data
            elif compression_type == 0x02:  # Признак ADPCM
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                adpcm_data = compressed_data[header_size:]
                
                # Распаковка ADPCM
                pcm_data = AudioCompressor.adpcm_to_pcm16(adpcm_data)
                
                return pcm_data
            else:
                # Несжатые данные
                return compressed_data
                
        except Exception as e:
            logger.error(f"Не удалось распаковать аудио: {e}")
            return compressed_data
    
    def get_compression_stats(self) -> dict:
        """Получить статистику сжатия"""
        total_original = sum(self._original_sizes.values())
        total_compressed = sum(len(data) for data in self._cache.values())
        
        return {
            "files_cached": len(self._cache),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "compression_ratio": total_compressed / total_original if total_original > 0 else 0,
            "bytes_saved": total_original - total_compressed
        }


# Глобальный экземпляр кэша сжатого аудио
# По умолчанию используется сжатие ADPCM: качество звука лучше, а степень сжатия тоже неплохая (75%)
# Можно задать через переменную окружения AIGLASS_COMPRESS_TYPE: none, ulaw, adpcm
import os
compression_type = os.getenv("AIGLASS_COMPRESS_TYPE", "adpcm").lower()
if compression_type not in ["none", "ulaw", "adpcm"]:
    compression_type = "adpcm"
compressed_audio_cache = CompressedAudioCache(compression_type=compression_type, use_downsample=False) 