# audio_player.py
# Воспроизведение заранее записанных аудиофайлов через динамик ESP32

import os
import wave
import json
import asyncio
import threading
import queue
import time
from audio_stream import broadcast_pcm16_realtime
from audio_compressor import compressed_audio_cache, AudioCompressor

# Импорт рекордера (во избежание циклического импорта — динамически, по необходимости)
_recorder_imported = False
_sync_recorder = None

def _get_recorder():
    """Отложенный импорт рекордера"""
    global _recorder_imported, _sync_recorder
    if not _recorder_imported:
        try:
            import sync_recorder as sr
            _sync_recorder = sr
            _recorder_imported = True
        except Exception as e:
            print(f"[AUDIO] Не удалось импортировать рекордер: {e}")
            _recorder_imported = True  # Отмечаем, что попытка была, чтобы не повторять
    return _sync_recorder

# Совместимость с примерами аудио из старого проекта (сохранено)
AUDIO_BASE_DIR = r"C:\Users\Administrator\Desktop\rebuild1002\music"

# Новое: каталог voice и таблица соответствий
# Используем папку voice в каталоге скрипта, чтобы не зависеть от рабочего каталога
VOICE_DIR = os.getenv("VOICE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice"))
VOICE_MAP_FILE = os.path.join(VOICE_DIR, "map.zh-CN.json")

# Соответствие аудиофайлов (будет объединено с соответствиями voice)
AUDIO_MAP = {
    "检测到物体": os.path.join(AUDIO_BASE_DIR, "音频1.wav"),
    "向上": os.path.join(AUDIO_BASE_DIR, "音频2.wav"),
    "向下": os.path.join(AUDIO_BASE_DIR, "音频3.wav"),
    "向左": os.path.join(AUDIO_BASE_DIR, "音频4.wav"),
    "向右": os.path.join(AUDIO_BASE_DIR, "音频5.wav"),
    "OK": os.path.join(AUDIO_BASE_DIR, "音频6.wav"),
    "向前": os.path.join(AUDIO_BASE_DIR, "音频7.wav"),
    "后退": os.path.join(AUDIO_BASE_DIR, "音频8.wav"),
    "拿到物体": os.path.join(AUDIO_BASE_DIR, "音频9.wav"),
}

# Кэш аудио, чтобы не читать файлы повторно
_audio_cache = {}

# Очередь воспроизведения аудио и рабочий поток — используется очередь с приоритетом
_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0  # Возрастающий счётчик приоритета
_worker_thread = None
_worker_loop = None
_is_playing = False  # Флаг: идёт ли сейчас воспроизведение аудио
_playing_lock = threading.Lock()  # Блокировка воспроизведения
_initialized = False
_last_play_ts = 0.0  # Время окончания последнего воспроизведения — для выбора длины прогревочной тишины

def load_wav_file(filepath):
    """Загрузить WAV-файл и вернуть PCM-данные (автоматически преобразуются в 8 кГц)"""
    if filepath in _audio_cache:
        return _audio_cache[filepath]
    
    # Используем сжатый кэш
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        compressed_data = compressed_audio_cache.load_and_compress(filepath)
        if compressed_data:
            # Сохраняем сжатые данные
            _audio_cache[filepath] = compressed_data
            return compressed_data
    
    # Исходный способ загрузки (без сжатия)
    try:
        with wave.open(filepath, 'rb') as wav:
            # Проверяем формат аудио
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            
            if channels != 1:
                print(f"[AUDIO] Предупреждение: {filepath} не моно, будет использован только первый канал")
            if sampwidth != 2:
                print(f"[AUDIO] Предупреждение: {filepath} не 16-битное аудио")
            
            # Читаем все кадры
            frames = wav.readframes(wav.getnframes())
            
            # Если стерео — берём только левый канал
            if channels == 2:
                import audioop
                frames = audioop.tomono(frames, sampwidth, 1, 0)
            
            # Приводим к 8 кГц (ratecv сохраняет высоту тона и скорость)
            if framerate != 8000:
                import audioop
                frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                print(f"[AUDIO] Передискретизация: {filepath} {framerate}Hz -> 8000Hz")
            
            _audio_cache[filepath] = frames
            return frames
            
    except Exception as e:
        print(f"[AUDIO] Не удалось загрузить аудиофайл {filepath}: {e}")
        return None

def _merge_voice_map():
    """Прочитать voice/map.zh-CN.json и объединить с AUDIO_MAP"""
    try:
        if not os.path.exists(VOICE_MAP_FILE):
            print(f"[AUDIO] Файл соответствий не найден: {VOICE_MAP_FILE}")
            return
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)
        added = 0
        for text, info in (m or {}).items():
            files = (info or {}).get("files") or []
            if not files:
                continue
            fname = files[0]
            fpath = os.path.join(VOICE_DIR, fname)
            if os.path.exists(fpath):
                AUDIO_MAP[text] = fpath
                added += 1
            else:
                print(f"[AUDIO] Отсутствует файл из таблицы соответствий: {fpath}")
        print(f"[AUDIO] Объединено соответствий voice: {added}")
    except Exception as e:
        print(f"[AUDIO] Не удалось прочитать соответствия voice: {e}")

def preload_all_audio():
    """Предзагрузить все аудиофайлы в память"""
    print("[AUDIO] Начинаем предзагрузку аудиофайлов...")
    loaded_count = 0
    
    # 【Изменение скорости временно отключено】требуется доработка механизма кэша
    # Список аудио, которые нужно ускорить (связанные с пешеходным переходом)
    # speedup_keywords = ["斑马线", "画面"]
    # speedup_factor = 1.3  # ускорение на 30%
    
    for audio_key, filepath in AUDIO_MAP.items():
        if os.path.exists(filepath):
            # 【Исправление】временно загружаем с обычной скоростью
            # need_speedup = any(keyword in audio_key for keyword in speedup_keywords)
            # speed = speedup_factor if need_speedup else 1.0
            
            data = load_wav_file(filepath)  # Параметры по умолчанию
            if data:
                loaded_count += 1
                # if need_speedup:
                #     print(f"[AUDIO] Загрузка (ускорение {speedup_factor}x): {audio_key}")
        else:
            # Уменьшаем лишний вывод в лог
            pass
    print(f"[AUDIO] Предзагрузка завершена, загружено аудиофайлов: {loaded_count}")

def _audio_worker():
    """Рабочий поток воспроизведения аудио"""
    global _worker_loop
    
    # Пытаемся задать приоритет потока (только для Windows)
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            # Устанавливаем потоку высокий приоритет
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                1  # THREAD_PRIORITY_ABOVE_NORMAL
            )
            print("[AUDIO] Аудиопотоку установлен высокий приоритет")
    except Exception as e:
        print(f"[AUDIO] Не удалось установить приоритет потока: {e}")
    
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)
    
    async def process_queue():
        while True:
            try:
                # Получаем данные из очереди с приоритетом
                priority_data = await asyncio.get_event_loop().run_in_executor(None, _audio_queue.get, True)
                if priority_data is None:
                    break
                # Распаковываем приоритет и сами аудиоданные
                if isinstance(priority_data, tuple) and len(priority_data) == 2:
                    _, audio_data = priority_data
                else:
                    audio_data = priority_data
                await _broadcast_audio_optimized(audio_data)
            except Exception as e:
                print(f"[AUDIO] Ошибка рабочего потока: {e}")
    
    _worker_loop.run_until_complete(process_queue())

async def _broadcast_audio_optimized(pcm_data: bytes):
    """Оптимизированная трансляция аудио: один вызов, нижний уровень отправляет с тактом 20 мс; убраны повторные такты и sleep на уровне Python"""
    global _last_play_ts, _is_playing
    try:
        # Устанавливаем флаг воспроизведения
        with _playing_lock:
            _is_playing = True
        # К этому моменту pcm_data уже должны быть распакованными 16-битными PCM-данными (8 кГц)
        now = time.monotonic()
        idle_sec = now - (_last_play_ts or now)
        # В первый раз или после долгого простоя — более длинная прогревочная тишина; иначе короткая
        lead_ms = 160 if idle_sec > 3.0 else 60
        tail_ms = 40

        lead_silence = b'\x00' * (lead_ms * 8000 * 2 // 1000)  # 8k * 2B
        tail_silence = b'\x00' * (tail_ms * 8000 * 2 // 1000)

        # Полные аудиоданные (включая тишину)
        full_audio = lead_silence + pcm_data + tail_silence
        
        # Примечание: запись выполняется централизованно в broadcast_pcm16_realtime, чтобы избежать дублирования

        # Один вызов, темп задаёт нижний уровень (такт 20 мс реализован внутри broadcast_pcm16_realtime)
        await broadcast_pcm16_realtime(full_audio)

        _last_play_ts = time.monotonic()
    except Exception as e:
        print(f"[AUDIO] Ошибка трансляции аудио: {e}")
    finally:
        # Сбрасываем флаг воспроизведения
        with _playing_lock:
            _is_playing = False

def initialize_audio_system():
    """Инициализация аудиосистемы"""
    global _initialized, _worker_thread, _last_play_ts
    
    if _initialized:
        return
    
    # Сначала объединяем соответствия voice, затем предзагружаем
    _merge_voice_map()
    preload_all_audio()
    
    _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
    _worker_thread.start()
    _initialized = True
    _last_play_ts = 0.0
    
    # Выводим статистику сжатия
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        stats = compressed_audio_cache.get_compression_stats()
        print(f"[AUDIO] Статистика сжатия аудио:")
        print(f"  - Файлов: {stats['files_cached']}")
        print(f"  - Исходный размер: {stats['total_original_size'] / 1024:.1f} KB")
        print(f"  - После сжатия: {stats['total_compressed_size'] / 1024:.1f} KB")
        print(f"  - Степень сжатия: {stats['compression_ratio']:.1%}")
        print(f"  - Сэкономлено: {stats['bytes_saved'] / 1024:.1f} KB")
    
    print("[AUDIO] Аудиосистема инициализирована (предзагрузка + рабочий поток)")

def play_audio_threadsafe(audio_key):
    """Потокобезопасная функция воспроизведения аудио"""
    global _audio_queue, _audio_priority
    
    if not _initialized:
        initialize_audio_system()
    
    if audio_key not in AUDIO_MAP:
        print(f"[AUDIO] Неизвестный ключ аудио: {audio_key}")
        return
    
    filepath = AUDIO_MAP[audio_key]
    pcm_data = _audio_cache.get(filepath)
    if pcm_data is None:
        print(f"[AUDIO] Аудио отсутствует в кэше: {audio_key}")
        return
    
    # Если данные сжаты — сначала распаковываем
    if pcm_data and len(pcm_data) > 5 and pcm_data[0] in [0x01, 0x02]:
        pcm_data = compressed_audio_cache.decompress(pcm_data)
        if not pcm_data:
            print(f"[AUDIO] Ошибка распаковки: {audio_key}")
            return
    
    # 【Оптимизация】стратегия озвучки в реальном времени: держим очередь минимальной, чтобы не копилась задержка
    queue_size = _audio_queue.qsize()
    
    # Проверяем, идёт ли воспроизведение
    with _playing_lock:
        currently_playing = _is_playing
    
    # Стратегия реального времени: допускается только 1 ожидающий элемент, при превышении — сразу очищаем
    if queue_size > 0 and not currently_playing:
        # Если ничего не играет — сразу очищаем и воспроизводим самую свежую фразу
        print(f"[AUDIO] Очистка очереди (сейчас элементов: {queue_size}), воспроизводим самую свежую фразу")
        _audio_queue = queue.PriorityQueue(maxsize=10)
    elif queue_size > 1 and currently_playing:
        # Во время воспроизведения, если накопилось >1 элемента — очищаем (сохраняем реальное время)
        print(f"[AUDIO] Очередь переполнена (элементов: {queue_size}), очищаем для работы в реальном времени")
        _audio_queue = queue.PriorityQueue(maxsize=10)
    try:
        # Используем очередь с приоритетом, чтобы аудио воспроизводилось по порядку
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, pcm_data))
        if queue_size >= 1:
            print(f"[AUDIO] Текущий размер очереди воспроизведения: {queue_size + 1}")
    except queue.Full:
        # Если очередь воспроизведения заполнена — отбрасываем, сохраняя реальное время
        print(f"[AUDIO] Очередь заполнена, отброшено: {audio_key}")
        pass

# Глобальное ограничение частоты озвучки
_last_voice_time = 0
_last_voice_text = ""
_voice_cooldown = 1.0  # Одинаковые фразы — с интервалом не менее 1 секунды

# Определение приоритетов озвучки
VOICE_PRIORITY = {
    'obstacle': 100,     # Препятствие — наивысший приоритет
    'direction': 50,     # Поворот/смещение — средний приоритет  
    'straight': 10,      # Движение прямо — наименьший приоритет
    'other': 30          # Прочее — приоритет по умолчанию
}

# Новое: воспроизведение напрямую по тексту китайской подсказки (с лёгкой нормализацией и запасными вариантами)
def play_voice_text(text: str):
    """
    Принимает китайскую подсказку, автоматически находит соответствие в voice и воспроизводит.
    - Пробуем исходный текст
    - Пробуем добавить/убрать конечный знак препинания (。.!！?？)
    - Если текст содержит «前方有…注意避让», но не найден — запасной вариант «前方有障碍物，注意避让。»
    """
    global _last_voice_time, _last_voice_text
    
    if not text:
        return
    if not _initialized:
        initialize_audio_system()
    
    # Глобальное ограничение: один и тот же текст не повторяется в течение короткого времени
    current_time = time.time()
    if text == _last_voice_text and current_time - _last_voice_time < _voice_cooldown:
        return  # Тихо пропускаем

    candidates = []
    t = text.strip()
    candidates.append(t)
    # Пробуем добавить точку
    if t[-1:] not in ("。", "！", "!", "？", "?", "."):
        candidates.append(t + "。")
    else:
        # Пробуем убрать знаки препинания
        t2 = t.rstrip("。.!！?？")
        if t2 and t2 != t:
            candidates.append(t2)

    # Поочерёдно пробуем найти соответствие
    for ck in candidates:
        if ck in AUDIO_MAP:
            play_audio_threadsafe(ck)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    # Запасной вариант для «前方有…注意避让»
    if ("前方有" in t) and ("注意避让" in t):
        fallback = "前方有障碍物，注意避让。"
        if fallback in AUDIO_MAP:
            play_audio_threadsafe(fallback)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    # Для фраз вида «请向…平移/微调/转动» пробуем распространённые варианты
    base = t.rstrip("。.!！?？")
    if base in AUDIO_MAP:
        play_audio_threadsafe(base)
        _last_voice_text = text
        _last_voice_time = current_time
        return
    if base + "。" in AUDIO_MAP:
        play_audio_threadsafe(base + "。")
        _last_voice_text = text
        _last_voice_time = current_time
        return

    # Если соответствие не найдено — пишем в лог (для отладки)
    print(f"[AUDIO] Не найдено подходящей озвучки: {text}")

# Совместимость со старым интерфейсом
play_audio_on_esp32 = play_audio_threadsafe