# audio_stream.py
# -*- coding: utf-8 -*-
import asyncio
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Any, Dict
from fastapi import Request
from fastapi.responses import StreamingResponse

# ===== Базовые параметры нисходящего WAV-потока =====
STREAM_SR = 8000  # изменено на 8 кГц, поддерживается ESP32
STREAM_CH = 1
STREAM_SW = 2
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000  # 320B (8kHz)

# ===== Рубильник задачи воспроизведения AI =====
current_ai_task: Optional[asyncio.Task] = None

async def cancel_current_ai():
    """Отменяет текущую голосовую задачу большой модели и ждёт её завершения."""
    global current_ai_task
    task = current_ai_task
    current_ai_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

def is_playing_now() -> bool:
    t = current_ai_task
    return (t is not None) and (not t.done())

# ===== Управление подключениями /stream.wav =====
@dataclass(frozen=True)
class StreamClient:
    q: asyncio.Queue
    abort_event: asyncio.Event

stream_clients: "Set[StreamClient]" = set()
STREAM_QUEUE_MAX = 96  # небольшой буфер, чтобы избежать накопления

def _wav_header_unknown_size(sr=16000, ch=1, sw=2) -> bytes:
    import struct
    byte_rate = sr * ch * sw
    block_align = ch * sw
    data_size = 0x7FFFFFF0
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, sw * 8,
        b"data", data_size
    )

async def hard_reset_audio(reason: str = ""):
    """
    **Очистка в одно действие**: сбрасывает все подключения плееров (выставляет abort_event) + отменяет текущую задачу AI.
    Так старому аудио больше некуда идти, и ни одна задача не продолжает его генерировать.
    """
    # 1) Разрываем все HTTP-подключения, по которым идёт воспроизведение
    for sc in list(stream_clients):
        try:
            sc.abort_event.set()
        except Exception:
            pass
    stream_clients.clear()

    # 2) Отменяем текущую задачу AI
    await cancel_current_ai()

    # 3) Лог
    if reason:
        print(f"[HARD-RESET] {reason}")

async def broadcast_pcm16_realtime(pcm16: bytes):
    """Отправляет pcm16 всем ещё живым подключениям с шагом 20 мс; при переполнении очереди хвост отбрасывается, чтобы сохранить реальное время."""
    # [Новое] запись аудио (записываем целиком до рассылки, чтобы избежать фрагментации)
    try:
        import sync_recorder
        sync_recorder.record_audio(pcm16, text="[Omni-диалог]")
    except Exception:
        pass  # тихий сбой, не влияет на воспроизведение
    
    loop = asyncio.get_event_loop()
    next_tick = loop.time()
    off = 0
    while off < len(pcm16):
        take = min(BYTES_PER_20MS_16K, len(pcm16) - off)
        piece = pcm16[off:off + take]

        dead: List[StreamClient] = []
        for sc in list(stream_clients):
            if sc.abort_event.is_set():
                dead.append(sc)
                continue
            try:
                if sc.q.full():
                    try: sc.q.get_nowait()
                    except Exception: pass
                sc.q.put_nowait(piece)
            except Exception:
                dead.append(sc)
        for sc in dead:
            try: stream_clients.discard(sc)
            except Exception: pass

        next_tick += 0.020
        now = loop.time()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
        off += take

# ===== Регистратор маршрутов FastAPI =====
def register_stream_route(app):
    @app.get("/stream.wav")
    async def stream_wav(_: Request):
        # —— Принудительно одно подключение (или немного), сначала обрываем все старые подключения ——
        for sc in list(stream_clients):
            try: sc.abort_event.set()
            except Exception: pass
        stream_clients.clear()

        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        sc = StreamClient(q=q, abort_event=abort_event)
        stream_clients.add(sc)

        async def gen():
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        break
                    if chunk:
                        yield chunk
            finally:
                stream_clients.discard(sc)
        return StreamingResponse(gen(), media_type="audio/wav")