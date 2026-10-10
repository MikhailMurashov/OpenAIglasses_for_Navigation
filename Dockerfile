# AI Glass System - Dockerfile
# Образ Python на базе NVIDIA CUDA

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Установка переменных окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Установка рабочей директории
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip
RUN python3 -m pip install --upgrade pip

# Копирование requirements.txt
COPY requirements.txt .

# Установка зависимостей Python
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Создание необходимых директорий
RUN mkdir -p recordings model music voice static templates

# Открытие порта
EXPOSE 8081 12345/udp

# Проверка работоспособности
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8081/api/health || exit 1

# Команда запуска
CMD ["python3", "app_main.py"]

