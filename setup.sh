#!/bin/bash
# AI Glass System - Linux/macOS скрипт быстрой установки

set -e  # немедленный выход при ошибке

echo "=========================================="
echo "  AI Glass System - скрипт автоматической установки"
echo "=========================================="
echo ""

# Определение цветов
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Проверка версии Python
echo "Проверка версии Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Ошибка: Python 3 не найден${NC}"
    echo "Сначала установите Python 3.9-3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Найден Python $PYTHON_VERSION${NC}"

# Проверяем, входит ли версия Python в поддерживаемый диапазон
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ] || [ "$PYTHON_MINOR" -gt 11 ]; then
    echo -e "${YELLOW}Внимание: версия Python $PYTHON_VERSION может не поддерживаться${NC}"
    echo "Рекомендуется Python 3.9-3.11"
    read -p "Продолжить? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Проверка CUDA (опционально)
echo ""
echo "Проверка CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ Обнаружен NVIDIA GPU${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    HAS_GPU=true
else
    echo -e "${YELLOW}! NVIDIA GPU не обнаружен, будет использован режим CPU (медленнее)${NC}"
    HAS_GPU=false
fi

# Создание виртуального окружения
echo ""
echo "Создание виртуального окружения..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Виртуальное окружение уже существует${NC}"
    read -p "Удалить и создать заново? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Виртуальное окружение создано заново${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Виртуальное окружение создано${NC}"
fi

# Активация виртуального окружения
echo "Активация виртуального окружения..."
source venv/bin/activate

# Обновление pip
echo ""
echo "Обновление pip..."
pip install --upgrade pip -q
echo -e "${GREEN}✓ pip обновлён${NC}"

# Установка PyTorch
echo ""
echo "Установка PyTorch..."
if [ "$HAS_GPU" = true ]; then
    echo "Установка GPU-версии PyTorch (CUDA 11.8)..."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo "Установка CPU-версии PyTorch..."
    pip install torch torchvision -q
fi
echo -e "${GREEN}✓ PyTorch установлен${NC}"

# Проверка PyTorch
echo "Проверка установки PyTorch..."
python3 -c "import torch; print(f'Версия PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}')"

# Установка системных зависимостей (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo ""
    echo "Проверка системных зависимостей..."
    
    # Определение дистрибутива
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS="unknown"
    fi
    
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        echo "Обнаружена система Ubuntu/Debian"
        echo "Для установки системных зависимостей могут понадобиться права sudo..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq portaudio19-dev libgl1-mesa-glx libglib2.0-0
        echo -e "${GREEN}✓ Системные зависимости установлены${NC}"
    else
        echo -e "${YELLOW}! Неизвестный дистрибутив Linux, установите зависимости вручную${NC}"
        echo "  Требуются: portaudio19-dev, libgl1-mesa-glx, libglib2.0-0"
    fi
fi

# Установка зависимостей Python
echo ""
echo "Установка зависимостей Python..."
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Зависимости Python установлены${NC}"

# Создание файла .env
echo ""
if [ ! -f ".env" ]; then
    echo "Создание конфигурационного файла .env..."
    cp .env.example .env
    echo -e "${GREEN}✓ Файл .env создан${NC}"
    echo -e "${YELLOW}Отредактируйте файл .env и укажите свой DASHSCOPE_API_KEY${NC}"
else
    echo -e "${YELLOW}Файл .env уже существует, пропускаем${NC}"
fi

# Создание необходимых директорий
echo ""
echo "Создание структуры директорий..."
mkdir -p recordings model music voice
echo -e "${GREEN}✓ Структура директорий создана${NC}"

# Проверка файлов моделей
echo ""
echo "Проверка файлов моделей..."
MODELS=("yolo-seg.pt" "yoloe-11l-seg.pt" "shoppingbest5.pt" "trafficlight.pt" "hand_landmarker.task")
MISSING_MODELS=()

for model in "${MODELS[@]}"; do
    if [ -f "model/$model" ]; then
        echo -e "${GREEN}✓ $model${NC}"
    else
        echo -e "${RED}✗ $model (отсутствует)${NC}"
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Внимание: отсутствуют следующие файлы моделей:${NC}"
    for model in "${MISSING_MODELS[@]}"; do
        echo "  - $model"
    done
    echo "Поместите файлы моделей в директорию model/"
fi

# Готово
echo ""
echo "=========================================="
echo -e "${GREEN}Установка завершена!${NC}"
echo "=========================================="
echo ""
echo "Дальнейшие шаги:"
echo "1. Отредактируйте файл .env и укажите свой API-ключ:"
echo "   nano .env"
echo ""
echo "2. Убедитесь, что все файлы моделей помещены в директорию model/"
echo ""
echo "3. Запустите систему:"
echo "   source venv/bin/activate"
echo "   python app_main.py"
echo ""
echo "4. Откройте http://localhost:8081"
echo ""

# Напоминание об активации виртуального окружения
echo -e "${YELLOW}Примечание: перед каждым использованием активируйте виртуальное окружение:${NC}"
echo "  source venv/bin/activate"
echo ""

