@echo off
chcp 65001 >nul
REM AI Glass System - скрипт быстрой установки для Windows

echo ==========================================
echo   AI Glass System - скрипт автоматической установки
echo ==========================================
echo.

REM Проверка Python
echo Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден
    echo Скачайте и установите Python 3.9-3.11 с https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo [УСПЕХ] Python установлен

REM Проверка CUDA
echo.
echo Проверка CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ВНИМАНИЕ] NVIDIA GPU не обнаружен, будет использован режим CPU — работает медленнее
    set HAS_GPU=0
) else (
    echo [УСПЕХ] Обнаружен NVIDIA GPU
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    set HAS_GPU=1
)

REM Создание виртуального окружения
echo.
echo Создание виртуального окружения...
if exist venv (
    echo [ВНИМАНИЕ] Виртуальное окружение уже существует
    set /p RECREATE="Удалить и создать заново? (y/n): "
    if /i "%RECREATE%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo [УСПЕХ] Виртуальное окружение создано заново
    )
) else (
    python -m venv venv
    echo [УСПЕХ] Виртуальное окружение создано
)

REM Активация виртуального окружения
echo.
echo Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Обновление pip
echo.
echo Обновление pip...
python -m pip install --upgrade pip -q
echo [УСПЕХ] pip обновлён

REM Установка PyTorch
echo.
echo Установка PyTorch...
if %HAS_GPU%==1 (
    echo Установка GPU-версии PyTorch ^(CUDA 11.8^)...
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 -q
) else (
    echo Установка CPU-версии PyTorch...
    pip install torch torchvision -q
)
echo [УСПЕХ] PyTorch установлен

REM Проверка PyTorch
echo.
echo Проверка установки PyTorch...
python -c "import torch; print(f'Версия PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}')"

REM Установка PyAudio
echo.
echo Установка PyAudio...
echo [ВНИМАНИЕ] В Windows PyAudio может потребовать ручной установки
echo Если автоматическая установка не удастся, скачайте wheel-файл по адресу:
echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
echo.
pip install pyaudio -q
if errorlevel 1 (
    echo [ВНИМАНИЕ] Не удалось автоматически установить PyAudio, установите вручную
) else (
    echo [УСПЕХ] PyAudio установлен
)

REM Установка остальных зависимостей
echo.
echo Установка зависимостей Python...
pip install -r requirements.txt -q
echo [УСПЕХ] Зависимости Python установлены

REM Создание файла .env
echo.
if not exist .env (
    echo Создание конфигурационного файла .env...
    copy .env.example .env >nul
    echo [УСПЕХ] Файл .env создан
    echo [ПОДСКАЗКА] Отредактируйте файл .env и укажите свой DASHSCOPE_API_KEY
) else (
    echo [ПРОПУСК] Файл .env уже существует
)

REM Создание необходимых директорий
echo.
echo Создание структуры директорий...
if not exist recordings mkdir recordings
if not exist model mkdir model
if not exist music mkdir music
if not exist voice mkdir voice
echo [УСПЕХ] Структура директорий создана

REM Проверка файлов моделей
echo.
echo Проверка файлов моделей...
set MISSING=0
if exist model\yolo-seg.pt (echo [УСПЕХ] yolo-seg.pt) else (echo [НЕТ ФАЙЛА] yolo-seg.pt & set MISSING=1)
if exist model\yoloe-11l-seg.pt (echo [УСПЕХ] yoloe-11l-seg.pt) else (echo [НЕТ ФАЙЛА] yoloe-11l-seg.pt & set MISSING=1)
if exist model\shoppingbest5.pt (echo [УСПЕХ] shoppingbest5.pt) else (echo [НЕТ ФАЙЛА] shoppingbest5.pt & set MISSING=1)
if exist model\trafficlight.pt (echo [УСПЕХ] trafficlight.pt) else (echo [НЕТ ФАЙЛА] trafficlight.pt & set MISSING=1)
if exist model\hand_landmarker.task (echo [УСПЕХ] hand_landmarker.task) else (echo [НЕТ ФАЙЛА] hand_landmarker.task & set MISSING=1)

if %MISSING%==1 (
    echo.
    echo [ВНИМАНИЕ] Некоторые файлы моделей отсутствуют, поместите их в директорию model\
)

REM Готово
echo.
echo ==========================================
echo [УСПЕХ] Установка завершена!
echo ==========================================
echo.
echo Дальнейшие шаги:
echo 1. Отредактируйте файл .env и укажите свой API-ключ:
echo    notepad .env
echo.
echo 2. Убедитесь, что все файлы моделей помещены в директорию model\
echo.
echo 3. Запустите систему:
echo    venv\Scripts\activate
echo    python app_main.py
echo.
echo 4. Откройте http://localhost:8081
echo.
echo [ПОДСКАЗКА] Перед каждым использованием активируйте виртуальное окружение:
echo   venv\Scripts\activate
echo.

pause

