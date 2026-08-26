#!/bin/bash
echo "========================================"
echo "PD Signal Analysis - Автозапуск"
echo "========================================"
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "[ОШИБКА] Python не установлен!"
    exit 1
fi

# Создание виртуального окружения
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активация
source venv/bin/activate

# Установка зависимостей
echo "Установка зависимостей..."
pip install -r requirements.txt --quiet

# Запуск сервера
echo ""
echo "Запуск сервера..."
echo "Откройте браузер: http://localhost:8000"
echo "Для остановки нажмите Ctrl+C"
echo ""

cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000