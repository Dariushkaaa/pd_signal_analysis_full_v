from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from models import model_manager
from processing import process_signal, recalculate_with_points

# Инициализация приложения
app = FastAPI(title="PD Signal Analysis API")

# Настройка CORS (чтобы фронтенд мог обращаться к бэкенду)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ГЛАВНАЯ СТРАНИЦА ====================
@app.get("/")
async def root():
    """
    Отдаёт файл index.html при запросе корня сайта.
    Если файл не найден — возвращает JSON с информацией о статусе.
    """
    # Путь к папке проекта (на уровень выше backend/)
    BASE_DIR = Path(__file__).parent.parent
    
    # Путь к файлу index.html
    frontend_path = BASE_DIR / "frontend" / "index.html"
    
    # Отладочный вывод в консоль сервера
    print(f"[root] Ищу файл: {frontend_path}")
    print(f"[root] Файл существует: {frontend_path.exists()}")
    
    # Если файл найден — отдаём его браузеру
    if frontend_path.exists():
        return FileResponse(
            frontend_path,
            media_type="text/html",
            headers={"Cache-Control": "no-cache"}
        )
    
    # Если файл не найден — возвращаем JSON (для отладки)
    return JSONResponse(
        status_code=200,
        content={
            "message": "PD Signal Analysis API",
            "status": "running",
            "models_loaded": model_manager.is_ready(),
            "error": f"Frontend file not found at {frontend_path}"
        }
    )


# ==================== СТАТУС МОДЕЛЕЙ ====================
@app.get("/api/models-status")
async def models_status():
    """Возвращает статус загрузки всех моделей"""
    return {
        "ready": model_manager.is_ready(),
        "detector": model_manager.detector_model is not None,
        "segmenter": model_manager.segmenter_model is not None,
        "classifier": model_manager.clf_model is not None,
        "feature_bounds": model_manager.feature_bounds is not None
    }


# ==================== ОБРАБОТКА СИГНАЛА ====================
@app.post("/api/process-signal")
async def process_signal_endpoint(
    acc_file: UploadFile = File(...),
    gyro_file: UploadFile = File(...)
):
    """
    Принимает два CSV файла (акселерометр и гироскоп),
    обрабатывает их и возвращает результат анализа.
    """
    try:
        if not model_manager.is_ready():
            raise HTTPException(
                status_code=500,
                detail="Модели не загружены. Проверьте пути к файлам в models.py"
            )
        
        # Проверяем имена файлов
        if not acc_file.filename or not gyro_file.filename:
            raise HTTPException(status_code=400, detail="Файлы не были переданы")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            acc_path = tmpdir_path / "AccelerometerUncalibrated.csv"
            gyro_path = tmpdir_path / "GyroscopeUncalibrated.csv"
            
            # Сохраняем файлы во временную папку
            with open(acc_path, "wb") as f:
                f.write(await acc_file.read())
            with open(gyro_path, "wb") as f:
                f.write(await gyro_file.read())
            
            # Читаем акселерометр
            df_acc = pd.read_csv(acc_path)
            df_acc.index = pd.to_datetime(df_acc['time'], unit='ns')
            df_acc.rename(columns={'z': 'acc_z', 'y': 'acc_y', 'x': 'acc_x'}, inplace=True)
            df_acc['time_sec'] = (df_acc.index - df_acc.index[0]).total_seconds()
            
            # Читаем гироскоп
            df_gyro = pd.read_csv(gyro_path)
            df_gyro.index = pd.to_datetime(df_gyro['time'], unit='ns')
            df_gyro.rename(columns={'z': 'gyr_z', 'y': 'gyr_y', 'x': 'gyr_x'}, inplace=True)
            
            # Объединяем данные по времени
            df = pd.merge_asof(
                df_acc[['time_sec', 'acc_x', 'acc_y', 'acc_z']],
                df_gyro[['gyr_x', 'gyr_y', 'gyr_z']],
                left_index=True,
                right_index=True,
                direction='nearest'
            )
            
            # Запускаем обработку
            result = process_signal(df)
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[process_signal] Ошибка: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ПЕРЕСЧЁТ ПО НОВЫМ ТОЧКАМ ====================
class PointsUpdate(BaseModel):
    """Модель данных для пересчёта по новым точкам"""
    points: Dict[str, Optional[float]]


@app.post("/api/recalculate")
async def recalculate_with_new_points(points_data: PointsUpdate):
    """
    Пересчитывает признаки и классификацию по новым точкам T1-T5,
    без повторного запуска нейросетей.
    """
    try:
        if not model_manager.is_ready():
            raise HTTPException(status_code=400, detail="Модели не загружены")
        
        points = points_data.points
        
        # Проверяем, что все 5 точек заданы
        required = ['T1', 'T2', 'T3', 'T4', 'T5']
        for key in required:
            if points.get(key) is None:
                raise HTTPException(status_code=400, detail=f"Точка {key} не задана")
        
        # Запускаем пересчёт
        result = recalculate_with_points(points)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[recalculate] Ошибка: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ЗАПУСК СЕРВЕРА ====================
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("Запуск сервера PD Signal Analysis")
    print("Откройте в браузере: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)