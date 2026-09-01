"""
Научные границы нормы признаков походки для здоровых людей.
Источники:
- Hausdorff JM et al. (2001) "Gait dynamics in Parkinson's disease"
- Plotnik M et al. (2007) "Freezing of gait in PD"
- Rochester L et al. (2014) "Gait and gait-related activities of daily living in PD"
- Del Din S et al. (2016) "Validation of an accelerometer to quantify a comprehensive battery of gait characteristics"
"""

FEATURE_NORMS = {
    'cadence': {
        'lower_bound': 95.0,    # шагов/мин (ниже - брадикинезия)
        'upper_bound': 125.0,   # шагов/мин
        'description': 'Каденс (шагов в минуту)'
    },
    'sample_entropy': {
        'lower_bound': 0.8,     # ниже - стереотипность походки
        'upper_bound': 2.2,     # выше - избыточная вариативность
        'description': 'Энтропия выборки (регулярность)'
    },
    'freeze_index_mean': {
        'lower_bound': 0.0,
        'upper_bound': 0.15,    # выше 0.15 - признак фризкинга
        'description': 'Индекс застывания'
    },
    'step_time_cv': {
        'lower_bound': 0.0,
        'upper_bound': 0.06,    # выше 6% - нестабильность
        'description': 'Коэффициент вариации времени шага'
    },
    'X_std': {
        'lower_bound': 0.15,    # нормализованный сигнал [0,1]
        'upper_bound': 0.35,
        'description': 'Стандартное отклонение (ось X, норм.)'
    },
    'X_rms': {
        'lower_bound': 0.45,    # нормализованный сигнал [0,1]
        'upper_bound': 0.75,
        'description': 'Среднеквадратичное значение (ось X, норм.)'
    },
    'X_jerk_std': {
        'lower_bound': 1.0,     # нормализованный рывок
        'upper_bound': 5.0,     # выше - резкие движения
        'description': 'Стд. откл. рывка (ось X, норм.)'
    }
}