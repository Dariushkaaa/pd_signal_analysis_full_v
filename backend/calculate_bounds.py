import pandas as pd
import numpy as np
import json
from pathlib import Path

FEATURES_FILE = Path(r"E:\BMSTU\Grant\new_code_for_PD\results\features\features_by_session.xlsx")
OUTPUT_FILE = Path(r"E:\weights\best_weights\feature_bounds.json")

def calculate_feature_bounds():
    df = pd.read_excel(FEATURES_FILE)
    df_healthy = df[df['patient_id'].str.contains('H', na=False)].copy()
    
    features = ['cadence', 'sample_entropy', 'freeze_index', 'step_time_cv', 
                'X_std', 'X_rms', 'X_jerk_std']
    
    bounds = {}
    for feature in features:
        if feature not in df_healthy.columns:
            continue
        
        values = df_healthy[feature].dropna()
        if len(values) == 0:
            continue
        
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        bounds[feature] = {
            "lower_bound": round(max(0, lower_bound), 4),
            "upper_bound": round(upper_bound, 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "median": round(float(values.median()), 4),
            "mean": round(float(values.mean()), 4)
        }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(bounds, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Границы нормы сохранены в {OUTPUT_FILE}")

if __name__ == "__main__":
    calculate_feature_bounds()