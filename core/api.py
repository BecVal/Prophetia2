from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import subprocess
import os

app = FastAPI(title="Prophetia2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ojo con las rutas, asumiendo que corres esto desde la carpeta /core
DATA_PATH = '../data/processed/matches_dataset.parquet'
MODEL_PATH = 'save_models/prophetia_xgb_model.pkl'

def run_script(script_name: str, folder: str):
    """Ejecuta los scripts en su carpeta correcta para no romper las rutas"""
    try:
        subprocess.run(["python", script_name], cwd=folder, check=True)
        return {"status": "success", "message": f"{script_name} ejecutado correctamente"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error ejecutando {script_name}: {str(e)}")

@app.post("/api/run/ingestion")
def run_ingestion():
    return run_script("statsbomb_ingestion.py", "../ingestion")

@app.post("/api/run/adapter")
def run_adapter():
    return run_script("data_adapter.py", ".")

@app.post("/api/run/features")
def run_features():
    return run_script("feature_engineering.py", ".")

@app.post("/api/run/train")
def run_train():
    return run_script("train.py", ".")

@app.get("/api/partidos")
def obtener_partidos():
    """Devuelve el catálogo de partidos desmenuzado para armar el H2H en React"""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Datos no encontrados")
    
    df = pd.read_parquet(DATA_PATH)
    match_list = df[df['is_home'] == 1].copy().sort_values('match_date', ascending=False)
    
    resultados = []
    for _, row in match_list.iterrows():
        competicion = str(row.get('competition_name', f"Temporada {str(row['match_date'])[:4]}"))
        resultados.append({
            "id": row['match_id'],
            "liga": competicion,
            "local": str(row['team']),
            "visita": str(row['opponent']),
            "fecha": str(row['match_date'])[:10]
        })
    return resultados

@app.get("/api/model-stats")
def obtener_estadisticas_modelo():
    """Extrae las métricas globales y los pesos del modelo"""
    try:
        df = pd.read_parquet(DATA_PATH)
        data = joblib.load(MODEL_PATH)
        modelo = data['model']
        feature_cols = data['features']
        
        total_variables = len(feature_cols)
        fuentes_datos = df['source'].value_counts().to_dict() if 'source' in df.columns else {"statsbomb": len(df)}
        
        distribucion = df['outcome'].replace({-1: 0, 0: 1, 1: 2}).value_counts().to_dict()
        
        calibrated_classifiers = modelo.calibrated_classifiers_
        voting_clf = calibrated_classifiers[0].estimator
        pipeline_xgb = voting_clf.estimators_[0]
        xgb_model = pipeline_xgb.named_steps['clf']
        selector = pipeline_xgb.named_steps['feature_selection']
        
        selected_mask = selector.get_support()
        selected_features = np.array(feature_cols)[selected_mask]
        importances = xgb_model.feature_importances_
        
        indices = np.argsort(importances)[-10:]
        top_features = [{"nombre": selected_features[i].replace('_', ' ').title(), "peso": float(importances[i])} for i in reversed(indices)]
        
        return {
            "accuracy_global": "68.4%", # Placeholder para César
            "log_loss": "0.892",       # Placeholder para César
            "total_variables": total_variables,
            "fuentes": fuentes_datos,
            "clases": distribucion,
            "feature_importance": top_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo stats: {str(e)}")

@app.get("/api/prediccion/{match_id}")
def obtener_prediccion(match_id: str):
    """Calcula probabilidades y devuelve métricas e insights"""
    try:
        df = pd.read_parquet(DATA_PATH)
        data = joblib.load(MODEL_PATH)
        modelo, feature_cols = data['model'], data['features']
        
        partido = df[df['match_id'] == match_id]
        if partido.empty:
            raise HTTPException(status_code=404, detail="Partido no encontrado")

        home_row = partido[partido['is_home'] == 1].iloc[0]
        away_row = partido[partido['is_home'] == 0].iloc[0]

        X_predict = home_row[feature_cols].to_frame().T
        probs = modelo.predict_proba(X_predict)[0]
        clases = modelo.classes_

        return {
            "id": match_id,
            "equipos": {"local": home_row['team'], "visitante": away_row['team']},
            "marcador_real": {"local": int(home_row.get('goals_scored', 0)), "visitante": int(away_row.get('goals_scored', 0))},
            "partido_info": {"liga": str(home_row.get('competition_name', 'Torneo Oficial'))},
            "probabilidades": {
                "local": round(probs[np.where(clases == 2)[0][0]] * 100, 1),
                "empate": round(probs[np.where(clases == 1)[0][0]] * 100, 1),
                "visitante": round(probs[np.where(clases == 0)[0][0]] * 100, 1)
            },
            "metricas": {
                "xG": {"local": round(float(home_row.get('xg_created', 0)), 2), "visitante": round(float(away_row.get('xg_created', 0)), 2)},
                "tiros": {"local": int(home_row.get('shots_on_target', 0)), "visitante": int(away_row.get('shots_on_target', 0))},
                "posesion": {"local": round(float(home_row.get('possession_pct', 50)), 1), "visitante": round(float(away_row.get('possession_pct', 50)), 1)},
                "pases_precisos": {"local": round(float(home_row.get('pass_accuracy', 0)), 1), "visitante": round(float(away_row.get('pass_accuracy', 0)), 1)},
                "faltas": {"local": int(home_row.get('fouls_committed', 0)), "visitante": int(away_row.get('fouls_committed', 0))},
                "tarjetas_amarillas": {"local": int(home_row.get('yellow_cards', 0)), "visitante": int(away_row.get('yellow_cards', 0))}
            },
            "insights": {
                "elo_local": round(float(home_row.get('team_elo', 1500))),
                "elo_visita": round(float(home_row.get('opp_elo', 1500))),
                "descanso_local": int(home_row.get('rest_days', 7)),
                "poder_ataque_relativo": round(float(home_row.get('relative_attack_strength', 1.0)), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))