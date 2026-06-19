import os
import pandas as pd
import glob

RAW_DATA_DIR = '../data/raw/statsbomb/events/'

def explore_data():
    # Obtener todos los archivos parquet
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.parquet"))
    
    if not files:
        print("No se encontraron archivos .parquet en la carpeta de eventos.")
        return
        
    print(f"Total de partidos descargados: {len(files)}")
    
    # Tomar el primer archivo para explorar
    sample_file = files[0]
    print(f"\nCargando archivo de muestra: {os.path.basename(sample_file)}")
    
    try:
        df = pd.read_parquet(sample_file, engine='fastparquet')
    except Exception as e:
        print(f"Error al cargar el archivo parquet: {e}")
        return
        
    print(f"Dimensiones del dataset del partido (filas, columnas): {df.shape}")
    
    # Mostrar las columnas más relevantes
    print("\nColumnas disponibles (primeras 30):")
    print(list(df.columns)[:30])
    
    # Contar tipos de eventos
    print("\nTipos de eventos en el partido (Top 10):")
    if 'type' in df.columns:
        print(df['type'].value_counts().head(10))
    else:
        print("Columna 'type' no encontrada.")
        
    # Buscar métricas avanzadas (xG, coordenadas)
    print("\nBuscando métricas avanzadas:")
    
    # Coordenadas (location)
    location_cols = [col for col in df.columns if 'location' in col]
    print(f"- Columnas de localización: {location_cols}")
    
    # Goles esperados (xG)
    xg_cols = [col for col in df.columns if 'xg' in col.lower()]
    print(f"- Columnas de goles esperados (xG): {xg_cols}")
    
    if xg_cols:
        for col in xg_cols:
            non_null = df[col].dropna()
            if not non_null.empty:
                print(f"  * Muestra de {col} (primeros 5 valores no nulos):")
                print(non_null.head(5))

    # Posesión
    possession_cols = [col for col in df.columns if 'possession' in col.lower()]
    print(f"- Columnas de posesión: {possession_cols}")

if __name__ == "__main__":
    # Ajustar directorio de trabajo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    explore_data()
