# Prophetia2

**Equipo de Desarrollo:**
* **César Becerra Valencia** y **José Luis Cortes Nava**: Cientificos de Computacion por la UNAM. Encargados de la arquitectura de software, la ingenieria de datos (Feature Engineering) y el entrenamiento del modelo predictivo (Machine Learning).
* **Dylan Eduardo Becerra Valencia**: Apostador profesional con solidos conocimientos en probabilidad. Encargado de auditar y validar los resultados matematicos del modelo, ofreciendo retroalimentacion experta para ajustar las probabilidades a escenarios reales de apuestas deportivas.

Prophetia2 es una arquitectura avanzada de Machine Learning diseñada para predecir los resultados de partidos de futbol utilizando datos tacticos detallados provenientes de StatsBomb. 

El nucleo predictivo del proyecto se basa en un modelo XGBoost Classifier, el cual evalua metricas tacticas complejas como Goles Esperados (xG), presiones, intercepciones y efectividad de pases. Para garantizar el rigor matematico y prevenir la fuga de datos (Data Leakage), el sistema emplea metodos de ingenieria de series temporales, calculando los promedios historicos (Rolling Averages) del estado de forma de los equipos antes del inicio de cada partido. En lugar de predecir resultados binarios, el modelo emite distribuciones probabilisticas (Victoria, Empate, Derrota) optimizadas mediante la metrica Log-Loss.

## Instalacion

Se recomienda utilizar un entorno virtual de Python. Para instalar las dependencias necesarias para la extraccion de datos, entrenamiento del modelo y visualizacion:

1. Clonar el repositorio.
2. Crear y activar el entorno virtual (opcional pero recomendado).
3. Ejecutar el archivo dependencias.bat (Solo la primera vez).
4. Ejecutar los archivos iniciar_backend.bat y iniciar_frontend.bat

## Flujo de Ejecucion

Para obtener predicciones y visualizar los resultados, se debe seguir un pipeline estandar de datos:

### 0. Descarga de Datos (Ingestion)
Para alimentar el modelo, primero debes descargar la base de datos abierta de StatsBomb (partidos y eventos tacticos). Ejecuta el script de ingestion:
```bash
python ingestion/statsbomb_ingestion.py
```
*Nota sobre el almacenamiento:* Los datos se guardan en formato `.parquet`, que es altamente comprimido. Descargar una buena cantidad de partidos (cientos o miles) ocupara aproximadamente entre **500 MB y 1.5 GB** de espacio en tu disco duro, dependiendo de las competiciones habilitadas en el script.

### 1. Adaptacion de Datos (Data Adapter)
Para permitir que Prophetia2 consuma diferentes fuentes de datos (StatsBomb, Understat, football-data, etc.), primero debes estandarizar los eventos crudos en un DataFrame Intermedio Universal. Ejecuta el adaptador:
```bash
python core/data_adapter.py
```
Este script leera los archivos especificos de tu proveedor de datos (por defecto StatsBomb) y generara un dataset intermedio tabular unificado en `data/interim/intermediate_dataset.parquet`.

### 2. Procesamiento Matematico (Feature Engineering)
Una vez estandarizados los datos base, el motor de Prophetia2 debe calcular las estadisticas tacticas avanzadas en series temporales (Rolling Averages) y el sistema de rating de calidad de los equipos (ELO) sin fuga de datos. Ejecute el siguiente script:
```bash
python core/feature_engineering.py
```
Este script consumira el dataset intermedio universal y generara el dataset final de entrenamiento listo para la IA en la carpeta `data/processed/matches_dataset.parquet`.

### 3. Entrenamiento del Modelo de IA
Una vez procesados los datos, proceda a entrenar el modelo XGBoost:
```bash
python core/train.py
```
Este proceso dividira los datos de forma cronologica, aplicara seleccion automatica de caracteristicas sin fugas de datos (Scikit-Learn Pipelines), utilizara Optuna para buscar hiperparametros que minimicen el Log-Loss y entrenara un ensamble de algoritmos (VotingClassifier combinando XGBoost y Regresion Logistica). Al finalizar, guardara el modelo matematico compilado (`prophetia_xgb_model.pkl`) en el directorio `core/save_models/`.

**¿Qué datos entran al modelo? (Inputs)**
El modelo consume un array de estadísticas tácticas en formato de *media móvil* (promedio de los últimos 3 partidos) para evitar fugas de datos. Entre las métricas principales ingresan:
- **Tácticas Base:** Goles esperados (`xg_created`, `xg_conceded`), tiros a puerta, córners, posesión y precisión de pases.
- **Acciones Defensivas:** Presiones, intercepciones, faltas cometidas y recuperaciones.
- **Contexto Avanzado:** Días de descanso (`rest_days`) y la **Fuerza Relativa de Ataque** (una métrica que cruza la capacidad ofensiva propia contra la solidez defensiva reciente del oponente).

*Nota: Internamente, el sistema usa `SelectFromModel` para filtrar automáticamente el ruido y quedarse solo con las métricas más predictivas.*

**¿Qué datos expulsa el modelo? (Outputs)**
En lugar de dar un resultado seco (ej. "Gana el Local"), el modelo expulsa **probabilidades calibradas** (Soft Probabilities) para las tres clases posibles:
- Probabilidad de Victoria Local (ej. 60.5%)
- Probabilidad de Empate (ej. 24.5%)
- Probabilidad de Derrota / Victoria Visitante (ej. 15.0%)

### 3. Visualizacion y Prediccion Cientifica (Jupyter Notebooks)
El analisis interactivo y las predicciones individuales se gestionan a traves de Jupyter Notebooks. Inicie su servidor de Jupyter:
```bash
jupyter notebook
```
Navegue a la carpeta `notebooks/` y utilice los siguientes archivos:

*   **01_data_exploration.ipynb**: Permite explorar la estructura cruda de los datos de StatsBomb.
*   **02_model_selection.ipynb**: Genera visualizaciones graficas sobre la correlacion tactica y revela que estadisticas (Feature Importance) considera XGBoost mas determinantes para ganar.
*   **03_live_dashboard.ipynb**: Contiene un panel interactivo (Dashboard). Ejecute todas las celdas para habilitar un selector desplegable de partidos. Al seleccionar un encuentro, el sistema utilizara el modelo entrenado para emitir las probabilidades exactas de Victoria, Empate o Derrota, comparando ademas la prediccion con el flujo tactico real del partido.
