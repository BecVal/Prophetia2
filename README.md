# Prophetia2

**Equipo de Desarrollo:**
* **César Becerra Valencia** y **José Luis Cortes Nava**: Cientificos de Computacion por la UNAM. Encargados de la arquitectura de software, la ingenieria de datos (Feature Engineering) y el entrenamiento del modelo predictivo (Machine Learning).
* **Dylan Eduardo Becerra Valencia**: Apostador profesional con solidos conocimientos en probabilidad. Encargado de auditar y validar los resultados matematicos del modelo, ofreciendo retroalimentacion experta para ajustar las probabilidades a escenarios reales de apuestas deportivas.

Prophetia2 es una arquitectura avanzada de Machine Learning Cuantitativo (Quant) diseñada para predecir los resultados de partidos de fútbol y encontrar valor financiero (Value Bets) en el mercado de apuestas.

El núcleo predictivo del proyecto ha evolucionado hacia un **Metamodelo (Corrector de Residuos)**. En lugar de predecir el resultado desde cero, el sistema ingiere las Cuotas de Apertura (Opening Odds) de casas asiáticas eficientes (Pinnacle/Bet365), las convierte a probabilidades puras, y utiliza un modelo **XGBoost Classifier** para buscar errores en la estimación de la casa. 

Para lograr esto, el motor procesa más de 12 años de histórico en 15 ligas europeas y cruza las cuotas con métricas tácticas avanzadas (xG), un sistema dinámico de Ratings Ofensivos/Defensivos (estilo Glicko), el Valor de Mercado de las plantillas (Transfermarkt) y un modelo bivariado Poisson Dixon-Coles para corregir la subestimación estadística de los empates. Finalmente, emite distribuciones probabilísticas calibradas mediante Regresión Isotónica libre de fuga de datos (Anti-Leakage), optimizando la rentabilidad financiera (Yield/ROI).

## Instalacion

Se recomienda utilizar un entorno virtual de Python. Para instalar las dependencias necesarias para la extraccion de datos, entrenamiento del modelo y visualizacion:

1. Clonar el repositorio.
2. Crear y activar el entorno virtual (opcional pero recomendado).
3. Ejecutar el archivo dependencias.bat (Solo la primera vez).
4. Ejecutar los archivos iniciar_backend.bat y iniciar_frontend.bat

## Arquitectura Matemática y Financiera

1. **Arquitectura Anti-Leakage y Single-Row:** El modelo filtra duplicados (Double-Row Betting) y calibra las probabilidades puramente fuera de muestra (OOS) garantizando métricas financieras honestas y robustas.
2. **Valoración de Plantillas (Proxy de Jugadores):** Mediante web scraping automatizado, Prophetia2 inyecta el valor de mercado contemporáneo (en millones de euros) de cada equipo desde Transfermarkt para sintetizar la calidad técnica de los jugadores sin sufrir la maldición de la dimensionalidad.
3. **Ratings Dinámicos (Ataque/Defensa) y ELO:** El sistema calcula ratings separados de ataque y defensa para cada equipo basándose en la métrica de Goles Esperados (xG), actualizándolos partido a partido sin contaminar el futuro.
2. **Modelo de Poisson Bivariado (Dixon-Coles):** Dado que el fútbol es un deporte de baja puntuación, los empates ocurren con mayor frecuencia de lo que sugiere la independencia estadística. Prophetia2 usa el ajuste de Dixon-Coles ($\rho \approx -0.15$) para inflar matemáticamente la probabilidad conjunta de resultados 0-0 y 1-1.
3. **Calibración Isotónica:** Las probabilidades crudas del XGBoost se ajustan de manera no-paramétrica para asegurar que una predicción del "60%" realmente se cumpla el 60% de las veces en la realidad.
4. **Simulación Financiera y Kelly Criterion:** Prophetia2 no solo clasifica; es un simulador de inversiones. El entrenamiento concluye con un backtest financiero riguroso. El algoritmo compara sus predicciones contra las **Cuotas de Cierre (Closing Odds)** (la línea más eficiente del mercado). Si encuentra Expectativa Matemática positiva (EV > 5%), el modelo aplica el **Criterio de Kelly** fraccionado para calcular exactamente qué porcentaje del Bankroll apostar (max 5%). Al final, reporta el ROI, Turnover y el Yield neto.

## Flujo de Ejecución

Para obtener predicciones y ejecutar las simulaciones financieras, sigue el pipeline estándar:

### 0. Descarga de Datos de Partidos (Ingestion)
Para alimentar el modelo, primero debes descargar los históricos de partidos y eventos tácticos (15 ligas principales y secundarias a lo largo de 12 años). Ejecuta el script:
```bash
python core/download_football_data.py
```

### 1. Descarga del Proxy de Jugadores (Transfermarkt)
Extrae el valor de mercado histórico de cada equipo para medir el nivel técnico de sus jugadores. Cuenta con guardado progresivo antibloqueos.
```bash
python ingestion/transfermarkt_scraper.py
```

### 2. Adaptacion de Datos (Data Adapter)
Para permitir que Prophetia2 consuma las diferentes fuentes de datos, primero debes estandarizar los eventos crudos en un DataFrame Intermedio Universal. Ejecuta el adaptador:
```bash
python core/data_adapter.py
```
Este script generara un dataset intermedio tabular unificado en `data/interim/intermediate_dataset.parquet`.

### 3. Extracción de Cuotas (Bookmaker Odds)
Descarga el historial de cuotas de casas de apuestas globales para entrenar el metamodelo y simular rentabilidad:
```bash
python ingestion/fetch_odds.py
```
Este script descarga cuotas de apertura (PSH) y cierre (PSCH) para las 15 ligas, convirtiéndolas a probabilidades implícitas (Vig-free).

### 4. Procesamiento Matemático (Feature Engineering)
El motor de Prophetia2 cruza las estadísticas tácticas avanzadas (Rolling Averages), el ELO Clásico, los ratings Glicko y fusiona los millones de euros de Transfermarkt con cada evento:
```bash
python core/feature_engineering.py
```
Se generará el dataset final de entrenamiento listo para la IA en `data/processed/matches_dataset.parquet`.

### 5. Entrenamiento del Metamodelo de IA
Una vez procesados los más de 65,000 partidos históricos, entrena el modelo optimizado:
```bash
python core/train.py
```
Este proceso dividirá los datos cronológicamente, buscará hiperparámetros óptimos para XGBoost con `Optuna`, aplicará la Calibración Isotónica y ejecutará una simulación financiera completa de Bankroll ($1,000 iniciales).

**¿Qué datos entran al modelo? (Inputs)**
- **Consenso del Mercado (Metamodelado):** Probabilidades de apertura del mercado (`open_prob_win`, `open_prob_draw`, `open_prob_loss`).
- **Calidad de Jugadores (Proxy):** Valor de mercado de la plantilla contemporánea extraído de Transfermarkt (`team_squad_value`, `squad_value_diff`).
- **Ratings Cuantitativos:** Fuerza Relativa de Ataque, ELO Clásico (`team_elo`, `elo_diff`), y Ratings Puros de Ataque/Defensa (`team_att_rating`, `team_def_rating`).
- **Tácticas Base (EMA-3/EMA-5):** Goles esperados creados y concedidos (`xg_created`, `xg_conceded`), xG-Chain, posesión, y acciones defensivas (presiones, intercepciones).
- **Contexto Avanzado:** Días de descanso, desgaste por competiciones europeas previas (`is_european_hangover`), e historial directo (H2H).

**¿Qué datos expulsa el modelo? (Outputs)**
En lugar de dar un resultado seco, el modelo expulsa **probabilidades calibradas de Valor** para las tres clases posibles:
- Probabilidad de Victoria Local ajustada (ej. 60.5%)
- Probabilidad de Empate ajustada por Dixon-Coles (ej. 24.5%)
- Expectativa Matemática de la Apuesta (EV) frente a la línea de cierre.

### 6. Visualizacion y Prediccion Cientifica (Jupyter Notebooks)
El analisis interactivo y las predicciones individuales se gestionan a traves de Jupyter Notebooks. Inicie su servidor de Jupyter:
```bash
jupyter notebook
```
Navegue a la carpeta `notebooks/` y utilice los siguientes archivos:

*   **01_data_exploration.ipynb**: Permite explorar la estructura cruda de los datos de StatsBomb.
*   **02_model_selection.ipynb**: Genera visualizaciones graficas sobre la correlacion tactica y revela que estadisticas (Feature Importance) considera XGBoost mas determinantes para ganar.
*   **03_live_dashboard.ipynb**: Contiene un panel interactivo (Dashboard). Ejecute todas las celdas para habilitar un selector desplegable de partidos. Al seleccionar un encuentro, el sistema utilizara el modelo entrenado para emitir las probabilidades exactas de Victoria, Empate o Derrota, comparando ademas la prediccion con el flujo tactico real del partido.

### 7. Predicción Financiera en Terminal (CLI Quant)
Para ejecutar pronósticos en tiempo real basados en los últimos partidos procesados y calcular el riesgo (Bankroll Staking), utiliza nuestro menú interactivo de predicción cuantitativa:
```bash
cd core
python cli_predictor.py
```
El script te permitirá seleccionar primero la competición (Liga) y luego autocompletará los equipos participantes. Podrás introducir las cuotas del mercado actual y aplicar penalizaciones de fuerza estocástica si hay jugadores clave lesionados.

**Ejemplo de Salida:**
```text
+------------------------------------------------------+
| Prophetia2 - Quant Value Betting CLI                 |
| Initializing stochastic models and feature stores... |
+------------------------------------------------------+

=== ANÁLISIS CUANTITATIVO DEL PARTIDO ===
+-----------------------------------------------------------------------------+
|             |            |             |            |          EV |         |
|             | Odds       |       Prob. |      Prob. |   (Expected |         |
| Mercado     | (Bookie)   |      Bookie |     Modelo |      Value) | Kelly % |
|-------------+------------+-------------+------------+-------------+---------|
| 1 (Local -  | 2.10       |       47.6% |      34.7% |      -27.2% |   0.00% |
| Arsenal)    |            |             |            |             |         |
| X (Empate)  | 3.40       |       29.4% |      40.2% |      +36.7% |  15.30% |
| 2 (Visita - | 3.50       |       28.6% |      25.1% |      -12.1% |   0.00% |
| Chelsea)    |            |             |            |             |         |
+-----------------------------------------------------------------------------+

=== RECOMENDACIÓN DE STAKING ===
+---------------------------- SISTEMA DE STAKING -----------------------------+
| VALUE DETECTADO                                                             |
| Selección: Empate (X) @ 3.40                                                |
| Stake Recomendado: $30.61 (3.06% del bankroll)                              |
+-----------------------------------------------------------------------------+
```
