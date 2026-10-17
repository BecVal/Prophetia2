# Prophetia2: Motor Cuantitativo de Trading Deportivo (Quantitative Sports Trading Engine)

**Prophetia2** es un motor algorítmico y matemático de *Machine Learning* Cuantitativo (*Quant*) diseñado para modelar la probabilidad de eventos deportivos, detectar ineficiencias en mercados de cuotas (Bookmakers, Polymarket) y ejecutar estrategias de *Value Betting* con gestión de riesgo institucional. 

A diferencia de los modelos predictivos estándar que buscan maximizar el *Accuracy*, Prophetia2 es fundamentalmente una arquitectura financiera que busca **maximizar el Valor Esperado (EV)** superando el *Vig* (margen) del mercado a través del cálculo riguroso del Closing Line Value (CLV).

---

## Arquitectura Matemática y Modelado Estocástico

El núcleo predictivo abandona los enfoques monolíticos en favor de una **Arquitectura de Stacking Jerárquico en Dos Etapas (Sub-Stacker Fundamental + Meta-Stacker Final)** sobre **9 modelos especializados**, combinando procesos estocásticos de conteo, filtros dinámicos de estado, aprendizaje profundo tabular y modelos de microestructura de mercado:

### 1. Modelo Cuantitativo Bivariante de Poisson + Dixon-Coles con Filtro de Kalman (`train_quant_advanced.py`)
El fútbol es un proceso de conteo estocástico acoplado donde los marcadores de baja anotación ($0\text{-}0, 1\text{-}0, 0\text{-}1, 1\text{-}1$) presentan correlación estructural. En lugar de una regresión de Poisson independiente, Prophetia2 implementa en **PyTorch** (`KalmanDixonColesQuantModel`) la optimización directa de la **Verosimilitud Conjunta Bivariante de Poisson con Corrección de Dixon-Coles ($\tau_{\rho}$)**:
$$P(X = x, Y = y) = \tau_{\lambda_H, \lambda_A, \rho}(x, y) \, \frac{e^{-\lambda_H}\lambda_H^x}{x!} \, \frac{e^{-\lambda_A}\lambda_A^y}{y!}$$
Las tasas latentes de gol ($\ln \lambda_H, \ln \lambda_A$) integran:
- **Ratings Dinámicos con Filtro de Kalman:** Antes de la optimización tensorial global (`L-BFGS` con regularización $L_2$), un filtro secuencial estilo Kalman actualiza partido a partido las fuerzas latentes de ataque ($\alpha_{i,t}$) y defensa ($\beta_{i,t}$) con varianza de innovación temporal ($Q$) proporcional a los días transcurridos entre encuentros.
- **Covariables Estructurales y Parámetro $\rho$ Aprendido:** Incorpora ventaja de localía ($\gamma_{\text{home}}$), diferencial de Elo ($\beta_{\text{elo}} \Delta \text{Elo}$) y diferencial logarítmico de valor de plantilla ($\beta_{\text{val}} \Delta \ln \text{SquadValue}$), aprendiendo simultáneamente el parámetro de correlación $\rho \in (-0.25, 0.25)$ mediante transformación $\tanh$.

### 2. Time-Decay Likelihood Ponderado ($T_{1/2} = 365$ días)
La capacidad competitiva de un club decae exponencialmente en el tiempo. El núcleo tensorial aplica un decaimiento temporal exponencial continuo $w_i = \exp(-\xi \cdot \Delta t_i)$ con una vida media (*Half-life*) de **365 días** ($\xi = \ln(2)/365$), ponderando directamente la log-verosimilitud negativa conjunta:
$$\mathcal{L}_{\text{decay}}(\theta) = -\frac{\sum_{i} w_i \ln P(x_i, y_i \mid \theta)}{\sum_{i} w_i} + \lambda_{L_2} \left(\|\alpha\|_2^2 + \|\beta\|_2^2\right)$$

### 3. Microestructura de Mercado: Desmarginalización de Shin (1993) y Dinámica GBM (`train_market.py` & `train_gbm_model.py`)
Para capturar la información implícita en las cuotas sin incurrir en *Look-Ahead Bias*, el sistema separa estrictamente las señales pre-partido de la dinámica histórica:
- **Probabilidades Verdaderas de Apertura (Shin 1993 & Power Method):** Sobre las cuotas **exclusivamente de apertura** ($O_{\text{open}}$), se resuelve iterativamente por bisección la proporción de *insider trading* $z \in (0, 0.3)$ del modelo de **Shin (1993)** y el exponente $k$ del **Método de Potencia (Štrumbelj 2014)** para eliminar el *overround* asimétrico de las casas de apuestas.
- **Movimiento Browniano Geométrico (GBM) Causal ($t-1$):** El *drift* ($\mu$), la volatilidad histórica ($\sigma$), las medias exponenciales (EWMA) y los indicadores de *Steam* se calculan sobre la trayectoria logarítmica de cuotas ($\ln(O_{\text{open}} / O_{\text{close}})$) y se desplazan estrictamente al pasado (`.shift(1)`) por equipo antes de entrar a los modelos **LightGBM + XGBoost + CatBoost**, evitando cualquier filtración de las cuotas de cierre del partido objetivo.

### 4. Aprendizaje Profundo Tabular (`PyTorchGatedResNet10`) y Ensambles Tácticos (`train_nn.py`, `train_context.py`, `train_draws.py`)
- **Red Neuronal Profunda (`PyTorchGatedResNet10`):** Arquitectura **Gated Residual Network (GRN)** en PyTorch con compuertas lineales (`GatedContext`, proyecciones sigmoides de atención contextual), bloques residuales con `BatchNorm1d`, activaciones `SiLU`, regularización `Dropout`, pérdida de entropía cruzada con suavizado de etiquetas ($\text{label\_smoothing}=0.03$), planificador `CosineAnnealingLR` y **Calibración por Temperatura ($T$)** ajustada estrictamente Out-Of-Sample.
- **Clasificador de Contexto Táctico y Fatiga (`train_context.py`):** Modelo **XGBoost** enriquecido con *priors* de dominio explícitos (`elo_prob_win`, `log_squad_ratio`, `recovery_index_team`, `elo_sharpe_ratio`, `elo_momentum_interaction`) y calibración probabilística temporal mediante regresión sigmoide (**Platt Scaling** con `TimeSeriesSplit`).
- **Ensamble Especializado en Empates (`HybridDrawsEnsemble` en `train_draws.py`):** Ensamble binario híbrido (**XGBoost + LightGBM** con `CalibratedClassifierCV` temporal) que integra la probabilidad analítica de empate de Poisson ($\sum_{k=0}^{8} P_H(k)P_A(k)$), la simetría táctica entre rivales y la tasa histórica suavizada de empates por competición (calculada exclusivamente sobre `train_df`).

### 5. Modelado Estocástico de Mercados Derivados (Tiros al Arco, Córneres y Tarjetas)
Prophetia2 modela los mercados secundarios mediante arquitecturas de **doble perspectiva (`Home` + `Away`)** sobre variables de conteo ($\mathbb{N}_0$), garantizando simetría táctica y evitando sesgos de perspectiva única:
- **Tiros al Arco (`train_shots_on_goal.py`):** Ensamble dual (`Home` y `Away`) de 3 familias algorítmicas: **XGBoost (`reg:tweedie`) + LightGBM (`regression_l1`) + Regresión Ridge ($L_2$)** con generación de predicciones Out-Of-Fold vía `TimeSeriesSplit`. Las expectativas individuales ($\lambda_{\text{SoT}, H}, \lambda_{\text{SoT}, A}$) alimentan una convolución **Bivariante de Poisson ($\rho = 0.08$)** y **Binomial Negativa** para estimar líneas `Over 7.5 / 8.5 / 9.5` y hándicaps de remates a puerta.
- **Tiros de Esquina (`train_corners_model.py`):** Regresores duales **LightGBM Poisson** (`Home` y `Away`) combinados con una parametrización **Binomial Negativa** de sobredispersión aprendida empíricamente ($\text{Var}(X) = \mu + c\mu^2$) y **Calibración Isotónica** ajustada en una partición temporal de calibración estrictamente disjunta del conjunto de prueba (`85%` sub-train / `15%` cal).
- **Tarjetas Totales (`train_cards_model.py`):** Regresores duales **XGBoost Tweedie** (`cards_home_xgboost_model.pkl` y `cards_away_xgboost_model.pkl`) condicionados por el perfil histórico del árbitro (`ref_Historical_Avg_Cards`, `ref_Card_Severity_Z_Score`) y la fricción competitiva. Las expectativas esperadas ($E[\text{Total}] = \hat{\lambda}_H + \hat{\lambda}_A$) se integran mediante una distribución **Bivariante de Poisson con corrección de Dixon-Coles** (`calc_cards_probabilities_vectorized`) para derivar analíticamente $P(\text{Cards} > 3.5, 4.5, 5.5)$.

### 6. Meta-Stacking Jerárquico y Rigor Causal Out-Of-Fold (`train_stacker.py`)
Para combinar los modelos base sin sufrir *In-Sample Overfitting* ni *Look-Ahead Bias*, `train_stacker.py` implementa un **Stacking Jerárquico en Dos Etapas** gobernado estrictamente por particiones temporales (`TimeSeriesSplit` de 5 folds + sub-splits temporales anidados):
1. **Nivel 1.5 — Sub-Stacker Fundamental (`stacker_fund_model.pkl`):** Un meta-clasificador lineal regularizado (`StandardScaler` + `LogisticRegression` $L_2, C=0.1$) combina exclusivamente los modelos fundamentales puros (`quant`, `context`, `nn`, `draws`). En el bloque inicial de *burn-in* de `TimeSeriesSplit` (donde no existe pasado para entrenar un fold previo), se asigna causalmente el consenso equiponderado de los modelos fundamentales sin consultar las etiquetas $y$.
2. **Nivel 2 — Meta-Stacker Final (`stacker_final_model.pkl`):** Combina las probabilidades Out-Of-Fold del Sub-Stacker Fundamental con los modelos de mercado (`market`, `gbm`), variables de incertidumbre inter-modelo (**desviación estándar por clase** `meta_std_*` y **entropía de Shannon del consenso** `meta_entropy`) y las probabilidades implícitas de apertura (`implied_open_*`) mediante una segunda etapa `StandardScaler` + `LogisticRegression(C=0.1)`.

### 7. Calibración Probabilística Temporal Estricta
Cada familia predictiva aplica el método de calibración acorde a su geometría estadística respetando siempre la frontera temporal `Train -> Calibration -> Test` (`2023-08-08`):
- **Temperature Scaling ($T$):** En `PyTorchGatedResNet10` (`train_nn.py`), optimizando la NLL sobre el split temporal de validación sin alterar el ranking relativo de los *logits*.
- **Platt Scaling / Sigmoid Calibration (`TimeSeriesSplit`):** En `train_context.py` y `train_draws.py`.
- **Calibración Isotónica Monótona Out-Of-Sample:** En `train_corners_model.py`, ajustada sobre el bloque temporal de validación previo al corte de prueba para mitigar la cola pesada en mercados de córneres.

---

## Gestión de Riesgo y Simulación de Bankroll (`simulate_bankroll.py` & `cli_predictor.py`)

### 1. Closing Line Value (CLV) Drift Filter (`train_clv_model.py`)
Dado que el mercado corrige ineficiencias entre la apertura y el cierre, Prophetia2 entrena un regresor **XGBoost** sobre el diferencial de probabilidades implícitas ($\text{CLV} = P_{\text{close}}^{\text{fair}} - P_{\text{open}}^{\text{fair}}$) utilizando únicamente variables observables en la apertura (`implied_open_*`, `overround_open`, divergencia fundamental vs. apertura e historial rezagado de *steam* de ambos equipos). Si el modelo proyecta que la línea se moverá en contra más allá del umbral de tolerancia ($\widehat{\text{CLV}} < -0.015$), la operación se descarta preventivamente.

### 2. Criterio de Kelly Fraccionario y Selección Ajustada por Riesgo
El dimensionamiento de cada posición escala en función de la ventaja matemática neta ($EV = p \cdot O_{\text{net}} - 1$, descontando fricción/comisión de ejecución) mediante el **Criterio de Kelly Fraccionario**:
$$f^* = \gamma_{\text{Kelly}} \cdot \frac{b p - q}{b}, \qquad b = O_{\text{net}} - 1, \quad q = 1 - p$$
Cuando múltiples mercados de un mismo partido superan el umbral mínimo de $EV$, el selector evalúa conjuntamente los mercados simples (`1`, `X`, `2`) y las **Dobles Oportunidades (`1X`, `X2`)** eligiendo el contrato que maximiza el **Edge de Kelly Ajustado por Riesgo**:
$$\text{Risk-Adjusted Edge} = \frac{\text{EV}}{\sqrt{O_{\text{net}} - 1}}$$

### 3. Optimización de Portafolio Multi-Activo de Markowitz (Modern Portfolio Theory)
Para asignar capital simultáneamente entre múltiples partidos y mercados de una jornada sin sobreexponer la banca, tanto `core/simulate_bankroll.py` como `core/cli_predictor.py` integran el optimizador cuadrático de **Harry Markowitz (`optimize_markowitz_portfolio`)**:
- **Matriz de Varianza-Covarianza Analítica ($\mathbf{\Sigma}$):** Para cada jornada, calcula la varianza exacta de retorno de cada apuesta ($\sigma_i^2 = p_i(1-p_i)O_{i,\text{net}}^2$) y la covarianza analítica exacta ($\Sigma_{ij} = (p_{ij} - p_i p_j) O_{i,\text{net}} O_{j,\text{net}}$) derivada de la matriz bivariante de goles de Poisson para apuestas de un mismo encuentro.
- **Maximización de Utilidad Media-Varianza:** Resuelve mediante programación cuadrática (`SLSQP`) con respaldo analítico:
  $$\max_{\mathbf{w}} \quad \mathbf{w}^T \boldsymbol{\mu} - \frac{\gamma}{2} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \quad \text{sujeto a} \quad \sum_{i} w_i \le \text{Cap}_{\text{diario}}, \quad 0 \le w_i \le w_{i,\max}$$
- **Dynamic Alpha Blending Basado en Divergencia Relativa:** Para evitar que errores extremos de estimación en *longshots* generen señales falsas de valor, las probabilidades del modelo se combinan con las probabilidades *fair* de apertura mediante un decaimiento gaussiano de la **divergencia relativa**:
  $$\alpha = \alpha_{\text{min}} + (\alpha_{\text{max}} - \alpha_{\text{min}}) \cdot \exp\left(-6.0 \cdot \left(\frac{|p_{\text{model}} - p_{\text{market}}|}{\max(p_{\text{market}}, 0.05)}\right)^2\right), \qquad p_{\text{blend}} = \alpha \, p_{\text{model}} + (1 - \alpha) \, p_{\text{market}}$$
- **Optimización Causal de Hiperparámetros Financieros:** Los parámetros operativos por liga ($\alpha_{\text{max}}, \alpha_{\text{min}}$, umbral mínimo de $EV$, cuota máxima y multiplicador de Kelly) se optimizan **exclusivamente sobre las predicciones Out-Of-Fold del conjunto de entrenamiento (`train_predictions.parquet`)** y se congelan antes de evaluar el conjunto de prueba (`test_predictions.parquet`).

### 4. Validación Out-Of-Sample y Test de Resistencia Monte Carlo
Sobre el conjunto de prueba estrictamente Out-Of-Sample (`test_predictions.parquet`, partidos posteriores al `2023-08-08`, con **20,918 partidos totales** y **15,081 partidos con cuotas de apertura válidas**):
- **Calidad Probabilística (Meta-Stacker Nivel 2):** Log-Loss global de **1.0065**, Brier Score multiclase de **0.2007** y Accuracy de **49.93%** en 3 vías (`1X2`).
- **Simulación Financiera Out-Of-Sample (100% Causal, Cero Look-Ahead Bias):** Alta selectividad (**301 apuestas**, **2.0%** de los partidos evaluados), tasa de acierto de **39.2%** a cuota promedio de **2.64**, **Probabilidad de Ruina (PoR) de 0.00%**, **Maximum Drawdown (MDD) de 21.99%**, Yield Real global de **-1.12%** (percentil **24.2%** de calibración en 1,000 trayectorias de Monte Carlo bajo nuestras probabilidades mezcladas) y rentabilidad positiva Out-Of-Sample en ligas de menor eficiencia como **Segunda División Española (`SP2`, +51.19% Yield)**, **Ligue 2 (`F2`, +33.57%)**, **EFL League One (`E2`, +23.38%)**, **Premier League (`E0`, +18.86%)**, **Super League Grecia (`G1`, +10.44%)** y **LaLiga (`SP1`, +3.06%)**.

---

## Flujo Operativo y de Ejecución Cuantitativo (CLI & API)

Prophetia2 consolida millones de filas y peticiones, por lo que demanda un ecosistema hardware apto (Optimizaciones **NVIDIA CUDA** recomendadas para aceleración de meta-modelos y redes neuronales).

### 1. Predictor Cuantitativo Interactivo (CLI)
Para ejecutar inferencias en tiempo real sobre los próximos partidos con asignación de portafolio de Markowitz:
```bash
python core/cli_predictor.py
```

### 2. Pipeline de Actualización y Reentrenamiento Mensual Automatizado (Recomendado)

Para mantener los modelos calibrados con los resultados más recientes del fútbol mundial, se dispone del orquestador institucional maestro `core/update_monthly_pipeline.py`. Este script ejecuta automáticamente las 4 fases del sistema, registra logs visuales en consola con `rich`, almacena el historial detallado en la carpeta `logs/` (`logs/pipeline_monthly_update_YYYY-MM-DD.log`) y valida el rendimiento financiero antes de certificar los nuevos modelos:

#### A. Actualización Completa (Modo Mensual por Defecto)
Descarga los últimos resultados de todas las ligas europeas y complementarias (Football-Data), enriquece automáticamente las estadísticas avanzadas (córneres, tiros, tarjetas, árbitros y cuotas de Liga MX, MLS, J1, etc. vía ESPN FC), actualiza los valores de mercado de las plantillas en Transfermarkt, procesa las características dinámicas (EMAs, Elo, fatiga, H2H, xG Share), reentrena los 9 modelos especializados, ajusta el Meta-Stacker final y ejecuta la suite de validación financiera:
```bash
python core/update_monthly_pipeline.py
```
> **¿Para qué sirve?** Es el comando principal a ejecutar periódicamente (ej. una vez al mes) para que el bot y el CLI aprendan de los partidos recién disputados y ajusten sus probabilidades a la forma actual de los equipos.

#### B. Modo Anti-Bloqueo de Red (Omitir Scraping de Transfermarkt)
Si el servidor de Transfermarkt presenta restricciones de tasa de peticiones (rate limiting) o bloqueos temporales por IP, este comando actualiza todos los partidos oficiales y cuotas utilizando las valoraciones de plantilla preexistentes:
```bash
python core/update_monthly_pipeline.py --skip-scraping
```
> **¿Para qué sirve?** Permite refrescar resultados y reentrenar todos los modelos sin riesgo de que el pipeline se detenga por fallas de conexión a Transfermarkt.

#### C. Modo sin Scraper de ESPN (Omitir Enriquecimiento de Ligas Extra)
Si ya cuentas con los CSVs de ligas complementarias (`MEX.csv`, `USA.csv`, etc.) previamente enriquecidos y con caché local:
```bash
python core/update_monthly_pipeline.py --skip-espn
```
> **¿Para qué sirve?** Acelera la Fase 1 reutilizando las estadísticas de córneres, tiros y tarjetas de ESPN ya presentes en disco.

#### D. Solo Reentrenamiento y Validación (Omitir Descargas)
Si ya cuentas con los datasets procesados y únicamente deseas reentrenar los 9 modelos cuantitativos, ajustar el Meta-Stacker y verificar las métricas financieras:
```bash
python core/update_monthly_pipeline.py --train-only
```
> **¿Para qué sirve?** Ahorra tiempo de descarga cuando estás ajustando hiperparámetros o algoritmos de Machine Learning sobre los datos locales existentes.

#### E. Validación Financiera y Pruebas del Motor CLI (Control de Calidad)
Ejecuta de manera inmediata la simulación de bankroll a gran escala (`simulate_bankroll.py`), la prueba de rendimiento en Liga MX 2026 (`test_cli_predictor_liga_mx.py`) y la validación en el conjunto Out-Of-Sample internacional 2026 (`test_cli_predictor_real_data.py`):
```bash
python core/update_monthly_pipeline.py --validate-only
```
> **¿Para qué sirve?** Para auditar el ROI, Winrate, Profit Factor, Drawdown y comportamiento de la gestión de capital (Markowitz/Kelly) sin modificar ni reentrenar los archivos de modelos existentes.

---

### 3. Enriquecimiento de Estadísticas de Partidos (ESPN FC Match Stats Scraper)

Para ligas complementarias fuera de los archivos estándar de 106 columnas de Football-Data (como Liga MX, MLS, J1 League, etc.), dispones del scraper especializado de alto rendimiento con multihilo y caché en disco `ingestion/espn_match_stats_scraper.py`:

```bash
# Enriquecer Liga MX (por defecto histórico 2014-2026):
python ingestion/espn_match_stats_scraper.py --league MEX

# Enriquecer MLS (USA):
python ingestion/espn_match_stats_scraper.py --league USA

# Enriquecer todas las ligas complementarias (MEX, USA, JPN, SWE, NOR, DNK, SWZ, AUT):
python ingestion/espn_match_stats_scraper.py --league ALL --max-workers 20

# Enriquecer rangos de años específicos o forzar refresco:
python ingestion/espn_match_stats_scraper.py --league MEX --start-year 2023 --end-year 2026 --force-refresh
```

---

### 4. Ejecución Manual Paso a Paso (Pipeline Modular)

Si prefieres ejecutar cada componente de forma individual:

1. **Ingesta, Scraping y Enriquecimiento de Datos**:
   Descarga de eventos crudos, enriquecimiento de estadísticas avanzadas de ESPN (córneres `HC`/`AC`, tiros `HS`/`AS`/`HST`/`AST`, tarjetas `HY`/`AY`/`HR`/`AR`, árbitros `Referee`), scraping de valores de plantilla y unificación del dataset intermedio.
   ```bash
   # Opción A: Descarga y enriquecimiento integrado
   python core/download_football_data.py --force-refresh --enrich-espn
   
   # Opción B: Por pasos separados
   python core/download_football_data.py --force-refresh
   python ingestion/espn_match_stats_scraper.py --league ALL
   python ingestion/transfermarkt_scraper.py
   python core/data_adapter.py
   ```
2. **Ingeniería de Características y Cuotas de Mercado**:
   Cálculo de medias móviles temporales (EMAs), Elo dinámico, xG Share, fatiga, H2H, extracción de cuotas Pinnacle/Bet365 y métricas disciplinarias arbitrales.
   ```bash
   python core/feature_engineering.py
   python ingestion/fetch_odds.py
   python ingestion/fetch_referees.py
   ```
3. **Reentrenamiento de los 9 Modelos Especializados y Meta-Stacker**:
   Entrenamiento paralelo con validación temporal estricta (`data_splitter.py`) y ensamble de segundo nivel libre de *Data Leakage*.
   ```bash
   python core/models/train_quant_advanced.py
   python core/models/train_context.py
   python core/models/train_nn.py
   python core/models/train_draws.py
   python core/market_models/train_market.py
   python core/market_models/train_gbm_model.py
   python core/models/train_corners_model.py
   python core/models/train_cards_model.py
   python core/models/train_shots_on_goal.py
   python core/models/train_stacker.py
   python core/market_models/train_clv_model.py
   ```
4. **Validación Financiera Global**:
   ```bash
   python core/simulate_bankroll.py
   python tests/test_cli_predictor_liga_mx.py
   python tests/test_cli_predictor_real_data.py
   ```

### 5. Servidor de Inferencia API (FastAPI)
Para exponer el motor cuantitativo y consultar predicciones vía REST API:
```bash
uvicorn api.main:app --reload
```

---

## Trading Automatizado: Polymarket HFT Bot
La rama `polymarket_bot/main.py` contiene un bot enfocado en el comercio de frecuencias de media a alta (HFT Trading adaptado) en entornos blockchain on-chain como **Polymarket**.
- Implementa *Dynamic Alpha Blending*: Castiga matemáticamente nuestra convicción predictiva si el mercado global (*Wisdom of Crowds*) presenta alta entropía y consenso en nuestra contra.
- Aplica Coberturas (Dutching) óptimas distribuyendo liquidez iterativamente minimizando el impacto de *Slippage* del AMM (Automated Market Maker).
- Protege los retornos netos descontando las *Network Fees* dinámicas antes de ejecutar el *Trade*.

### Análisis Visual Exploratorio (Notebooks)
Para propósitos de investigación cuántica (R&D), los investigadores disponen del entorno:
```bash
jupyter notebook
```
Donde `03_live_dashboard.ipynb`, `02_model_selection.ipynb` y `01_data_exploration.ipynb` permiten inspeccionar matrices de Permutation Importance para auditar qué vectores del espacio n-dimensional guían las inferencias, así como simulaciones de volatilidad pre-partido.

---
**Desarrollado y Auditado por:** 
* **César Becerra Valencia** & **José Luis Cortés Nava**: Ingeniería de Software, Machine Learning y Procesamiento Estocástico.
* **Dylan Eduardo Becerra Valencia**: Auditoría Cuantitativa de Riesgo y Arbitraje Estadístico.

---
## Licencia (Uso Académico y Restricción Comercial)

Este proyecto está protegido bajo una **Licencia Propietaria de Uso Académico y Restricción Comercial Estricta**.
El código fuente está disponible exclusivamente para fines de **investigación científica y académica**. 

Queda **estrictamente prohibido el uso de este software (total o parcial) para fines comerciales**, generación de ingresos monetarios, ejecución en fondos cuantitativos, *syndicates* deportivos, o automatización de trading sin el consentimiento explícito por escrito de los autores originales (César Becerra y José Luis Cortés) y un **Acuerdo Formal de Partición de Ingresos (Revenue Sharing)**.

Cualquier infracción, copia del código para beneficio propio o uso no autorizado en entornos de producción financiera será perseguido legalmente. Para más detalles, consulta el archivo legal [LICENSE](file:///c:/Users/cesar/OneDrive/Escritorio/Proyectos%20De%20Cesar/Proyectos%20Python/Prophetia2/LICENSE).
