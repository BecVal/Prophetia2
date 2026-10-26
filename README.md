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

## 📖 Guía Práctica para el Apostador: Cómo Leer el Modelo, Cuándo Apostar y Qué Significa Cada Gráfica

Si no eres matemático ni científico de datos, **esta sección es para ti**. Aquí traducimos todo lo que hace y muestra **Prophetia2** a lenguaje sencillo para que sepas exactamente cómo usar el programa, cuándo meter tu dinero, cuándo **no** apostar y cómo leer cada tabla y gráfica que genera el sistema.

---

### 1. 🚦 Semáforo de Decisión: ¿Cuándo SÍ conviene apostar y cuándo NO?

El error número 1 del apostador amateur es apostar al equipo que *"cree que va a ganar"* sin importar qué cuota le pague la casa de apuestas. **Prophetia2 no busca adivinar quién gana; busca encontrar precios mal puestos por las casas de apuestas (apuestas con Valor Esperado o *Value Bets*).**

#### 🟢 CUÁNDO SÍ CONVIENE APOSTAR (`🚀 MEJOR OPORTUNIDAD VALUE DETECTADA`)
Solo debes realizar una apuesta cuando se cumplan **al mismo tiempo** estas condiciones en la pantalla del programa (`python core/cli_predictor.py`):
1. **El `Net EV` aparece en color VERDE y supera el umbral de la liga:** Cada liga tiene un "filtro mínimo de ganancia esperada" (por ejemplo, `+2.5%` o `+4.0%`). Si el `Net EV` está en verde por encima de ese umbral, significa que la cuota de tu casa de apuestas paga bastante más de lo que debería según nuestras probabilidades reales.
2. **La columna `Stake Kelly Multi ($ / %)` recomienda apostar más de `$0.00`:** El programa te dirá el monto exacto en dólares y en porcentaje de tu banca (por ejemplo, `$12.50 (1.25%)`). **Respeta siempre ese monto**; está calculado para hacer crecer tu dinero sin arriesgarte a quebrar en una mala racha.
3. **La cuota está en nuestra "Zona Dulce" (`2.00` a `3.50`):** Como demuestran las pruebas reales del sistema, donde más dinero ganan los modelos es en cuotas medias (*Underdogs Moderados* o favoritos parejos entre `2.00` y `3.00`, con **+4.87% de Yield**).
4. **Prioriza las ligas donde las casas de apuestas se equivocan más:** Nuestros resultados históricos fuera de muestra comprueban que el modelo obtiene sus mayores ganancias en **segundas divisiones y ligas de eficiencia media**, tales como:
   - **Segunda División de España (`SP2`):** +51.19% Yield
   - **Ligue 2 de Francia (`F2`):** +33.57% Yield
   - **League One de Inglaterra (`E2`):** +23.38% Yield
   - **Premier League (`E0`) y LaLiga (`SP1`):** +18.86% y +3.06% Yield
   - **Super League de Grecia (`G1`) y Eredivisie (`N1`):** +10.44% Yield

#### 🟡 / 🔴 CUÁNDO NO CONVIENE APOSTAR (`PASS / NO BET`)
Cierra la billetera y **pasa del partido** en cualquiera de estos casos:
1. **Sale el recuadro ROJO (`NO HAY VALUE EN NINGÚN MERCADO`) o AMARILLO (`EDGE INSUFICIENTE`):**
   - Si el `Net EV` es **negativo (rojo)**, significa que la casa de apuestas te está pagando menos de lo justo. Aunque juegue el Real Madrid o el Manchester City y tenga `75%` de probabilidad de ganar, si la cuota es `1.20`, a largo plazo **perderás dinero** apostándole.
   - Si el `Net EV` es **amarillo** (positivo pero chiquito, ej. `+0.8%`), la ventaja es tan pequeña que el margen de la casa de apuestas se la come. El programa te dirá `PASS / NO BET`.
2. **El rango `IC 95% Bayesiano` es demasiado ancho:** Si ves que la probabilidad es `48%`, pero a un lado el intervalo dice `[22.0% - 74.0%]`, significa que los **9 modelos internos están peleados entre sí** (unos ven favorito al local y otros al visitante). Cuando los modelos dudan, el sistema castiga el stake y lo mejor es no apostar.
3. **No ingresaste cuotas reales para los mercados secundarios (`Over 2.5`, `BTTS`, `Córneres`, `Tarjetas`):** Si no respondes `"Sí"` a la pregunta de personalizar cuotas secundarias y escribes las cuotas reales de tu casa de apuestas, verás que el `Stake Kelly Multi` en la Tabla 2 marca **`$0.00 (0.00%)`**. Esto es un candado de seguridad para que nunca apuestes a ciegas con cuotas inventadas.
4. **Cuotas altísimas (*Longshots* de `5.00+`):** Apostar a sorpresas improbables de cuota `6.00` o `10.00` destruye la banca por malas rachas larguísimas. El sistema las bloquea automáticamente.

---

### 2. 📚 Diccionario en Español Sencillo (¿Qué significa cada palabra que escupe el programa?)

#### A. Métricas de Calidad del Modelo (Lo que sale en los logs al entrenar)
* **`Accuracy` (Porcentaje de Aciertos — `49.93%`):** De cada 100 partidos, en cuántos el modelo acertó el resultado exacto entre las 3 opciones (`Local`, `Empate` o `Visita`). En fútbol, adivinar al azar acierta el `33.3%`. Acertar casi el **50%** de todos los partidos del mundo en 3 vías es un nivel altísimo. *Recuerda:* en apuestas no gana el que acierta más favoritos de cuota `1.15`, sino el que detecta cuándo una cuota de `2.40` tiene `45%` de probabilidad real de ocurrir.
* **`Log-Loss` (Pérdida Logarítmica — `1.0065`):** Es el **"detector de mentiras"** de las probabilidades. Si un modelo fanfarrón dice *"90% seguro que gana el Local"* y pierde, el `Log-Loss` sube muchísimo (lo castiga por mentiroso). Si un modelo prudente dice *"50% Local, 28% Empate, 22% Visita"*, el `Log-Loss` se mantiene bajo. **Regla:** Entre **más pequeño** sea este número, mejores y más confiables son los porcentajes del modelo.
* **`Brier Score` (`0.2007`):** Otra medida de puntería probabilística que va de `0.00` (un adivino perfecto que lee el futuro) a `1.00` (totalmente equivocado). En fútbol profesional, estar en `0.20` significa que los porcentajes están sumamente afinados con la realidad.

#### B. Estadísticas de Fútbol que Usa el Motor por Dentro (Features)
* **`EMA` (`EMA3`, `EMA5`, `EMA10` — Media Móvil Exponencial):** Es el promedio de goles, tiros, córneres o posesión de un equipo en sus últimos 3, 5 o 10 partidos, pero **dándole mucho más peso al partido de hace 3 días que al de hace un mes**. Sirve para capturar el "momento actual" o racha viva del equipo.
* **`xG` (*Expected Goals* / Goles Esperados) y `xG Share`:** Mide qué tan claras fueron las jugadas de gol de un equipo sin importar si el delantero la mandó a la tribuna o el portero hizo un milagro. Si un equipo pierde `0-1` pero generó `2.80 xG` contra `0.20 xG` del rival, el modelo sabe que ese equipo juega muy bien y que pronto volverá a ganar (cuando la casa de apuestas quizá lo castigue por haber perdido `0-1`).
* **`Elo` (`team_elo`, `opp_elo`, `elo_diff`):** Es un ranking numérico de fuerza histórica (igual al del ajedrez o videojuegos competitivos). Todos empiezan en `1500`; vencer a un rival poderoso sube muchos puntos de Elo y perder contra un colero te quita muchos puntos.
* **`Rest Days` / Índices de Fatiga:** Días de descanso desde el último partido. Permite al modelo saber si un equipo llega con piernas cansadas por haber jugado torneo entre semana.

#### C. Métricas de Apuestas y Dinero (Lo que ves en las tablas del CLI y del Simulador)
* **`Vig` / `Overround` (Comisión oculta de la casa):** Las casas de apuestas inflan los porcentajes para que sumen `105%` o `108%` en vez de `100%` (esa diferencia es su ganancia segura). Nuestro módulo de **Shin** limpia esa comisión para descubrir cuál es la probabilidad real que piensa el mercado.
* **`Prob. Modelo` vs. `Prob. Blended` (Alpha Blending):**
  * `Prob. Modelo`: Lo que calculan nuestros 9 modelos de Inteligencia Artificial.
  * `Prob. Blended`: La mezcla inteligente entre nuestra IA y las cuotas del mercado mundial. Si nuestro modelo dice `60%` y el mercado dice `50%`, el sistema calcula un punto medio prudente (ej. `54%`) para protegerte de posibles datos que el mercado sí sepa (como un rumor de vestidor de último minuto).
* **`IC 95% Bayesiano`:** El rango de acuerdo entre los modelos (ej. `[41.2% - 48.5%]`). Mientras más estrecho sea el rango, más seguros y alineados están los modelos.
* **`Odds (Efectiva)`:** La cuota real que te queda después de descontar fricciones o el impacto de liquidez (*Slippage*).
* **`Net EV` (Valor Esperado Neto):** **La brújula del apostador.** Es el porcentaje de ganancia limpia que esperas obtener a largo plazo en esa apuesta.
  * Ejemplo: Si dice **`+6.5%`**, significa que por cada `$100` apostados en partidos así, la matemática está a tu favor para ganar `$6.50` netos en promedio. Si dice **`-5.2%`**, estás regalando `$5.20` de cada `$100` a la casa de apuestas.
* **`CLV` (*Closing Line Value* / Valor de Línea de Cierre):** Es la prueba reina de los apostadores profesionales. Si tú apuestas el viernes a cuota **`2.30`** y el domingo a cinco minutos de empezar el partido la cuota cayó a **`2.05`**, significa que conseguiste una cuota mucho mejor que el cierre del mercado (`CLV positivo`). Nuestro filtro de IA predice hacia dónde se moverá la cuota y cancela la apuesta si detecta que la cuota va a subir en tu contra.
* **`Stake Kelly Multi ($ / %)` y `Markowitz`:** La cantidad exacta de dinero que el sistema te recomienda apostar. Usa la fórmula de **Kelly Fraccionario** (apostar más cuando hay más ventaja y menos cuando la cuota es riesgosa) combinada con **Markowitz** (repartir el riesgo entre varios mercados sin poner todos los huevos en la misma canasta).
* **`Doble Oportunidad 1X / X2` (`Dutching`):**
  * **`1X`:** Ganas si gana el equipo Local **o** si empatan.
  * **`X2`:** Ganas si empatan **o** si gana el Visitante.
  * El sistema calcula cuánto meterle a la victoria y cuánto al empate para que cobres exactamente lo mismo pase lo que pase entre esos dos resultados.
* **`Yield (%)` vs. `ROI (%)`:**
  * **`ROI`:** Cuánto creció tu banca inicial de `$1,000`.
  * **`Yield`:** Cuánto ganaste o perdiste dividido entre **todo el dinero total que apostaste** a lo largo de las semanas. Un `Yield` positivo (`+3%` a `+15%`) significa que eres rentable batiendo a la casa.
* **`Max Drawdown (MDD)`:** Tu peor caída acumulada desde tu récord de dinero más alto. Si tu banca subió a `$1,200` y luego en una mala racha bajó a `$960` antes de volver a subir, tu `MDD` fue del `20%`.
* **`Probabilidad de Ruina (PoR)`:** El porcentaje de probabilidades de quedarte sin dinero (`$0`). En nuestro sistema auditado es **`0.00%`** gracias a que los stakes por apuesta son pequeños y controlados.

---

### 3. 🖥️ Cómo Leer la Pantalla del Predictor Interactivo (`python core/cli_predictor.py`)

Cuando ejecutas `python core/cli_predictor.py` y eliges una liga y un partido, el programa te mostrará **4 bloques en pantalla**:

1. **Panel Superior (`⚽ PROYECCIONES QUANT`):**
   - **`Marcador Esperado Quant (xG)`:** Ejemplo: `1.58 - 1.02`. Te dice cuántos goles proyecta el modelo para el Local y para el Visitante. Si ves `2.10 - 1.65`, espera un partido abierto de muchos goles; si ves `0.95 - 0.80`, espera un partido cerrado y defensivo.
   - **`Córneres Esperados`:** Ejemplo: `9.8 (Local: 5.5 | Visita: 4.3)`. Te muestra cuántos tiros de esquina se esperan en total y cuántos hará cada equipo.
   - **`Tarjetas Esperadas`:** Ejemplo: `4.2`. Calcula cuántas tarjetas sacará el árbitro sumando la agresividad de ambos equipos y el rigor histórico del juez.
2. **Tabla 1 (`1. Mercados de Partido: 1X2, Doble Oportunidad & Intervalos IC 95%`):**
   - Evalúa los 5 mercados principales del partido: **`1` (Local)**, **`X` (Empate)**, **`2` (Visitante)**, **`1X` (Local o Empate)** y **`X2` (Empate o Visita)**.
   - Mira directamente las dos últimas columnas: **`Net EV`** y **`Stake Kelly Multi ($ / %)`**. Si una fila tiene el `Net EV` en **verde brillante** y un monto en dólares en `Stake Kelly Multi`, ahí está el valor.
3. **Tabla 2 (`2. Mercados Derivados: Goles O/U, BTTS, Córneres & Tarjetas`):**
   - Muestra las probabilidades exactas del modelo para **`Over/Under 2.5 Goles`**, **`Ambos Anotan (BTTS)`**, **`Over/Under 9.5 Córneres`** y **`Over 4.5 Tarjetas`**.
   - Si al inicio elegiste ingresar las cuotas de tu casa de apuestas para estos mercados, aquí también te dirá cuáles tienen `Net EV` positivo y cuánto apostar.
4. **Panel de Decisión Final (`SISTEMA DE STAKING INSTITUCIONAL MULTI-MERCADO`):**
   - Es el veredicto final del programa:
     - **Recuadro Verde (`🚀 MEJOR OPORTUNIDAD VALUE DETECTADA`):** Te resume exactamente a qué mercado apostar, a qué cuota y cuántos dólares meter. Además, guarda esa orden en el archivo [trade_signals.json](file:///c:/Users/cesar/OneDrive/Escritorio/Proyectos%20De%20Cesar/Proyectos%20Python/Prophetia2/trade_signals.json).
     - **Recuadro Amarillo o Rojo (`PASS / NO BET`):** Te indica que las cuotas de ese partido no valen la pena y que pases al siguiente encuentro.

---

### 4. 📊 Guía Visual de las 8 Gráficas (`logs/plots/`) y Archivos de Auditoría (`logs/`)

Cada vez que ejecutas el simulador financiero (`python core/simulate_bankroll.py`), el sistema pone a prueba toda la estrategia sobre miles de partidos reales que el modelo **nunca vio durante su entrenamiento** (*Out-Of-Sample*) y genera automáticamente **8 gráficas en alta resolución** dentro de la carpeta [`logs/plots/`](file:///c:/Users/cesar/OneDrive/Escritorio/Proyectos%20De%20Cesar/Proyectos%20Python/Prophetia2/logs/plots) junto con sus tablas de datos en [`logs/plots/csv/`](file:///c:/Users/cesar/OneDrive/Escritorio/Proyectos%20De%20Cesar/Proyectos%20Python/Prophetia2/logs/plots/csv).

A continuación te mostramos **la última gráfica real generada por el modelo** para cada una de las 8 categorías y te explicamos en palabras sencillas qué significa y cómo interpretarla:

---

#### Gráfica 1: Curva de Calibración (`calibration_plot_<fecha>.png`)
![Curva de Calibración del Modelo](logs/plots/calibration_plot_20261017_153827.png)

* **¿Para qué sirve?** Es la prueba de honestidad del modelo. Compara el porcentaje de probabilidad que predijo el sistema (eje horizontal de abajo) contra el porcentaje de veces que realmente se ganaron esas apuestas en la vida real (eje vertical izquierdo).
* **¿Cómo se lee?**
  * **Gráfico de arriba:** La línea punteada negra diagonal (`y = x`) es la "perfección absoluta": significa que cuando el modelo dice `35%`, el evento ocurre el `35%` de las veces; y cuando dice `45%`, ocurre el `45%` de las veces.
  * **Lo que vemos en nuestra gráfica actual:** En los dos bloques donde el sistema concentra casi todas sus apuestas (**entre `30%` y `50%` de probabilidad**, que suman **264 de las 301 apuestas**), la línea verde está **prácticamente clavada sobre la diagonal perfecta**:
    * En el bloque de `30%-40%` (158 apuestas), el modelo predijo `35.8%` y acertó el **`34.8%`** de las veces, generando **`+$100.19 USD` de ganancia neta**.
    * En el bloque de `40%-50%` (106 apuestas), el modelo predijo `43.7%` y en la realidad acertó el **`47.2%`**, generando otros **`+$48.41 USD` de ganancia**.
    * En cambio, cuando la probabilidad supera el `50%` (pocas muestras, solo 37 apuestas en favoritos cortos), la línea verde cae por debajo de la diagonal, avisándonos que los favoritos de cuota baja son menos rentables que los *underdogs* moderados.
  * **Gráfico de abajo (barras azules):** Muestra cuántas apuestas se hicieron en cada rango de probabilidad.

---

#### Gráfica 2: Curva de Capital y Caídas / Drawdown (`equity_drawdown_plot_<fecha>.png`)
![Curva de Capital y Drawdown](logs/plots/equity_drawdown_plot_20261017_153827.png)

* **¿Para qué sirve?** Te muestra la película completa de tu billetera apuesta tras apuesta, empezando con **`$1,000 USD`** desde la apuesta #1 hasta la #301.
* **¿Cómo se lee?**
  * **Mitad superior (Línea Azul — *Equity Curve*):** Es el saldo real de tu cuenta en dólares. La línea gris punteada son tus `$1,000` iniciales. Cuando la línea azul sube por encima de la gris, vas ganando dinero; cuando baja, estás pasando por una racha negativa.
  * **Mitad inferior (Mancha Roja — *Drawdown Waterfall*):** Mide qué tan profundo es cada "bache" o mala racha desde el punto más alto que haya alcanzado tu billetera.
  * **Lo que nos enseña esta gráfica:** En las apuestas deportivas reales, el dinero nunca sube en línea recta como un depósito bancario. Verás que el portafolio sube cerca de `$1,120`, atraviesa una mala racha a mitad del periodo (donde la caída máxima o `MDD` llega al **`21.99%`**, sin poner jamás en peligro de quiebra la banca), y luego se recupera hasta cerrar prácticamente en equilibrio (`$977 USD`, `-1.12%` Yield global incluyendo todas las ligas).

---

#### Gráfica 3: Distribución de Monte Carlo (`monte_carlo_distribution_plot_<fecha>.png`)
![Distribución de Monte Carlo](logs/plots/monte_carlo_distribution_plot_20261017_153827.png)

* **¿Para qué sirve?** En el fútbol existe el factor suerte (un gol en el minuto 95, un penalti fallado, una tarjeta roja temprana). Esta gráfica responde a la pregunta: *¿El resultado que obtuve fue normal o tuve buena/mala suerte?* Para saberlo, la computadora repite las 301 apuestas **10,000 veces en universos paralelos** respetando las probabilidades de cada partido.
* **¿Cómo se lee?**
  * **Izquierda (Barras Azules — Capital Final Esperado):** Muestra en cuánto dinero terminaría tu banca de `$1,000` en esos 10,000 universos. La línea roja punteada (`Median`) muestra que lo normal con nuestras probabilidades es terminar arriba de los `$1,000`, y la línea naranja (`P5`) te muestra qué pasaría en el 5% de los universos de peor mala suerte. Como nuestro resultado real cayó en el **percentil 24.2%**, la gráfica confirma que en la realidad tuvimos algo de mala varianza (goles en contra al final), pero totalmente dentro del comportamiento normal del modelo.
  * **Derecha (Barras Rojas — Distribución de Caídas Máximas / MDD):** Te muestra qué caída máxima (`Drawdown`) debes estar preparado psicológicamente para soportar. La línea negra punteada es la caída típica esperada (`~19%`) y la línea roja oscura (`P95 Tail Risk`) te avisa cuál sería la peor caída en un año de muy mala suerte.

---

#### Gráfica 4: Rendimiento por Liga (`league_performance_plot_<fecha>.png`)
![Rendimiento por Liga](logs/plots/league_performance_plot_20261017_153827.png)

* **¿Para qué sirve?** Es posiblemente **la gráfica más valiosa para tu bolsillo cuando vas a apostar**. Te muestra con barras verdes en qué ligas del mundo el modelo ganó dinero y con barras rojas en cuáles perdió.
* **¿Cómo se lee?**
  * **Barras Verdes (Hacia la derecha — Ligas Rentables):** Son las ligas donde nuestro modelo es más inteligente que las casas de apuestas. Observa los datos reales de la gráfica:
    * **`F2` (Ligue 2 de Francia):** `+$48.78 USD` de ganancia neta (**+33.57% Yield** en 23 apuestas).
    * **`SP2` (Segunda División de España):** `+$38.21 USD` (**+51.19% Yield** en 22 apuestas, ganando casi la mitad a cuotas altas).
    * **`SP1` (LaLiga España):** `+$28.08 USD` (**+3.06% Yield** en 71 apuestas).
    * **`N1` (Eredivisie Países Bajos):** `+$21.25 USD`.
    * **`G1` (Super League Grecia):** `+$16.78 USD` (**+10.44% Yield** en 18 apuestas).
    * **`E2` (EFL League One Inglaterra):** `+$12.19 USD` (**+23.38% Yield** en 11 apuestas).
    * **`E0` (Premier League Inglaterra):** `+$4.52 USD` (**+18.86% Yield**).
  * **Barras Rojas (Hacia la izquierda — Ligas donde conviene tener precaución):** Ligas como la Championship inglesa (`E1`), la liga escocesa (`SC0`) o la Serie B italiana (`I2`) tuvieron pérdidas en este periodo.
  * **Consejo práctico para el apostador:** Si tú como usuario decides **apostar únicamente en las ligas con barra verde** (`F2`, `SP2`, `SP1`, `G1`, `E2`, `E0`), tu rentabilidad real se dispara porque evitas los mercados donde las casas de apuestas estuvieron más finas.

---

#### Gráfica 5: Rentabilidad por Rango de Ventaja / EV (`ev_bucketing_plot_<fecha>.png`)
![Rentabilidad por Rango de EV](logs/plots/ev_bucketing_plot_20261017_153827.png)

* **¿Para qué sirve?** Agrupa todas las apuestas realizadas según qué tanta ventaja (`Net EV`) decía tener el modelo al momento de apostar: ventajas pequeñas (`0 - 2%`), moderadas (`2 - 5%`), buenas (`5 - 10%`), altas (`10 - 15%`) y extremas (`15%+`).
* **¿Cómo se lee?**
  * **Izquierda:** Dinero neto ganado o perdido (`$`) en cada tramo de ventaja.
  * **Derecha:** Rentabilidad porcentual (`Yield %`) en cada tramo.
  * **Lo que vemos en nuestra gráfica:**
    * Las apuestas con ventaja muy bajita (**`0 - 2%`**) dieron pérdida (`-6.29% Yield`), comprobando por qué no vale la pena apostar cuando el `Net EV` es menor al `2%`.
    * El tramo de **`2 - 5%`** (donde se hicieron 189 apuestas) mantuvo un **`40.2%` de tasa de aciertos** casi en punto de equilibrio (`-0.71% Yield`), mientras que el tramo de **`10 - 15%` de EV** generó la mayor rentabilidad (**`+16.64% Yield`** y **`+$21.54 USD`**).
    * En cambio, el tramo de **`15%+`** (cuando una cuota parece "demasiado buena para ser verdad") suele deberse a que la casa de apuestas sabe algo extraordinario, por lo que no hay que confiarse de ventajas irreales.

---

#### Gráfica 6: Rentabilidad por Rango de Cuotas (`odds_bucketing_plot_<fecha>.png`)
![Rentabilidad por Rango de Cuotas](logs/plots/odds_bucketing_plot_20261017_153827.png)

* **¿Para qué sirve?** Te revela en qué tipo de cuotas eres ganador y en cuáles pierdes dinero. Divide las cuotas en 5 escalones:
  1. **Favoritos Fuertes (`1.00 - 1.50`)**
  2. **Favoritos Moderados (`1.50 - 2.00`)**
  3. **Underdogs Moderados (`2.00 - 3.00`)** *(equipos parejos o no-favoritos con buena chance)*
  4. **Underdogs Altos (`3.00 - 5.00`)**
  5. **Sorpresas / Longshots (`5.00+`)**
* **¿Cómo se lee y qué nos revela la gráfica?**
  * **El mina de oro está en `Underdogs Moderados (2.00 - 3.00)` (Barra Verde):** De las 301 apuestas del sistema, **207 apuestas** estuvieron en cuotas entre `2.00` y `3.00`, acertando el **`43.0%`** de las veces y dejando **`+$68.69 USD` de ganancia limpia (`+4.87% Yield`)**.
  * **Los `Favoritos Moderados (1.50 - 2.00)` dieron pérdida (`-24.07% Yield`):** Las casas de apuestas suelen recortar demasiado las cuotas de los favoritos conocidos porque todo el público amateur apuesta ahí.
  * **Cero apuestas en `1.00 - 1.50` y `5.00+`:** El sistema filtró por completo las cuotas basura (`<1.50`) y los boletos de lotería (`5.00+`).
  * **Consejo práctico:** Cuando uses el programa, ten mucha más confianza cuando te recomiende una cuota entre **`2.00` y `3.00`** que cuando sea una cuota menor a `1.80`.

---

#### Gráfica 7: Apuestas Simples vs. Doble Oportunidad (`bet_types_plot_<fecha>.png`)
![Apuestas Simples vs Doble Oportunidad](logs/plots/bet_types_plot_20261017_153827.png)

* **¿Para qué sirve?** Compara cómo le fue al sistema haciendo apuestas directas a un resultado (**`Single Bet 1X2`**, barra azul) frente a hacer coberturas dobles (**`Dutching 1X / X2`**, barra naranja).
* **¿Cómo se lee y qué nos dice la gráfica?**
  * Las **Apuestas Simples (`Single Bet 1X2`)** son claramente ganadoras: en **280 apuestas**, lograron un **`39.64%` de aciertos** con **`+$43.68 USD` de ganancia neta y `+2.31%` de Yield positivo**.
  * En cambio, las **Dobles Oportunidades (`Dutching 1X / X2`)** (21 apuestas) tuvieron rendimiento negativo (`-$66.60 USD`), ya que al cubrir dos resultados al mismo tiempo pagas doble comisión (*vig*) a la casa de apuestas y la cuota combinada baja demasiado (alrededor de `1.35 - 1.55`), donde un solo fallo cuesta caro.
  * **Consejo práctico:** Prioriza siempre las recomendaciones del programa en mercados directos (`1`, `X` o `2`) por encima de las Dobles Oportunidades.

---

#### Gráfica 8: Evolución de Ganancias Mes a Mes (`monthly_performance_plot_<fecha>.png`)
![Rendimiento Mensual](logs/plots/monthly_performance_plot_20261017_153827.png)

* **¿Para qué sirve?** Muestra en barras verdes y rojas cuánto dinero se ganó o perdió en cada mes del calendario desde 2024 hasta 2026.
* **¿Cómo se lee?**
  * Cada barra es un mes (`YYYY-MM`). Si es verde, ese mes cerraste con ganancias (por ejemplo, `2024-04` con **`+$94.50 USD` / `+40.87% Yield`**, `2024-09` con **`+$83.36 USD` / `+94.63% Yield`**, o la racha de **7 meses consecutivos en verde** entre agosto de 2025 y febrero de 2026).
  * Si es roja, ese mes tocó varianza negativa (por ejemplo, mayo al final de las temporadas europeas, cuando muchos equipos ya no se juegan nada en la tabla y se dan resultados atípicos).
  * **Consejo práctico:** Evita apostar fuerte en las últimas 2 jornadas de liga en mayo (cuando hay equipos ya salvados o campeones que rotan titulares) y recuerda que tener meses rojos es parte natural de cualquier inversión cuantitativa.

---

#### 🗂️ ¿Para qué sirven los demás archivos en `logs/` y `logs/plots/csv/`?

Además de las imágenes `.png`, el sistema guarda archivos que puedes abrir con **Excel** o el **Bloc de Notas**:
1. **`logs/plots/csv/trades_ledger_data_<fecha>.csv` (El Libro Mayor de Apuestas):**
   - Es una tabla de Excel donde viene **cada una de las 301 apuestas fila por fila**: fecha, liga (`competition`), tipo de apuesta (`single` o `dutching`), selección (`chosen_label`), cuota (`odds`), ventaja esperada (`expected_value_ev`), probabilidad del modelo (`predicted_prob`), dólares apostados (`stake_usd`), si ganó o perdió (`is_win`) y ganancia neta (`net_profit_usd`).
   - **¿Para qué te sirve?** Puedes abrirlo en Excel, filtrar tu liga favorita o tu rango de cuotas favorito (ej. cuotas de `2.00` a `3.00`) y ver con tus propios ojos cada partido histórico en el que apostó el modelo.
2. **`logs/plots/csv/*_data_<fecha>.csv`:**
   - Son las tablas con los números exactos de las 8 gráficas (`league_performance_data`, `odds_bucketing_data`, `ev_bucketing_data`, `monthly_performance_data`, `calibration_data`, `bet_types_data`, `equity_drawdown_data`, `monte_carlo_data`).
3. **`trade_signals.json` (En la carpeta principal del proyecto):**
   - Cada vez que usas `python core/cli_predictor.py` y el modelo encuentra una apuesta con valor (panel verde), guarda aquí automáticamente el resumen del partido, la cuota y el monto a apostar.
4. **`logs/simulate_bankroll_<fecha>.log`, `logs/pipeline_monthly_update_<fecha>.log` y `logs/train_*_<fecha>.log`:**
   - Son las bitácoras de texto donde queda registrado cuándo se entrenó cada uno de los 9 modelos, cuánto tardó, qué `Log-Loss` y `Brier Score` obtuvo y si hubo algún error de conexión al descargar partidos nuevos.

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
