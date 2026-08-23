import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Ensure root directories are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)

from core.logger_config import get_logger

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

console = Console()
logger = get_logger('monthly_pipeline', 'pipeline_monthly_update')

# Python executable
PYTHON_EXE = sys.executable or 'python'

def run_step(step_name, script_rel_path, allow_fail=False, timeout_seconds=None):
    """
    Ejecuta un script individual del pipeline, midiendo tiempo y registrando logs duales.
    """
    script_full_path = os.path.abspath(os.path.join(root_dir, script_rel_path))
    if not os.path.exists(script_full_path):
        err_msg = f"Script no encontrado: {script_full_path}"
        logger.error(err_msg)
        console.print(f"[bold red]❌ ERROR:[/bold red] {err_msg}")
        if not allow_fail:
            sys.exit(1)
        return False, 0.0

    logger.info(f"==================================================")
    logger.info(f"INICIANDO PASO: {step_name} [{script_rel_path}]")
    logger.info(f"==================================================")
    
    console.print(f"\n[bold bright_cyan]▶ [{datetime.now().strftime('%H:%M:%S')}] Ejecutando:[/bold bright_cyan] [bright_yellow]{step_name}[/bright_yellow] [dim]({script_rel_path})[/dim]")
    
    t0 = time.time()
    try:
        # Ejecución en el directorio donde reside el script para mantener consistencia de rutas relativas
        script_parent = os.path.dirname(script_full_path)
        proc = subprocess.Popen(
            [PYTHON_EXE, script_full_path],
            cwd=script_parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Transmitir salida en tiempo real al logger y consola
        for line in iter(proc.stdout.readline, ''):
            clean_line = line.strip()
            if clean_line:
                logger.info(f"[{step_name}] {clean_line}")
                # Imprimir líneas informativas en terminal con formato sutil
                if any(k in clean_line.lower() for k in ['error', 'exception', 'traceback', 'failed']):
                    console.print(f"  [red]{clean_line}[/red]")
                elif any(k in clean_line.lower() for k in ['guardado', 'exitosamente', 'completado', 'accuracy', 'winrate', 'roi', 'profit']):
                    console.print(f"  [bright_green]{clean_line}[/bright_green]")
                    
        proc.wait(timeout=timeout_seconds)
        elapsed = time.time() - t0
        
        if proc.returncode != 0:
            err_msg = f"Fallo en {step_name} (Código de salida: {proc.returncode})"
            logger.error(err_msg)
            console.print(f"[bold red]❌ {err_msg}[/bold red] [dim]({elapsed:.2f}s)[/dim]")
            if not allow_fail:
                logger.critical("Deteniendo pipeline debido a fallo en paso crítico.")
                sys.exit(1)
            else:
                logger.warning(f"Continuando pipeline ya que {step_name} es opcional/tolerante a fallos.")
                return False, elapsed
                
        logger.info(f"PASO COMPLETADO: {step_name} en {elapsed:.2f}s")
        console.print(f"[bold bright_green]✔ {step_name} completado con éxito[/bold bright_green] [dim]({elapsed:.2f}s)[/dim]")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"Excepción al ejecutar {step_name}: {e}", exc_info=True)
        console.print(f"[bold red]❌ Excepción en {step_name}: {e}[/bold red]")
        if not allow_fail:
            sys.exit(1)
        return False, elapsed

def main():
    parser = argparse.ArgumentParser(description="Pipeline Automatizado de Actualización y Reentrenamiento Mensual de Prophetia2")
    parser.add_argument('--full', action='store_true', default=True, help="Ejecutar pipeline completo (Ingesta, Features, Modelos, Stacker y Tests)")
    parser.add_argument('--train-only', action='store_true', help="Omitir ingesta y ejecutar solo reentrenamiento de modelos y validación")
    parser.add_argument('--validate-only', action='store_true', help="Omitir reentrenamiento y ejecutar solo la validación financiera y tests")
    parser.add_argument('--skip-scraping', action='store_true', help="Omitir el web scraper de Transfermarkt si hay bloqueos de red")
    args = parser.parse_args()

    start_time_all = time.time()
    
    console.print(Panel.fit(
        "[bold bright_yellow]PROPHETIA2 - PIPELINE DE ACTUALIZACIÓN Y REENTRENAMIENTO MENSUAL[/bold bright_yellow]\n"
        "[dim white]Orquestación Institucional de Ingesta, Feature Engineering, 9 Modelos, Meta-Stacker y Validación Financiera[/dim white]\n"
        f"[cyan]Fecha y Hora de Inicio:[/cyan] [white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/white]",
        border_style="bright_blue"
    ))

    logger.info("================================================================================")
    logger.info(f"INICIANDO PIPELINE MENSUAL DE REENTRENAMIENTO PROPHETIA2 ({datetime.now()})")
    logger.info("================================================================================")

    step_results = []

    # =========================================================================
    # FASE 1: INGESTA, SCRAPING Y FEATURE ENGINEERING
    # =========================================================================
    if not args.train_only and not args.validate_only:
        console.print("\n[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")
        console.print("[bold magenta]📁 FASE 1: SINCRONIZACIÓN DE DATOS, VALORACIÓN DE PLANTELES Y FEATURES[/bold magenta]")
        console.print("[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")
        
        # 1.1 Descarga de resultados de partidos más recientes
        success, t = run_step(
            "1.1 Descarga de Partidos Recientes (Football-Data)",
            "core/download_football_data.py"
        )
        step_results.append(("Descarga Football-Data", success, t))

        # 1.2 Web Scraping de Valores de Planteles (Transfermarkt)
        if not args.skip_scraping:
            success, t = run_step(
                "1.2 Scraping Valores de Mercado (Transfermarkt)",
                "ingestion/transfermarkt_scraper.py",
                allow_fail=True # Si Transfermarkt bloquea temporalmente por IP, se usan valores existentes
            )
            step_results.append(("Transfermarkt Scraping", success, t))

        # 1.3 Adaptador Universal de Eventos (StatsBomb + Football-Data)
        success, t = run_step(
            "1.3 Adaptador Universal de Datos (Data Adapter)",
            "core/data_adapter.py"
        )
        step_results.append(("Data Adapter", success, t))

        # 1.4 Ingeniería de Características Avanzadas (EMAs, Elo, xG Share, Fatiga, H2H)
        success, t = run_step(
            "1.4 Feature Engineering & Dynamic Elo/EMAs",
            "core/feature_engineering.py"
        )
        step_results.append(("Feature Engineering", success, t))

        # 1.5 Sincronización y Merge de Cuotas de Mercado (Pinnacle/B365)
        success, t = run_step(
            "1.5 Extracción y Fusión de Cuotas (Fetch Odds)",
            "ingestion/fetch_odds.py"
        )
        step_results.append(("Fetch Odds", success, t))

        # 1.6 Sincronización de Métricas Arbitrales y Disciplinarias
        success, t = run_step(
            "1.6 Extracción y Fusión de Árbitros (Fetch Referees)",
            "ingestion/fetch_referees.py"
        )
        step_results.append(("Fetch Referees", success, t))

    # =========================================================================
    # FASE 2: ENTRENAMIENTO DE MODELOS ESPECIALIZADOS Y GENERACIÓN DE OOF
    # =========================================================================
    if not args.validate_only:
        console.print("\n[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")
        console.print("[bold magenta]🧠 FASE 2: REENTRENAMIENTO DE 9 MODELOS CUANTITATIVOS Y GENERACIÓN OOF[/bold magenta]")
        console.print("[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")

        # 2.1 Modelo Cuantitativo Dinámico (Dixon-Coles + Kalman Filter)
        success, t = run_step(
            "2.1 Dixon-Coles Cuantitativo con Filtro de Kalman",
            "core/models/train_quant_advanced.py",
            allow_fail=True
        )
        step_results.append(("Quant Dixon-Coles/Kalman", success, t))

        # 2.2 Modelo de Contexto Táctico y Fatiga (CatBoost/RandomForest)
        success, t = run_step(
            "2.2 Clasificador de Contexto Táctico y Fatiga",
            "core/models/train_context.py"
        )
        step_results.append(("Context Classifier", success, t))

        # 2.3 Red Neuronal Profunda PyTorch Gated ResNet10
        success, t = run_step(
            "2.3 Deep Neural Network (PyTorch Gated ResNet10)",
            "core/models/train_nn.py"
        )
        step_results.append(("PyTorch ResNet10", success, t))

        # 2.4 Ensamble Especializado en Empates (Draws Ensemble)
        success, t = run_step(
            "2.4 Ensamble Especializado en Empates (Draws Ensemble)",
            "core/models/train_draws.py"
        )
        step_results.append(("Draws Ensemble", success, t))

        # 2.5 Modelo de Probabilidades y Fricción de Mercado
        success, t = run_step(
            "2.5 Modelo de Dinámicas de Mercado y Cuotas",
            "core/market_models/train_market.py"
        )
        step_results.append(("Market Classifier", success, t))

        # 2.6 Gradient Boosting de Mercado y Features GBM
        success, t = run_step(
            "2.6 Gradient Boosting de Mercado (GBM Features)",
            "core/market_models/train_gbm_model.py"
        )
        step_results.append(("GBM Model", success, t))

        # 2.7 Modelo de Córneres Totales (LightGBM + NegBinomial + Isotonic)
        success, t = run_step(
            "2.7 Modelo Especializado en Córneres Totales",
            "core/models/train_corners_model.py"
        )
        step_results.append(("Corners Model", success, t))

        # 2.8 Modelo de Tarjetas Totales (XGBoost)
        success, t = run_step(
            "2.8 Modelo Especializado en Tarjetas Totales",
            "core/models/train_cards_model.py"
        )
        step_results.append(("Cards Model", success, t))

        # 2.9 Modelo de Remates a Portería (Shots on Goal)
        success, t = run_step(
            "2.9 Modelo Especializado en Remates a Portería",
            "core/models/train_shots_on_goal.py"
        )
        step_results.append(("Shots on Goal Model", success, t))

        # =====================================================================
        # FASE 3: META-STACKING CUANTITATIVO Y SEGUIMIENTO CLV
        # =====================================================================
        console.print("\n[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")
        console.print("[bold magenta]🔗 FASE 3: META-STACKING CUANTITATIVO Y SEGUIMIENTO CLV[/bold magenta]")
        console.print("[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")

        # 3.1 Meta-Stacker (Fundamental + Final con Regularización L2/Optuna)
        success, t = run_step(
            "3.1 Meta-Stacker Fundamental y Final (Meta-Ensemble)",
            "core/models/train_stacker.py"
        )
        step_results.append(("Meta-Stacker", success, t))

        # 3.2 Modelo de Cierre de Línea y CLV
        success, t = run_step(
            "3.2 Modelo de Rastreo de CLV (Closing Line Value)",
            "core/market_models/train_clv_model.py",
            allow_fail=True
        )
        step_results.append(("CLV Tracker", success, t))

    # =========================================================================
    # FASE 4: VALIDACIÓN FINANCIERA INSTITUCIONAL Y CONTROL DE CALIDAD
    # =========================================================================
    console.print("\n[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]📈 FASE 4: VALIDACIÓN FINANCIERA GLOBAL Y PRUEBAS DEL MOTOR CLI[/bold magenta]")
    console.print("[bold magenta]════════════════════════════════════════════════════════════════════════════════[/bold magenta]")

    # 4.1 Simulación Macro-Financiera de Bankroll con Optimización Markowitz/Kelly
    success, t = run_step(
        "4.1 Simulación Macro-Financiera de Bankroll (Markowitz Optimizer)",
        "core/simulate_bankroll.py",
        allow_fail=True
    )
    step_results.append(("Simulación Bankroll Macro", success, t))

    # 4.2 Suite de Pruebas Cuantitativa de Liga MX (Partidos Recientes 2026)
    success, t = run_step(
        "4.2 Validación en Liga MX (Partidos Oficiales 2026)",
        "tests/test_cli_predictor_liga_mx.py"
    )
    step_results.append(("Test CLI Liga MX", success, t))

    # 4.3 Suite de Pruebas Multi-Liga Out-Of-Sample (Partidos Reales 2026)
    success, t = run_step(
        "4.3 Validación Multi-Liga Out-Of-Sample (Partidos Reales 2026)",
        "tests/test_cli_predictor_real_data.py"
    )
    step_results.append(("Test CLI Multi-Liga", success, t))

    total_pipeline_time = time.time() - start_time_all

    # =========================================================================
    # RESUMEN FINAL Y REPORTE EJECUTIVO
    # =========================================================================
    console.print("\n[bold bright_green]════════════════════════════════════════════════════════════════════════════════[/bold bright_green]")
    console.print(f"[bold bright_green]✨ PIPELINE MENSUAL COMPLETADO CON ÉXITO EN {total_pipeline_time/60:.2f} MINUTOS[/bold bright_green]")
    console.print("[bold bright_green]════════════════════════════════════════════════════════════════════════════════[/bold bright_green]")

    table = Table(title="Resumen de Ejecución del Pipeline Mensual", show_header=True, header_style="bold white on blue", border_style="cyan")
    table.add_column("Fase / Script", style="bold white")
    table.add_column("Estado", justify="center")
    table.add_column("Tiempo (s)", justify="right")

    for name, success, t in step_results:
        status_str = "[bold green]ÉXITO[/bold green]" if success else "[bold yellow]OMITIDO/AVISO[/bold yellow]"
        table.add_row(name, status_str, f"{t:.2f}s")

    console.print(table)

    log_file_today = f"pipeline_monthly_update_{datetime.now().strftime('%Y-%m-%d')}.log"
    console.print(f"\n[cyan]📄 Registro completo de logs guardado en:[/cyan] [bold white]logs/{log_file_today}[/bold white]")
    console.print("[bold bright_yellow]🚀 El sistema y los modelos del CLI Predictor se encuentran actualizados y listos para operar.[/bold bright_yellow]\n")

    logger.info("================================================================================")
    logger.info(f"PIPELINE MENSUAL FINALIZADO EXITOSAMENTE ({datetime.now()}) - Duración: {total_pipeline_time:.2f}s")
    logger.info("================================================================================")

if __name__ == '__main__':
    main()
