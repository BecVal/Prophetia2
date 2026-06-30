import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name):
    logger.info(f"=== INICIANDO {script_name} ===")
    result = subprocess.run(['python', script_name], capture_output=False)
    if result.returncode != 0:
        logger.error(f"Error al ejecutar {script_name}. Deteniendo pipeline.")
        exit(1)
    logger.info(f"=== {script_name} COMPLETADO ===\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    scripts = [
        "models/train_poisson.py",
        "models/train_context.py",
        "models/train_nn.py",
        "models/train_draws.py",
        "models/train_market.py",
        "models/train_stacker.py",
        "train_clv_model.py",
    ]
    
    for script in scripts:
        run_script(script)
        
    logger.info("Pipeline de Stacking completado exitosamente.")
    logger.info("Puedes ejecutar 'python core/simulate_bankroll.py' para probar los resultados financieros.")
