import os
import subprocess

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'run_pipeline')


def run_script(script_name, allow_fail=False):
    logger.info(f"=== INICIANDO {script_name} ===")
    result = subprocess.run(['python', script_name], capture_output=False)
    if result.returncode != 0:
        logger.error(f"Error al ejecutar {script_name}.")
        if not allow_fail:
            logger.error("Deteniendo pipeline.")
            exit(1)
        else:
            logger.warning("Continuando pipeline ya que este script es opcional (plan B disponible)...")
    else:
        logger.info(f"=== {script_name} COMPLETADO ===\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run quant model first, allow it to fail (Plan A)
    run_script("models/train_quant_advanced.py", allow_fail=True)
    
    scripts = [
        "models/train_poisson.py", # Plan B
        "models/train_context.py",
        "models/train_nn.py",
        "models/train_draws.py",
        "models/train_market.py",
        "models/train_gbm_model.py",
        "models/train_corners_model.py",
        "models/train_shots_on_goal.py",
        "models/train_cards_model.py",
        "models/train_stacker.py",
        "train_clv_model.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    logger.info("Pipeline de Stacking completado exitosamente.")
    logger.info("Puedes ejecutar 'python core/simulate_bankroll.py' para probar los resultados financieros.")
