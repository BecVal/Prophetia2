import logging
import os
from datetime import datetime

def get_logger(name, log_filename):
    """
    Configura y devuelve un logger que escribe tanto en la consola como en un archivo.
    El archivo se guarda en la carpeta 'logs' en la raíz del proyecto.
    """
    # Directorio raíz (un nivel arriba de 'core')
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(root_dir, 'logs')
    
    # Crear carpeta logs si no existe
    os.makedirs(log_dir, exist_ok=True)
    
    # Nombre del archivo con la fecha de hoy
    timestamp = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(log_dir, f"{log_filename}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Evitar handlers duplicados si el logger ya existe
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File Handler (escribe en el archivo)
    file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler (escribe en la terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
