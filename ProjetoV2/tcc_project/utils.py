import os
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configura logger com formatação colorida para terminal."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def ensure_output_dir(subdir: str = "") -> Path:
    """Garante que o diretório de saída existe e retorna o caminho."""
    base_dir = Path("outputs")
    if subdir:
        output_dir = base_dir / subdir
    else:
        output_dir = base_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_timestamp() -> str:
    """Retorna timestamp para nomes de arquivo."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_step(logger: logging.Logger, step: str, details: str = ""):
    """Log formatado para passos do processo."""
    msg = f"[STEP] {step}"
    if details:
        msg += f" - {details}"
    logger.info(msg)


def log_progress(logger: logging.Logger, current: int, total: int, item: str = "items"):
    """Log de progresso."""
    percentage = (current / total) * 100
    logger.info(f"[PROGRESS] {current}/{total} {item} ({percentage:.1f}%)")
