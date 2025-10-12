"""
Logging Utility for Qrucible
Provides consistent logging across all modules
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'qrucible',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_mode: str = 'a'
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, logs to logs/qrucible.log)
        level: Logging level
        console: Whether to log to console
        file_mode: File mode ('a' for append, 'w' for overwrite)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'qrucible_{datetime.now().strftime("%Y%m%d")}.log'
    
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)