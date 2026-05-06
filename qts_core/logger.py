import sys
import os
from loguru import logger

def setup_logger(log_file="logs/system.log", level="INFO"):
    """
    Configures loguru to log to stdout and a file.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Remove default handler
    logger.remove()

    # Add stdout handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler with rotation
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="10 days",
        compression="zip"
    )

    return logger

# Initialize with default settings
# Note: we don't call setup_logger() here because we want to be able to 
# configure it from run.py or other entry points if needed.
# But we can provide a default configuration.
setup_logger()
