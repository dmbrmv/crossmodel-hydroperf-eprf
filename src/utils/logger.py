"""Comprehensive logging utilities for MeteoSources application.

This module provides:
- setup_logger: Configure logger with file rotation and console output
- configure_third_party_loggers: Reduce noise from third-party libraries
- main_logger: Pre-configured logger instance for immediate use

The logger supports file rotation, customizable log levels, and formatted output
for both console and file logging. Third-party library loggers are automatically
configured to reduce verbose output.
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logger(
    name: str = "MeteoSources",
    log_level: str = "INFO",
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """Set up a comprehensive logger with file rotation and console output.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, creates 'logs/meteosources.log'
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output logs to console

    Returns:
        Configured logger instance

    Raises:
        ValueError: If invalid log level is provided

    """
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up file handler with rotation
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "meteosources.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set up console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def configure_third_party_loggers() -> None:
    """Configure third-party library loggers to reduce noise."""
    # Suppress verbose third-party loggers
    third_party_loggers = [
        "rasterio",
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "matplotlib",
        "PIL",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Suppress specific warnings
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)


# Initialize the main logger
configure_third_party_loggers()
main_logger = setup_logger()

# Export for easy import
__all__ = ["setup_logger", "configure_third_party_loggers", "main_logger"]
