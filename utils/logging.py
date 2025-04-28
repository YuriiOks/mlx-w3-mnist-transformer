# MNIST Digit Classifier (Transformer)
# File: utils/logging.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Logging setup for the project with colored console output.
# Created: 2025-04-28
# Updated: 2025-04-28 # <-- Update date

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
import multiprocessing

# --- 👇 Import coloredlogs ---
try:
    import coloredlogs
    COLOREDLOGS_AVAILABLE = True
except ImportError:
    COLOREDLOGS_AVAILABLE = False
    print("⚠️ 'coloredlogs' library not found.")
    print("   Console output will not be colored.")
    print("   Install with: pip install coloredlogs")
# --- End import ---


# --- Config ---
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FILE_ENABLED = os.environ.get(
    'LOG_FILE_ENABLED', 'true'
).lower() in ('true', 'yes', '1')
LOG_CONSOLE_ENABLED = os.environ.get(
    'LOG_CONSOLE_ENABLED', 'true'
).lower() in ('true', 'yes', '1')
LOGS_DIR = os.environ.get('LOGS_DIR', 'logs')
LOG_FILE_NAME = os.environ.get(
    'LOG_FILE_NAME', 'mnist_vit_train.log'
) # Project-specific name
LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10*1024*1024)) # 10MB
LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
LOG_FORMAT = os.environ.get(
    'LOG_FORMAT',
    # Adjust format slightly for better alignment with coloredlogs levels
    '%(asctime)s | %(name)s | %(levelname)-8s | '
    '[%(filename)s:%(lineno)d] | %(message)s'
)
# Example coloredlogs field styles (optional customization)
FIELD_STYLES = coloredlogs.DEFAULT_FIELD_STYLES
FIELD_STYLES['levelname'] = {'color': 'white', 'bold': True}
FIELD_STYLES['name'] = {'color': 'blue'}
# Example coloredlogs level styles (can customize INFO, WARNING, ERROR etc.)
LEVEL_STYLES = coloredlogs.DEFAULT_LEVEL_STYLES
LEVEL_STYLES['info'] = {'color': 'green'}
LEVEL_STYLES['warning'] = {'color': 'yellow'}
LEVEL_STYLES['error'] = {'color': 'red', 'bold': True}
LEVEL_STYLES['critical'] = {
    'color': 'red', 'bold': True, 'background': 'white'
}


DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOGGER_NAME = "Backprop Bunch" # Your team/project name
# --- End Config ---


logger = logging.getLogger(LOGGER_NAME)
_logging_initialized = False

def setup_logging(log_dir=LOGS_DIR, log_file=LOG_FILE_NAME):
    """Configures the project logger with colored console output."""
    global _logging_initialized
    if _logging_initialized: return

    print(f"⚙️  Configuring {LOGGER_NAME} logging...")
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    # Prevent adding handlers multiple times if called again somehow
    if logger.hasHandlers():
        print("  Clearing existing handlers...")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    print(f"  Logger '{LOGGER_NAME}' level set to: {LOG_LEVEL}")

    # --- File Handler (No Color) ---
    if LOG_FILE_ENABLED:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            # Use standard formatter for the file
            file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            fh = RotatingFileHandler(
                log_path,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            fh.setLevel(level)
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
            print(f"  ✅ File handler added: {log_path}")
        except Exception as e:
            print(f"  ❌ ERROR setting up file log: {e}")

    # --- Console Handler (WITH Color if available) ---
    if LOG_CONSOLE_ENABLED:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)

        if COLOREDLOGS_AVAILABLE:
             # Use ColoredFormatter for console
             console_formatter = coloredlogs.ColoredFormatter(
                 fmt=LOG_FORMAT,
                 datefmt=DATE_FORMAT,
                 level_styles=LEVEL_STYLES,
                 field_styles=FIELD_STYLES
             )
             print("  🎨 Applying colored formatter to console handler.")
        else:
             # Fallback to standard formatter if coloredlogs not installed
             console_formatter = logging.Formatter(
                 LOG_FORMAT, datefmt=DATE_FORMAT
             )
             print("  Falling back to standard console formatter.")

        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
        print("  ✅ Console handler added.")


    if logger.hasHandlers():
        logger.info("🎉 Logging system initialized!")
    else:
        print(f"⚠️ Warning: No handlers configured for {LOGGER_NAME}.")
    _logging_initialized = True

# Setup logging only if this is the main process and not initialized.
if (multiprocessing.current_process().name == 'MainProcess' and
        not _logging_initialized):
    setup_logging()

# Optional: Keep the direct run check if you want to test logging.py
if __name__ == "__main__":
     if multiprocessing.current_process().name == 'MainProcess':
          logger.info("Logging module test (MainProcess). INFO")
          logger.warning("Logging module test (MainProcess). WARNING")
          logger.error("Logging module test (MainProcess). ERROR")
     else:
          # Worker processes might still hit this if run directly
          proc_name = multiprocessing.current_process().name
          print(f"Logging module test (Worker: {proc_name}). "
                "Logger setup skipped.")