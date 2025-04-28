# MNIST Digit Classifier (Transformer)
# File: utils/logging.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Logging setup for the project.
# Created: 2025-04-28
# Updated: 2025-04-28

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
import multiprocessing

# --- Config ---
# Read from env or use defaults likely set by config.yaml loading later
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FILE_ENABLED = os.environ.get('LOG_FILE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOG_CONSOLE_ENABLED = os.environ.get('LOG_CONSOLE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOGS_DIR = os.environ.get('LOGS_DIR', 'logs')
LOG_FILE_NAME = os.environ.get('LOG_FILE_NAME', 'mnist_transformer.log') # Default name
LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10*1024*1024)) # 10MB
LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
LOG_FORMAT = os.environ.get(
    'LOG_FORMAT',
    '%(asctime)s | %(name)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s'
)
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOGGER_NAME = "Backprop Bunch" # Specific logger name

logger = logging.getLogger(LOGGER_NAME)
_logging_initialized = False

def setup_logging(log_dir=LOGS_DIR, log_file=LOG_FILE_NAME): # Allow override
    '''Configures the project-specific logger.'''
    global _logging_initialized
    if _logging_initialized: return

    print(f"⚙️  Configuring {LOGGER_NAME} logging...")
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    print(f"  Logger '{LOGGER_NAME}' level set to: {LOG_LEVEL}")

    if logger.hasHandlers():
        print("  Clearing existing handlers...")
        for handler in logger.handlers[:]: logger.removeHandler(handler); handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if LOG_CONSOLE_ENABLED:
        ch = logging.StreamHandler(sys.stdout); ch.setLevel(level)
        ch.setFormatter(formatter); logger.addHandler(ch)
        print("  ✅ Console handler added.")

    if LOG_FILE_ENABLED:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            with open(log_path, 'a', encoding='utf-8') as f: f.write("") # Check writability
            fh = RotatingFileHandler(log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8')
            fh.setLevel(level); fh.setFormatter(formatter); logger.addHandler(fh)
            print(f"  ✅ File handler added: {log_path}")
        except Exception as e: print(f"  ❌ ERROR setting up file log: {e}")

    if logger.hasHandlers(): logger.info("🎉 Logging system initialized!")
    else: print(f"⚠️ Warning: No handlers configured for {LOGGER_NAME}.")
    _logging_initialized = True

# Modify the automatic setup call at the very bottom:
if multiprocessing.current_process().name == 'MainProcess' and not _logging_initialized:
    setup_logging()

# Optional: Keep the direct run check if you want to test logging.py itself
if __name__ == "__main__":
     if multiprocessing.current_process().name == 'MainProcess':
          logger.info("Logging module test (MainProcess).")
     else:
          # Worker processes might still hit this if run directly, but it's less common
          print(f"Logging module test (Worker Process: {multiprocessing.current_process().name}). Logger setup skipped.")