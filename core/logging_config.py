"""
logging_config.py — Loglama Yapılandırması

Projede kullanılacak merkezi logging setup.
Console ve dosya çıktısı, güzel formatlanma.
"""

import logging
from pathlib import Path
from core.config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

# Logs klasörünü oluştur
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 📝 LOGGER SETUP
# ============================================================================


def setup_logger(name: str, log_file: str = None, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Logger nesnesini konfigüre et.
    
    Args:
        name: Logger adı (genellikle __name__)
        log_file: Log dosya adı (None = sadece console)
        level: Log seviyesi
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Eski handler'ları temizle (tekrar import etmeyi engelle)
    if logger.handlers:
        return logger
    
    # ──────────────────────────────────────────────────────────────────────
    # CONSOLE HANDLER (güzel formatlanmış)
    # ──────────────────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt=LOG_DATE_FORMAT
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # ──────────────────────────────────────────────────────────────────────
    # FILE HANDLER (detaylı)
    # ──────────────────────────────────────────────────────────────────────
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        
        file_format = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# 🌍 GLOBAL LOGGER
# ============================================================================

# Tüm modüller bunu import edebilir
logger = setup_logger("JPEG2000", log_file="compression.log")

# ============================================================================
# ✅ TEST
# ============================================================================

if __name__ == "__main__":
    logger = setup_logger("test_logging", log_file="test.log")
    
    logger.debug("Bu DEBUG mesajı")
    logger.info("Bu INFO mesajı")
    logger.warning("Bu WARNING mesajı")
    logger.error("Bu ERROR mesajı")
    
    print(f"\n✅ Logger hazır! Log dosyası: {LOGS_DIR / 'compression.log'}")
