import logging

def get_logger(name, log_level=logging.INFO):
    """
    Configure et retourne un logger avec un format standard.
    
    Args:
        name (str): Nom du logger.
        log_level (int): Niveau du log (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Logger configuré.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Éviter d'ajouter plusieurs fois des handlers
        logger.setLevel(log_level)
        handler = logging.StreamHandler()  # Log vers la console
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
