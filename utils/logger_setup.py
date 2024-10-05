# logger_setup.py
from loguru import logger
import os

# Configure the logger (e.g., logging to file, setting format, etc.)
logger.add("hybrid_search_app.log", rotation="500 MB", retention="10 days", level="INFO")

# You can add more configuration here as needed

# Export the logger object for other modules to use
__all__ = ["logger"]


def log_function_call(func):
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Calling function '{func.__name__}' with arguments {args} and keyword arguments {kwargs}")
            result = func(*args, **kwargs)
            logger.info(f"Function '{func.__name__}' returned {result}")
            return result
        except Exception as e:
            logger.exception(f"An error occurred in function '{func.__name__}'")
            raise e
    return wrapper