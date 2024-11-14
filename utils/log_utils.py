"""
logger封装
"""
import logging
import sys

from loguru import logger

# 移除所有默认的处理器
logger.remove()

# 自定义格式并添加到标准输出
# log_format = "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"
log_format = "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl> \n{message}"

# logger.add(sys.stdout, level="INFO", format=log_format, backtrace=True, diagnose=True)


def configure_logging():
    log_file = 'logs/api.log'
    logger = logging.getLogger('whisper')
    logger.setLevel(logging.INFO)
    handel_format = '%(asctime)s - %(levelname)s - %(message)s'
    # 设置 propagate 为 False
    # propagate 用于控制日志消息的传播行为，如果设置为 True（默认值），那么该 logger 记录的消息会向上层的 logger 传播，导致记录两次日志。
    logger.propagate = False
    # 移除现有的处理器（防止重复）
    if logger.hasHandlers():
        logger.handlers.clear()
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter(handel_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)
    return logger


def setup_console_logger(level="INFO"):
    """设置控制台日志"""
    logger.remove()  # 移除默认的日志输出
    # 在日志中包含完整的异常堆栈跟踪, 增加更多的诊断信息
    logger.add(sys.stdout, level=level, format=log_format, backtrace=True,
               diagnose=True)
    return logger


def setup_file_logger(log_file="file.log", level="INFO"):
    """设置文件日志"""
    logger.remove()  # 移除默认的日志输出
    logger.add(log_file, level=level, format=log_format, backtrace=True,
               diagnose=True)
    return logger


def setup_console_and_file_logger(log_file="file.log", level="INFO"):
    """设置同时输出到控制台和文件的日志"""
    logger.remove()  # 移除默认的日志输出
    logger.add(sys.stdout, level=level, format=log_format, backtrace=True,
               diagnose=True)
    logger.add(log_file, level=level, format=log_format, backtrace=True,
               diagnose=True)
    return logger


