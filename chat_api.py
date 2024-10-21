import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Union, Literal

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from ChatGPT import Chat


def init_app():
    logs = f"Service started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    gpt_config_path = 'gpt_conf.json'
    user_info_path = 'user_info.json'
    chat_instance = Chat(gpt_config_path, user_info_path)
    chat_logger.info(logs)
    chat_logger.info(chat_instance.init_messages)
    return chat_instance


def configure_logging():
    logger = logging.getLogger('chat')
    logger.setLevel(logging.INFO)
    # 创建一个流处理器并设置级别和格式
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n %(message)s')
    stream_handler.setFormatter(stream_formatter)
    # 将处理器添加到日志器
    logger.addHandler(stream_handler)
    return logger


# 创建一个日志器
chat_logger = configure_logging()
chat_app = FastAPI()
# 创建一个线程池
executor = ThreadPoolExecutor(max_workers=10)


class ChatSettingsRequest(BaseModel):
    uid: Optional[Union[int, str]] = 'admin'
    channel: Literal['FastGPT', 'Langchain'] = 'FastGPT'
    model: Literal['Qwen2-7B-Instruct', 'ChatGLM-6B'] = 'Qwen2-7B-Instruct'
    mode: Literal['knowledge_base', 'direct'] = 'knowledge_base'
    stream: bool = True
    knowledge_base: Literal['zyy', 'guangxin', 'dentistry'] = 'zyy'
    project_type: int = 1


class ChatSettingsResponse(BaseModel):
    code: int
    messages: Optional[str]
    data: Optional[List[str]] = None


class ChatCompletionRequest(BaseModel):
    sno: Optional[Union[int, str]] = Field(default_factory=lambda: int(time.time() * 100))
    uid: Optional[Union[int, str]] = 'admin'
    content: str = None
    query: str = None


class ChatCompletionResponse(BaseModel):
    code: int
    messages: Optional[str]
    sno: Optional[Union[int, str]] = None
    data: Optional[List[str]] = None
    answers: Optional[List[str]] = []


# 将 Pydantic 模型实例转换为 JSON 字符串
def dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


@chat_app.middleware("http")
async def log_requests(request: Request, call_next):
    logs = "Request arrived"
    chat_logger.debug(logs)
    response = await call_next(request)
    return response


@chat_app.get("/")
async def index():
    service_name = """
        <html> <head> <title>tts_service</title> </head>
            <body style="display: flex; justify-content: center;"> <h1> chat_service_api</h1></body> </html>
        """
    return HTMLResponse(status_code=200, content=service_name)


@chat_app.get("/health")
async def health():
    """Health check."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    health_data = {"status": "healthy", "timestamp": timestamp}
    # 返回JSON格式的响应
    return JSONResponse(status_code=200, content=health_data)


@chat_app.get("/status")
def get_status():
    """获取电脑运行状态"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_total = f"{memory_info.total / (1024 ** 3):.2f} GB"
    memory_available = f"{memory_info.available / (1024 ** 3):.2f} GB"
    memory_used = f"{memory_info.used / (1024 ** 3):.2f} GB"
    memory_percent = memory_info.percent
    gpuInfo = []
    devices = ["cpu"]
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpuInfo.append(
            {
                "gpu_id": gpu.id,
                "gpu_load": f"{gpu.load}%",
                "gpu_memory": {
                    "total_MB": f"{gpu.memoryTotal} MiB",
                    "total_GB": f"{gpu.memoryTotal / 1024:.2f} GiB",
                    "used": f"{gpu.memoryUsed} MiB",
                    "free": f"{gpu.memoryFree} MiB"
                }
            }
        )
    return {
        "devices": devices,
        "cpu_percent": cpu_percent,
        "memory_total": memory_total,
        "memory_available": memory_available,
        "memory_used": memory_used,
        "memory_percent": memory_percent,
        "gpu": gpuInfo,
    }


@chat_app.post("/v1/chat/settings")
async def chat_settings(request: ChatSettingsRequest):
    try:
        settings_data = request.model_dump()
        logs = f"Settings request param: {settings_data}"
        chat_logger.info(logs)
        code, messages = await chat.set(settings_data)
        results = ChatSettingsResponse(
            code=code,
            messages=messages
        )
        logs = f"Settings response results: {results}"
        chat_logger.info(logs)
        # return results
        return JSONResponse(status_code=200, content=results.model_dump())
    except Exception as e:
        error_message = ChatSettingsResponse(
            code=-1,
            messages=str(e)
        )
        logs = f"Settings response error: {error_message}"
        chat_logger.error(logs)
        return JSONResponse(status_code=500, content=error_message.model_dump())


@chat_app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        request_data = request.model_dump()
        sno = request_data['sno']
        logs = f"Completions request param: {request_data}"
        chat_logger.info(logs)
        code, messages, answers = await chat.completions(request_data)
        results = ChatCompletionResponse(
            code=code,
            sno=sno,
            messages=messages,
            data=answers,
            answers=answers
        )
        logs = f"Completions response results: {results}\n"
        chat_logger.info(logs)
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = ChatCompletionResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Completions response  error: {error_message}\n"
        chat_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except Exception as e:
        error_message = ChatCompletionResponse(
            code=-1,
            messages=f"Exception, {str(e)}"
        )
        logs = f"Completions response error: {error_message}\n"
        chat_logger.error(logs)
        return JSONResponse(status_code=500, content=error_message.model_dump())


if __name__ == '__main__':
    chat = init_app()
    uvicorn.run(chat_app, host='0.0.0.0', port=8090)
    # uvicorn.run(chat_app, host='0.0.0.0', port=8090, workers=2, limit_concurrency=4, limit_max_requests=100)
