import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from ChatGPT import Chat
from openai_schema import SettingsRequest, SettingsResponse, ChatRequest, ChatResponse
from utils.log_utils import setup_logger, get_loger


def init_app():
    logs = f"Service started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    gpt_config_path = 'llm_conf.json'
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


def rename_file(ori_dir, ori_file):
    ori_log_path = os.path.join(ori_dir, ori_file)
    if os.path.exists(ori_log_path):
        # 获取文件的创建时间
        creation_time = os.path.getctime(ori_log_path)
        # 将创建时间格式化为字符串
        formatted_time = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
        # 构建新的文件路径
        new_log_path = os.path.join(ori_dir, f'{formatted_time}_{ori_file}')
        os.rename(ori_log_path, new_log_path)
        ori_log_path = new_log_path
    return ori_log_path


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str):
        super().__init__(app)
        self.required_credentials = secret_key

    async def dispatch(self, request: StarletteRequest, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization and authorization.startswith('Bearer '):
            provided_credentials = authorization.split(' ')[1]
            # 比较提供的令牌和所需的令牌
            if provided_credentials == self.required_credentials:
                return await call_next(request)
        # 返回一个带有自定义消息的JSON响应
        return JSONResponse(
            status_code=400,
            content={"detail": "Unauthorized: Invalid or missing credentials"},
            headers={'WWW-Authenticate': 'Bearer realm="Secure Area"'}
        )


log_path = os.path.join('logs', 'api.log')
# 创建一个日志器
setup_logger(log_type="console_file", log_file=log_path)
chat_logger = get_loger()
# 创建一个线程池
executor = ThreadPoolExecutor(max_workers=10)
chat_app = FastAPI()
secret_key = os.getenv('CHAT-API-SECRET-KEY', 'sk-chat-api')
chat_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'], )
chat_app.add_middleware(BasicAuthMiddleware, secret_key=secret_key)


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
        <html> <head> <title>chat_service_api</title> </head>
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


@chat_app.get("/monitor")
def fetch_system_status():
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


@chat_app.get("/status")
async def get_system_status():
    return FileResponse('status.html')


@chat_app.post("/v1/settings")
@chat_app.post("/v1/chat/settings")
async def chat_settings(request: SettingsRequest):
    try:
        sno = request.sno
        logs = f"Settings request param: {request.model_dump()}"
        chat_logger.info(logs)
        code, messages = await chat.set(request.model_dump())
        results = SettingsResponse(
            sno=sno,
            code=code,
            messages=messages
        )
        logs = f"Settings response results: {results.model_dump()}"
        chat_logger.info(logs)
        # return results
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = ChatResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Completions response  error: {error_message.model_dump()}"
        chat_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except ValueError as ve:
        error_message = ChatResponse(
            code=-1,
            messages=f"ValueError, Invalid value encountered: {str(ve)}"
        )
        logs = f"Completions response error: {error_message.model_dump()}"
        chat_logger.error(logs)
        return JSONResponse(status_code=422, content=error_message.model_dump())
    # except Exception as e:
    #     sno = request.sno
    #     error_message = SettingsResponse(
    #         sno=sno,
    #         code=-1,
    #         messages=str(e)
    #     )
    #     logs = f"Settings response error: {error_message.model_dump()}"
    #     chat_logger.error(logs)
    #     return JSONResponse(status_code=500, content=error_message.model_dump())


@chat_app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        sno = request.sno
        logs = f"Completions request param: {request.model_dump()}"
        chat_logger.info(logs)
        code, messages, answers = await chat.completions(request.model_dump())
        results = ChatResponse(
            code=code,
            sno=sno,
            messages=messages,
            data=answers,
            answers=answers
        )
        logs = f"Completions response results: {results.model_dump()}"
        chat_logger.info(logs)
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = ChatResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Completions response  error: {error_message.model_dump()}"
        chat_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except ValueError as ve:
        error_message = ChatResponse(
            code=-1,
            messages=f"ValueError, Invalid value encountered: {str(ve)}"
        )
        logs = f"Completions response error: {error_message.model_dump()}"
        chat_logger.error(logs)
        return JSONResponse(status_code=422, content=error_message.model_dump())
    # except Exception as e:
    #     error_message = ChatResponse(
    #         code=-1,
    #         messages=f"Exception, {str(e)}"
    #     )
    #     logs = f"Completions response error: {error_message.model_dump()}"
    #     chat_logger.error(logs)
    #     return JSONResponse(status_code=500, content=error_message.model_dump())


if __name__ == '__main__':
    chat = init_app()
    # uvicorn.run(chat_app, host='0.0.0.0', port=8090, workers=2, limit_concurrency=4, limit_max_requests=100)
    uvicorn.run(chat_app, host='0.0.0.0', port=8090)
