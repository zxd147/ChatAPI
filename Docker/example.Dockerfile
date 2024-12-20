# 基础镜像
FROM base/python:ubuntu22.04-cuda11.8-python3.10

# 设置环境变量
#ENV all_proxy=http://192.168.0.64:7890
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# 设置默认时区禁止交互
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
# 设置访问密钥
ENV CHAT-API-SECRET-KEY=sk-chat-api

# 设置工作目录
WORKDIR /app/Chat-API

# 克隆项目
RUN git clone https://github.com/zxd147/ChatAPI ../Chat-API


# 安装依赖
RUN mv llm_conf.json.example llm_conf.json \
    && mv user_info.json.example user_info.json \
    && pip install -r ./requirements.txt \
    && pip cache purge

# 映射端口
EXPOSE 8090

# 容器启动时默认执行的命令
CMD ["python", "chat_api.py"]

