FROM base/api:python3.10-torch2.3.1

# 设置环境变量
ENV all_proxy=http://192.168.0.64:7890
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
# 设置访问密钥
ENV CHAT-API-SECRET-KEY=sk-chat-api

# 映射端口
EXPOSE 8090

# 设置工作目录
WORKDIR /app/Chat-API

# 克隆项目
RUN cd /app/ && git clone https://github.com/zxd147/ChatAPI Chat-API


# 安装依赖
RUN mv gpt_conf.json.example gpt_conf.json && mv user_info.json.example user_info.json && pip install -r requirements.txt

# 容器启动时默认执行的命令
CMD ["python", "chat_api.py"]