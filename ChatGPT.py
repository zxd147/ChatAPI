import os
import re
import copy
import json
import uuid
import aiohttp
import asyncio
import logging
import requests
from datetime import datetime
from seg_sentences import split_text
from update_load_config import load_config, update_config
from utils.log_utils import logger


# # 创建一个日志器
# api_logger = logging.getLogger('GPT')
# api_logger.setLevel(logging.INFO)
# # 创建一个文件处理器并设置级别和格式
# file_handler = logging.FileHandler(log_path)
# file_handler.setLevel(logging.INFO)
# file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n %(message)s')
# file_handler.setFormatter(file_formatter)
# # # 将处理器添加到日志器
# api_logger.addHandler(file_handler)
# 创建一个信号量，限制同时运行的任务数为10
semaphore = asyncio.Semaphore(4)
api_logger = logger


class Chat:
    def __init__(self, gpt_config_path, user_info_path):
        self.gpt_config_path = gpt_config_path
        self.user_info_path = user_info_path
        self.init_messages = " * init successfully: " + self._load_gpt_conf() + ", " + self._load_user_info()

    def load_user_info(self):
        return self._load_user_info()

    def load_gpt_conf(self):
        return self._load_gpt_conf()

    async def set(self, settings_args):
        user_code, user_messages = self.update_user_info(settings_args)
        gpt_code, gpt_messages = self.update_gpt_conf()
        messages = user_messages + ", " + gpt_messages
        code = user_code or gpt_code
        return code, messages

    async def completions(self, completions_args):
        uid = completions_args['uid']
        query = completions_args['query'] or completions_args['content']
        if uid not in self.user_info['all_user_info']:
            code = 1
            messages = f"User UID: {uid}, Please upload your settings before the conversation"
            answer = ''
        else:
            if uid != self.current_user_id:
                self.current_user_info = self.user_info['all_user_info'][completions_args['uid']]
                self.update_gpt_conf()
                self.update_user_info(self.current_user_info)
            if query in ["/reset", "清空对话", "清空对话。"]:
                code = 0
                messages = self.clear_history()
                answers = ['/reset']
                return code, messages, answers
            if self.current_chat_channel == "FastGPT":
                code, messages, answer = await self.chat_with_fastgpt(completions_args)
            elif self.current_chat_channel == "GraphRAG":
                code, messages, answer = await self.chat_with_graphrag(completions_args)
            # elif self.current_chat_channel == "LangChain":
            else:
                code, messages, answer = self.chat_with_langchain(completions_args)
        answers = split_text(answer, self.min_tokens, self.max_tokens, sentences=[])
        return code, messages, answers

    async def chat_with_fastgpt(self, completions_args):
        # 等待获取响应，并立即返回Response对象
        sno = completions_args['sno']
        uid = completions_args['uid']
        messages = completions_args['messages']
        query = completions_args['query'] or completions_args['content']
        chat_url = self.mode_conf['chat_url']
        headers = copy.deepcopy(self.mode_conf['headers'])
        param = copy.deepcopy(self.mode_conf['param'])
        model = self.current_user_info['model']
        stream = self.current_user_info['stream']
        conversation_id = self.conversation_id
        if messages:
            param.pop("chatId", None)
            if len(messages) > 1:
                messages = self.history + messages
        elif query:
            param.update({"chatId": conversation_id})
            messages = [{"role": "user", "content": query}]
        else:
            # 在没有 messages 和 query 时执行其他处理
            raise ValueError("No messages or query provided in the request.")
        param['model'] = model
        param['stream'] = stream
        param['variables']['uid'] = uid
        param['variables']['name'] = sno
        param['messages'] = messages
        logs = f"FastGPT request param: \n{json.dumps(param, ensure_ascii=False, indent=None)}\n"
        api_logger.debug(logs)
        answer = ''
        response_data = ''
        async with semaphore:  # 整个函数的执行都受到信号量的控制
            async with aiohttp.ClientSession() as session:
                # 准备参数并发起请求
                async with session.post(chat_url, headers=headers, data=json.dumps(param), timeout=10) as response:
                    if response.status == 200:
                        code = 0
                        messages = 'FastGPT response session successfully'
                        if response.content_type == 'application/json':
                            response_data = await response.json()  # 如果是 JSON，直接解析
                            answer = response_data['choices'][0]['message'].get('content', '')
                            if isinstance(answer, list):
                                answer = next(
                                    (item['text']['content'] for item in answer if item.get('type') == 'text'), answer)
                        elif response.content_type == 'text/event-stream':
                            encoding = response.charset
                            async for line in response.content:
                                json_string = line.decode(encoding).strip().replace('data: ', '')
                                response_data += json_string + '\n'
                                if json_string == "[DONE]":
                                    continue
                                if json_string:  # 检查内容是否为空
                                    try:
                                        # 尝试解析为 JSON 对象
                                        data = json.loads(json_string)
                                        # 提取content
                                        choice = data['choices'][0]
                                        content = choice['delta'].get('content', '')
                                        if content:  # 如果content不为空
                                            answer += content  # 添加到最终内容中
                                    except json.JSONDecodeError:
                                        code = -1
                                        messages = f"{messages}, JSONDecodeError, FastGPT Data Invalid JSON: {json_string}. "
                                        api_logger.error(messages)
                        else:
                            code = -1
                            messages = f"{messages}, Unknown response.content_type: {response.content_type}"
                    else:
                        code = -1
                        messages = f'FastGPT response failed with status code: {response.status}. '
        if answer.startswith("0:") or answer.startswith("1:"):
            answer = answer[2:].strip()  # 去除前两个字符
        if answer != '':
            logs = f'{messages}, response_data: ---\n{response_data}\n---'
            api_logger.debug(logs)
            user_record = {"role": "user", "content": query}
            assistant_record = {"role": "assistant", "content": answer}
            self.add_history(user_record)
            self.add_history(assistant_record)
            self.write_history(user_record)
            self.write_history(assistant_record)
        else:
            if code != -1:
                code = -1
                if response_data:
                    messages = f"{messages}, ChatGPT response text is empty, response_data: ---\n{response_data}\n---"
            api_logger.error(messages)
            print(messages)
        return code, messages, answer

    async def chat_with_graphrag(self, completions_args):
        query = completions_args['query'] or completions_args['content']
        chat_url = self.mode_conf['chat_url']
        headers = copy.deepcopy(self.mode_conf['headers'])
        param = copy.deepcopy(self.mode_conf['param'])
        param.update(self.current_user_info)
        messages = [{"role": "user", "content": query}]
        param['messages'] = self.history + messages
        logs = f"GraphRAG request param: \n{json.dumps(param, ensure_ascii=False, indent=None)}\n"
        api_logger.debug(logs)
        answer = ''
        response_data = ''
        async with semaphore:  # 整个函数的执行都受到信号量的控制
            async with aiohttp.ClientSession() as session:
                # 准备参数并发起请求
                async with session.post(chat_url, headers=headers, data=json.dumps(param), timeout=10) as response:
                    if response.status == 200:
                        code = 0
                        messages = 'GraphRAG response session successfully'
                        if response.content_type == 'application/json':
                            response_data = await response.json()  # 如果是 JSON，直接解析
                            answer = response_data['choices'][0]['message'].get('content', '')
                            if isinstance(answer, list):
                                answer = next(
                                    (item['text']['content'] for item in answer if item.get('type') == 'text'), answer)
                        elif response.content_type == 'text/event-stream':
                            encoding = response.charset
                            async for line in response.content:
                                json_string = line.decode(encoding).strip().replace('data: ', '')
                                response_data += json_string + '\n'
                                if json_string == "[DONE]":
                                    continue
                                if json_string:  # 检查内容是否为空
                                    try:
                                        # 尝试解析为 JSON 对象
                                        data = json.loads(json_string)
                                        # 提取content
                                        choice = data['choices'][0]
                                        content = choice['delta'].get('content', '')
                                        if content:  # 如果content不为空
                                            answer += content  # 添加到最终内容中
                                    except json.JSONDecodeError:
                                        code = -1
                                        messages = f"{messages}, JSONDecodeError, GraphRAG Data Invalid JSON: {json_string}. "
                                        api_logger.error(messages)
                        else:
                            code = -1
                            messages = f"{messages}, Unknown response.content_type: {response.content_type}"
                    else:
                        code = -1
                        messages = f'GraphRAG response failed with status code: {response.status}. '
        if answer.startswith("0:") or answer.startswith("1:"):
            answer = answer[2:].strip()  # 去除前两个字符
        # 定义正则表达式
        # pattern = r'\[Data: Sources \(.*?\); Entities \(.*?\)\]'
        pattern = r'\[Data: (Entities|Sources)[^\]]*\]'  # 使用正则表达式去掉匹配的数据
        # 使用正则表达式去掉形如[Data: Sources (0); Entities (1)], [Data: Entities（16、88、75、83、78、46、82、81、77）]']的数据
        answer = re.sub(pattern, '', answer)
        if answer != '':
            logs = f'{messages}, response_data: ---\n{response_data}\n---'
            api_logger.debug(logs)
            user_record = {"role": "user", "content": query}
            assistant_record = {"role": "assistant", "content": answer}
            self.add_history(user_record)
            self.add_history(assistant_record)
            self.write_history(user_record)
            self.write_history(assistant_record)
        else:
            if code != -1:
                code = -1
                if response_data:
                    messages = f"{messages}, ChatGPT response text is empty, response_data: ---\n{response_data}\n---"
            api_logger.error(messages)
            print(messages)
        return code, messages, answer

    async def chat_with_langchain(self, completions_args):
        query = completions_args['query'] or completions_args['content']
        chat_url = self.mode_conf['chat_url']
        headers = copy.deepcopy(self.mode_conf['headers'])
        param = copy.deepcopy(self.mode_conf['param'])
        param['query'] = f'{query} \n回复需要清晰且高度概括'
        param['conversation_id'] = self.conversation_id
        param['history'] = self.history
        param['history_len'] = self.history_len
        param['knowledge_base_name'] = self.current_user_info['knowledge_base']
        logs = f"LangChain Request param: \n{json.dumps(param, ensure_ascii=False, indent=None)}\n"
        api_logger.debug(logs)
        answer = ''
        response_data = ''
        response = requests.post(chat_url, headers=headers, data=json.dumps(param))
        # 检查请求是否成功
        if response.status_code == 200:
            result = response.json()
            api_logger.debug(result)
            code = 0
            messages = 'ChatGPT Response session successfully'
            answer = result["answer"] if self.current_chat_mode == "knowledge_base" else result["text"]
            user_record = {"role": "user", "content": query}
            assistant_record = {"role": "assistant", "content": answer}
            self.add_history(user_record)
            self.add_history(assistant_record)
            self.write_history(user_record)
            self.write_history(assistant_record)
        else:
            code = 1
            messages = f'ChatGPT Response failed with status code {response.status_code}, \n{response.text}'
            answer = ''
            api_logger.error(messages)
            print(messages)
        return code, messages, answer

    def load_history(self):
        self.history_dir = f'history/{self.current_user_id}'
        self.history_file = f'{self.history_dir}/{self.conversation_id}.log'
        os.makedirs(self.history_dir, exist_ok=True)
        try:
            with open(self.history_file, mode='r', encoding='utf-8') as file:
                history_text = file.read()
                timestamp_pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\n'  # 去除时间戳部分
                # 使用正则表达式去掉时间戳
                history_text = re.sub(timestamp_pattern, '', history_text)
                # 加上方括号
                history_text = '[' + history_text + ']'  # 解析 JSON 数据
                self.history = json.loads(history_text)
                if len(self.history) > self.history_len:
                    # self.history = self.history[0] + self.history[-self.history_len:]
                    self.history = self.history[-self.history_len:]
        except FileNotFoundError:
            self.history = []  # 如果文件不存在，初始化为空

    def add_history(self, record):
        self.history.append(record)
        if len(self.history) > self.history_len:
            # self.history = self.history[0] + self.history[-self.history_len:]
            self.history = self.history[-self.history_len:]

    def write_history(self, record):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(self.history_file, mode='a', encoding='utf-8') as file:
            if file.tell() != 0:
                # 如果文件不为空，先写入一个逗号
                file.write(", \n")
            file.write(f"{timestamp}\n")
            # 将历史记录转换为 JSON 格式字符串并写入文件
            json.dump(record, file, ensure_ascii=False, indent=4)

    def clear_history(self):
        self.conversation_id = str(uuid.uuid4().hex[:8])
        self.user_info["all_user_info"][self.current_user_id]["conversation_id"].append(self.conversation_id)
        self.load_history()
        messages = "You have successfully reset the conversation"
        # 更新配置文件
        update_config(self.user_info_path, self.user_info)
        # 加载配置文件
        self.load_user_info()
        return messages

    def update_user_info(self, settings_args):
        uid = settings_args['uid']
        # 更新当前用户ID
        if uid not in self.user_info['all_user_info']:
            conversation_id = [str(uuid.uuid4().hex[:8])]
        else:
            conversation_id = self.user_info['all_user_info'][uid]['conversation_id']
        # 更新用户信息
        self.user_info['all_user_info'][uid] = {
            'uid': uid,  # 设置默认聊天模式
            'channel': settings_args['channel'],  # 设置默认聊天模式
            'model': settings_args['model'],
            'mode': settings_args['mode'],  # 设置默认聊天模式
            'stream': settings_args['stream'],  # 设置流式
            'conversation_id': conversation_id,  # 初始化对话列表
            'knowledge_base': settings_args['knowledge_base'],
            'project_type': settings_args['project_type'],
        }
        self.user_info['current_user_id'] = uid
        self.llm_list = self.gpt_config['channel'][settings_args['channel']]['llm_list']
        if settings_args['channel'] not in ["FastGPT", "GraphRAG", "LangChain"]:
            code = 1
            messages = "Unsupported chat mode. Please use 'FastGPT', 'GraphRAG' or 'LangChain'."
            return code, messages
        elif settings_args['mode'] not in ["direct", "knowledge_base", "local", "global", "full"]:
            code = 1
            messages = "Unsupported chat channel. Please use 'direct', 'knowledge_base', 'local', 'global' or 'full'."
            return code, messages
        elif settings_args['model'] not in self.llm_list:
            code = 1
            messages = f"GPT config update fail, {settings_args['channel']} doesn't support the LLM {settings_args['model']}, please choose in {self.llm_list}"
            return code, messages
        # 更新配置文件
        update_config(self.user_info_path, self.user_info)
        # 加载配置文件
        code = 0
        messages = self.load_user_info()
        return code, messages

    def _load_user_info(self):
        self.user_info = load_config(self.user_info_path)  # 初始化配置
        self.current_user_id = self.user_info['current_user_id']
        self.all_user_info = self.user_info['all_user_info']
        self.current_user_info = self.all_user_info[self.current_user_id]
        self.conversation_id = self.current_user_info['conversation_id'][-1]
        self.load_history()
        messages = "User info update successfully"
        return messages

    def update_gpt_conf(self):
        # 在gpt配置文件中更新对话模式
        self.current_chat_channel = self.current_user_info['channel']
        self.current_chat_llm = self.current_user_info['model']
        self.current_chat_mode = self.current_user_info['mode']
        self.gpt_config['current_chat_channel'] = self.current_chat_channel
        self.gpt_config['current_chat_llm'] = self.current_chat_llm
        self.gpt_config['current_chat_mode'] = self.current_chat_mode
        # 更新配置文件
        update_config(self.gpt_config_path, self.gpt_config)
        # 加载配置文件
        code = 0
        messages = self.load_gpt_conf()
        return code, messages

    def _load_gpt_conf(self):
        self.gpt_config = load_config(self.gpt_config_path)  # 初始化配置
        self.current_chat_channel = self.gpt_config['current_chat_channel']
        self.current_chat_llm = self.gpt_config['current_chat_llm']
        self.current_chat_mode = self.gpt_config['current_chat_mode']
        self.min_tokens = self.gpt_config['min_tokens']
        self.max_tokens = self.gpt_config['max_tokens']
        self.mode_conf = self.gpt_config['channel'][self.current_chat_channel][self.current_chat_llm][
            self.current_chat_mode]
        self.history_len = self.mode_conf['history_len'] * 2
        messages = "GPT config update successfully"
        return messages


class ConversationManager:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_history(self):
        return self.history