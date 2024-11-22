import time
import uuid
from typing import Optional, Union, List, Dict, Literal

from pydantic import BaseModel, Field, ConfigDict


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="allow")


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(OpenAIBaseModel):
    object: str = 'list'
    data: List[ModelCard] = []


class CompletionUsage(OpenAIBaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class FunctionCallResponse(OpenAIBaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{str(uuid.uuid4().hex)}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(OpenAIBaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Optional[str]
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=lambda: [])


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal['stop', 'length', 'function_call']


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']] = None


class SettingsRequest(OpenAIBaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100))
    uid: Union[int, str]
    channel: Literal['FastGPT', 'GraphRAG', 'LangChain'] = 'FastGPT'
    model: Literal['Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct', 'ChatGLM-6B'] = 'Qwen2.5-7B-Instruct'
    mode: Literal['knowledge', 'direct', 'local', 'global', 'full'] = 'knowledge'
    knowledge_base: Literal['zyy', 'hengsha', 'guangxin', 'dentistry', 'ecology', 'test'] = 'dentistry'
    stream: bool = False
    project_type: Union[int, str] = 1


class SettingsResponse(OpenAIBaseModel):
    sno: Union[int, str] = None
    code: int
    messages: Optional[str]
    data: Optional[List[str]] = None


class ChatRequest(OpenAIBaseModel):
    model: str = 'Qwen2.5-7B-Instruct'
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100))
    uid: Optional[Union[int, str]] = 'null'
    stream: bool = None
    messages: List[dict[str, str]] = []
    content: str = ''
    query: str = ''


class ChatResponse(OpenAIBaseModel):
    sno: Optional[Union[int, str]] = None
    code: int
    messages: str
    data: Optional[List[str]] = []
    answers: Optional[List[str]] = []


class ChatCompletionRequest(OpenAIBaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100))
    uid: Optional[Union[int, str]] = 'null'
    query: str = ''

    model: Optional[str] = 'Qwen2.5-7B-Instruct'
    stream: Optional[bool] = None
    # messages: List[dict[str, str]] = []
    messages: List[ChatMessage] = []
    extra_body: Optional[dict] = {}

    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = 1
    max_completion_tokens: Optional[int] = 1024
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionResponse(OpenAIBaseModel):
    sno: Optional[Union[int, str]] = None
    code: int
    messages: str
    answers: Optional[List[str]] = []

    model: str
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal['chat.completion', 'chat.completion.chunk'] = "chat.completion.chunk"
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = Field(default=None)





