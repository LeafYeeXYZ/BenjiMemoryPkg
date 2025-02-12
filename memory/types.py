from dataclasses import dataclass
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


@dataclass
class Memo:
    vector: list[float]
    content: str
    session_id: str


@dataclass
class ChatResult:
    output_messages: list[ChatCompletionMessageParam]
    new_reduce_num: int
    new_summary: str
    new_memo_calls: list[str]
