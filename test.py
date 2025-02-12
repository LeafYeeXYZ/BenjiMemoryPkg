# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "einops",
#     "openai",
#     "torch",
#     "transformers",
# ]
#
# [[tool.uv.index]]
# url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
# default = true
# ///


from memory import chat, update_user_info
from openai import OpenAI
from transformers import AutoModel # type: ignore
from threading import Lock
from typing import Any


model = OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ollama')
model_name = 'qwen2.5:7b'


class Embedding:
    '''
    ## 文本嵌入向量, 具有内置的锁机制
    '''
    model: Any
    lock: Lock

    def __init__(self):
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True) # type: ignore
        self.lock = Lock()

    def embedding_api(self, text: str) -> list[float]:
        embedding: list[float] = []
        self.lock.acquire()
        try:
            embeddings = self.model.encode(
                [text],
                task='text-matching',
            )
            embedding = embeddings[0].tolist()
        finally:
            self.lock.release()

        return embedding
    

emb = Embedding()


def test_embedding():
    '''
    ## 测试文本嵌入向量

    embedding: [0.07485110312700272, -0.08314336836338043, 0.10287508368492126] ...
    '''
    result = emb.embedding_api('来访者关于失恋的故事')
    print('embedding:', result[:3], '...')


def test_update_user_info():
    '''
    ## 测试更新用户信息

    来访者因失恋寻求心理咨询.首次会谈中,来访者表达了悲伤和迷茫的情绪,表示不确定如何向前走。咨询师建议来访者通过书写日记来表达情感,并计划下一次咨询讨论更长远的应对策略。
    '''
    result = update_user_info(
        model=model,
        model_name=model_name,
        latest_summary='来访者失恋了, 想找人聊聊.',
        old_user_info=None,
    )
    print(result)


def test_chat():
    '''
    ## 测试对话

    ChatResult(output_messages=[{'role': 'user', 'content': '早上好'}, {'role': 'assistant', 'content': '你好，早上好！今天有什么想和我聊聊的事情吗？或者你觉得最近有些事情让你感到困扰呢？有时候一杯茶的时光也能让我们更加放松地讨论问题。'}], new_reduce_num=0, new_summary='今天的初次咨询中，来访者表示早上好。咨询师回应问候并询问来访者是否有什么想聊的话题或感到困扰的事情。这次会谈营造了一个轻松的氛围，为后续交流奠定了基础。', new_memo_calls=[])
    '''
    result = chat(
        model_a=model,
        model_a_name=model_name,
        model_b=model,
        model_b_name=model_name,
        embbeding_api=emb.embedding_api,
        old_reduce_num=0,
        max_token_usage=3000,
        history_messages=[],
        user_input='早上好',
        old_summary=None,
        user_info=None,
        existing_memo=[],
        exclude_memo=[],
    )
    print(result)


if __name__ == '__main__':
    test_embedding()
    test_update_user_info()
    test_chat()
