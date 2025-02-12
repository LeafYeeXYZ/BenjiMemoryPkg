from typing import Callable
from .types import Memo
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai import OpenAI
import json


def update_summary(
    *,
    model: OpenAI,
    model_name: str,
    new_messages: list[ChatCompletionMessageParam],
    old_summary: str | None,
    user_info: str | None,
) -> str:
    '''
    ## 生成/更新当前对话总结
    '''
    if old_summary is None:
        old_summary = '(这是第一轮对话, 没有总结)'
    if user_info is None:
        user_info = '(这是第一次咨询, 你没有对来访者的记忆)'

    messages: list[str] = []
    for item in new_messages:
        if item['role'] == 'tool':
            messages.append(f'- 系统 (记忆调用结果): {item["content"]}')
        elif item['role'] == 'assistant':
            if 'tool_calls' in item:
                tool_call = list(item['tool_calls'])[0]
                description: str = json.loads(tool_call['function']['arguments'])['memoryDescription']
                if type(description) != str:
                    raise ValueError('通过 json.loads 解析出的 memoryDescription 不是字符串')
                messages.append(f'- 咨询师 (进行记忆调用): 调用和"{description}"相关的记忆')
            elif 'content' in item:
                messages.append(f'- 咨询师: {item["content"]}')
            else:
                raise ValueError('assistant 类型的消息中没有 content 或 tool_calls 字段')
        elif item['role'] == 'user':
            messages.append(f'- 来访者: {item["content"]}')
        else:
            raise ValueError('输入的消息中包含了除 "tool", "assistant", "user" 之外的角色')
    messages_str = '\n'.join(messages)

    prompt = f'你是一个心理咨询师, 在你与来访者的一次咨询中, 你需要不断根据最新的内容来生成/更新对本次咨询的总结. 你将收到此前你对已有内容的总结、新增的对话内容、你对用户的记忆. 你的目标是参考新的对话内容, 在原有总结中纳入新的内容, 同时确保总结长度不超过 20 个句子, 最后输出新的总结. 为了更好地更新总结, 请按照以下步骤操作: \n\n1. 仔细分析已有的总结, 从中提取出已有的信息和事实. \n2. 考虑新增的对话内容, 找出需要纳入总结的任何新的或已改变的信息. \n3. 结合新旧信息, 创建最新的总结. \n4. 以简洁明了的方式组织更新的总结, 确保不超过 20 句话. \n5. 注意信息的相关性和重要性, 重点抓住最重要的方面, 同时保持总结的整体连贯性. \n\n此外, 请不要把你和来访者的名字包含在总结中, 用"咨询师"代表自己, 用"来访者"代表来访者即可; 你的输出应只包含总结, 请不要额外输出任何其他内容.\n\n# 已有总结\n\n{old_summary}\n\n# 新增对话内容\n\n{messages_str}\n\n# 你对来访者的记忆\n\n{user_info}'

    response = model.chat.completions.create(
        model=model_name,
        messages=[
            { 'role': 'user', 'content': prompt }
        ],
        stream=False,
    )
    
    new_summary = response.choices[0].message.content
    if new_summary is None:
        raise ValueError('response.choices[0].message.content is None')
    
    return new_summary


def similarity(
    vec1: list[float], 
    vec2: list[float], 
    /
) -> float:
    '''
    ## 计算两个向量的余弦相似度
    '''
    if len(vec1) != len(vec2):
        raise ValueError('两个向量的维度不一致')
    
    dot_product = sum(val1 * val2 for val1, val2 in zip(vec1, vec2))
    magnitude1 = sum(val ** 2 for val in vec1) ** 0.5
    magnitude2 = sum(val ** 2 for val in vec2) ** 0.5

    return dot_product / (magnitude1 * magnitude2)


def get_memory(
    *,
    embedding_api: Callable[[str], list[float]],
    existing_memo: list[Memo],
    memo_description: str,
) -> list[Memo]:
    '''
    ## 在记忆库中提取记忆
    '''
    memo_vector = embedding_api(memo_description)

    similarities = [
        (memo, similarity(memo.vector, memo_vector))
        for memo in existing_memo
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    high_similarities = [memo for memo, sim in similarities if sim >= 0.7]

    if len(high_similarities) > 0 and len(high_similarities) <= 3:
        return high_similarities
    elif len(similarities) <= 3:
        return [memo for memo, _ in similarities]
    else:
        return [memo for memo, _ in similarities[:3]]
    

def get_reduce(
    *,
    old_reduce: int,
    token_usage: int,
    token_limit: int,
) -> int:
    '''
    ## 计算调节因子变化量
    '''
    pressure = token_usage / token_limit

    if pressure >= 0.9:
        return old_reduce + 4
    elif pressure >= 0.85:
        return old_reduce + 3
    elif pressure >= 0.8:
        return old_reduce + 2
    elif pressure >= 0.75:
        return old_reduce + 1
    elif pressure >= 0.7:
        return old_reduce
    elif pressure >= 0.65:
        return max(old_reduce - 1, 0)
    elif pressure >= 0.6:
        return max(old_reduce - 2, 0)
    elif pressure >= 0.55:
        return max(old_reduce - 3, 0)
    else:
        return max(old_reduce - 4, 0)


get_memory_tool = ChatCompletionToolParam(
    type='function',
    function={
        'name': 'get_memory',
        'description': '在记忆库中提取记忆',
        'parameters': {
            'type': 'object',
            'properties': {
                'memoryDescription': {
                    'type': 'string',
                    'description': '对要提取的记忆的描述'
                },
            },
            'required': [
                'memoryDescription'
            ],
            'additionalProperties': False
        },
        'strict': True # type: ignore (see https://platform.openai.com/docs/guides/function-calling#defining-functions)
    }
)
