from .utils import get_memory, get_memory_tool, get_reduce, update_summary
from .types import Memo, ChatResult
from typing import Callable
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import ChatCompletionMessageToolCallParam, Function
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
import json


__all__ = [
    'chat',
    'update_user_info',
    'Memo',
    'ChatResult',
]


def chat(
    *,
    model_a: OpenAI,
    model_a_name: str = '',
    model_b: OpenAI,
    model_b_name: str = '',

    embbeding_api: Callable[[str], list[float]],

    old_reduce_num: int,
    max_token_usage: int = 3000,
    history_messages: list[ChatCompletionMessageParam],
    user_input: str,
    old_summary: str | None = None,
    user_info: str | None = None,

    existing_memo: list[Memo] | Callable[[], list[Memo]],
    exclude_memo: list[str] | Callable[[], list[str]],
) -> ChatResult:
    '''
    ## 聊天主函数
    '''
    result: ChatResult = ChatResult(
        output_messages=[
            { 'role': 'user', 'content': user_input },
        ],
        new_reduce_num=0,
        new_summary='',
        new_memo_calls=[],
    )

    if old_summary is None:
        old_summary = '(这是第一轮对话, 没有总结)'
    if user_info is None:
        user_info = '(这是第一次咨询, 你没有对来访者的记忆)'

    system_prompt = f'你是一个心理咨询师, 请用合适的方式和来访者对话.\n\n你还拥有一个记忆库, 记录了你对来访者的过往若干次咨询的总结. 你需要根据当前对话内容, 判断是否需要在记忆库中调取记忆. 如果需要, 请通过函数调用 (function calling), 调用 get_memory 函数来提取关于来访者的记忆; 你需要提供用于检索记忆的描述, 该描述将被用于与你过往的记忆进行相似度匹配, 并由系统根据相似度来返回0-3条给你; 请不要把来访者的名字包含在记忆描述中, 用"我"代表自己, 用"来访者"代表来访者即可.\n\n# 对本次咨询已有内容的总结\n\n{old_summary}\n\n# 你对来访者的用户画像\n\n{user_info}'

    response = model_a.chat.completions.create(
        model=model_a_name,
        messages=[
            { 'role': 'system', 'content': system_prompt },
            *history_messages,
            { 'role': 'user', 'content': user_input },
        ],
        tools=[get_memory_tool],
        stream=False,
    )

    token_usage: int = 0
    if response.usage is not None:
        token_usage = response.usage.total_tokens
    else:
        raise ValueError('response.usage is None')
    
    tool_call: ChatCompletionMessageToolCall | None = None
    if response.choices[0].message.tool_calls is not None:
        tool_call = response.choices[0].message.tool_calls[0]
        result.output_messages.append(ChatCompletionAssistantMessageParam(
            role='assistant',
            content='',
            tool_calls=[ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                type=tool_call.type,
                function=Function(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )]
        ))

    if tool_call is not None:
        memory_description: str = json.loads(tool_call.function.arguments)['memoryDescription']
        if type(memory_description) != str:
            raise ValueError('通过 json.loads 解析出的 memoryDescription 不是字符串')
        
        if callable(existing_memo):
            existing_memo = existing_memo()
        if callable(exclude_memo):
            exclude_memo = exclude_memo()

        recalled_memories: list[Memo] = get_memory(
            embedding_api=embbeding_api, 
            memo_description=memory_description,
            existing_memo=existing_memo, 
        )
        recalled_memories = [memo for memo in recalled_memories if memo.content not in exclude_memo]
        recalled_str: str = ''
        if len(recalled_memories) > 0:
            recalled_str = f'在记忆库里找到了一些相关的记忆:\n\n{'\n'.join(f'- {memo.content}' for memo in recalled_memories)}'
        else:
            recalled_str = '没能在记忆库中找到更多相关的记忆'

        result.new_memo_calls = [memo.session_id for memo in recalled_memories]
        result.output_messages.append({ 'role': 'tool', 'content': recalled_str, 'tool_call_id': tool_call.id })

        new_response = model_a.chat.completions.create(
            model=model_a_name,
            messages=[
                { 'role': 'system', 'content': system_prompt },
                *history_messages,
                *result.output_messages,
            ],
            tools=[get_memory_tool],
            stream=False,
            tool_choice='none',
        )

        if new_response.usage is not None:
            token_usage = max(token_usage, new_response.usage.total_tokens)
        else:
            raise ValueError('new_response.usage is None')
        
        result.output_messages.append({ 'role': 'assistant', 'content': new_response.choices[0].message.content })
    else:
        result.output_messages.append({ 'role': 'assistant', 'content': response.choices[0].message.content })

    result.new_summary = update_summary(
        model=model_b,
        model_name=model_b_name,
        old_summary=old_summary,
        new_messages=result.output_messages,
        user_info=user_info,
    )
    result.new_reduce_num = get_reduce(
        old_reduce=old_reduce_num,
        token_usage=token_usage,
        token_limit=max_token_usage,
    )

    return result


def update_user_info(
    *,
    model: OpenAI,
    model_name: str,
    latest_summary: str,
    old_user_info: str | None = None,
) -> str:
    '''
    ## 生成/更新用户画像
    '''
    if old_user_info is None:
        old_user_info = '(这是第一次咨询, 你没有对来访者的记忆)'

    prompt = f'你是一个心理咨询师, 在你与来访者的一次咨询结束后, 你需要根据本次咨询的总结来更新你对来访者的记忆. 你将收到这次咨询的总结和本次咨询前你对来访者的记忆. 你的目标是参考本次咨询总结, 在原有记忆中纳入新的内容, 同时确保记忆长度不超过 20 个句子, 最后输出新的记忆. 为了更好地完成任务, 请按照以下步骤操作: \n\n1. 仔细分析本次咨询前你对来访者的记忆, 从中提取出已有的信息和事实. \n2. 考虑本次咨询的总结, 找出需要纳入记忆的任何新的或已改变的信息. \n3. 结合新旧信息, 创建最新的记忆. \n4. 以简洁明了的方式组织更新的记忆, 确保不超过 20 句话. \n5. 注意信息的相关性和重要性, 重点抓住最重要的方面, 同时保持记忆的整体连贯性. \n\n此外, 请不要把你和来访者的名字包含在记忆中, 用"咨询师"代表自己, 用"来访者"代表来访者即可; 你的输出应只包含记忆, 请不要额外输出任何其他内容.\n\n# 本次咨询的总结\n\n{latest_summary}\n\n# 本次咨询前你对来访者的记忆\n\n{old_user_info}'

    response = model.chat.completions.create(
        model=model_name,
        messages=[
            { 'role': 'user', 'content': prompt }
        ],
        stream=False,
    )
    
    new_user_info = response.choices[0].message.content
    if new_user_info is None:
        raise ValueError('response.choices[0].message.content is None')
    
    return new_user_info
