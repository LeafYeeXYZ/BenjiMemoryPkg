**我的本基项目的咨询师长时记忆模块**

# 记忆模块

- `Memo`: 记忆对象的结构
- `ChatResult`: `chat` 函数的返回值的结构
- `chat`: 对话函数
- `update_user_info`: 更新用户画像函数

```python
from memory import Memo, ChatResult, chat, update_user_info
```

## `Memo`

这个 `dataclass` 定义了传入的记忆对象的结构. 

- `content`: 记忆内容 (一次咨询中最后一条对话内总结)
- `vector`: 记忆内容的向量表示
- `session_id`: 这条记忆对应的会话 ID.

```python
@dataclass
class Memo:
    vector: list[float]
    content: str
    session_id: str
```

## `ChatResult`

这个 `dataclass` 定义了 `chat` 函数的返回值的结构.

- `output_messages`: 新增的对话消息 (取决于是否调用记忆, 长度为 `2` 或 `4`)
- `new_reduce_num`: 新的调节因子数值
- `new_summary`: 新的对话内总结
- `new_memo_calls`: 如果有记忆调用, 则为调用的所有记忆的 `session_id`

```python
@dataclass
class ChatResult:
    output_messages: list[ChatCompletionMessageParam]
    new_reduce_num: int
    new_summary: str
    new_memo_calls: list[str]
```

## `chat`

对话函数, 接受一系列信息, 返回 `ChatResult`.

- `model_a`: 模型 A 的实例
- `model_a_name`: (默认为 `''`) 模型 A 的名称
- `model_b`: 模型 B 的实例
- `model_b_name`: (默认为 `''`) 模型 B 的名称
- `embbeding_api`: 一个函数, 接受一个字符串, 返回一个向量表示
- `old_reduce_num`: 旧的调节因子数值, 对于第一轮对话, 请传入 `0`
- `max_token_usage`: (默认为 `3000`) 最大 token 使用量
- `history_messages`: 历史消息, 不含用户的最新输入; **这个历史消息应当已经根据调节因子进行了截取**
- `user_input`: 用户输入
- `old_summary`: (默认为 `None`) 旧的对话内总结, 对于第一轮对话, 请不传或传入 `None`
- `user_info`: (默认为 `None`) 用户画像, 对于第一次咨询, 请不传或传入 `None`
- `existing_memo`: 所有可供调用的记忆, 请传入一个列表或一个返回列表的函数 (如果可以实现的话推荐后者, 以减少数据库查询)
- `exclude_memo`: **在根据调节因子截取后的历史消息中**已经调用过的记忆的 `session_id` 列表, 请传入一个列表或一个返回列表的函数 (如果可以实现的话推荐后者, 以减少数据库查询) (`tips`: 记得把 `ChatResult.new_memo_calls` 存到数据库中)

```python
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
```

## `update_user_info`

更新用户画像函数, 接受一系列信息, 返回新的用户画像.

- `model`: 模型 (B) 的实例
- `model_name`: 模型 (B) 的名称
- `latest_summary`: 一次咨询中的最后一条对话内总结
- `old_user_info`: (默认为 `None`) 旧的用户画像, 对于第一次咨询, 请不传或传入 `None`

```python
def update_user_info(
    *,
    model: OpenAI,
    model_name: str,
    latest_summary: str,
    old_user_info: str | None = None,
) -> str:
```

# 示例

见 `./test.py` 文件
