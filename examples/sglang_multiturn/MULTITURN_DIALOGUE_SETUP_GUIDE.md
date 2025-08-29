# 多轮对话奖励函数设置指南

本指南将教您如何设置自己的多轮对话奖励函数和相应的训练脚本，其中一个模型用于对每轮对话进行打分，另一个模型用于提供对话回复。

## 概述

我们的多轮对话强化学习系统包含以下核心组件：

1. **对话交互系统 (Interaction System)**: 管理多轮对话的状态和流程
2. **奖励函数 (Reward Function)**: 评估对话质量并计算奖励分数
3. **评分模型 (Scoring Model)**: 专门用于评估对话质量的模型
4. **响应模型 (Response Model)**: 主要的对话生成模型
5. **训练脚本**: 整合所有组件的训练配置

## 架构图

```
用户输入 -> 响应模型 -> 生成对话 -> 评分模型 -> 质量分数
    ↓                                            ↓
训练数据 <- 奖励函数 <- 交互系统 <- 多轮对话管理 <- 分数整合
```

## 第一步：创建多轮对话交互类

我们已经创建了 `MultiturnDialogueInteraction` 类，它实现了以下功能：

### 核心特性
- **每轮评分**: 使用专门的评分模型对每轮对话进行质量评估
- **上下文感知**: 根据对话历史生成合适的回复
- **可配置的评分标准**: 支持自定义评分阈值和权重
- **对话终止条件**: 智能决定何时结束对话

### 配置选项
```yaml
interaction:
  - name: "multiturn_dialogue"
    class_name: "verl.interactions.multiturn_dialogue_interaction.MultiturnDialogueInteraction"
    config:
      scoring_model: "dialogue_scorer"           # 评分模型名称
      response_model: "dialogue_responder"       # 响应模型名称
      min_score_threshold: 0.3                   # 最低分数阈值
      good_score_threshold: 0.7                  # 良好分数阈值
      max_turns: 8                               # 最大对话轮数
      turn_weights: [1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6]  # 每轮权重
```

## 第二步：实现自定义奖励函数

我们提供了两种奖励函数实现方式：

### 1. 基于规则的奖励函数 (`compute_multiturn_dialogue_score`)

这种方法使用启发式规则来评估对话质量：

```python
def compute_multiturn_dialogue_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    基于规则的多轮对话评分函数
    
    评估维度：
    - 连贯性 (coherence): 对话的逻辑连贯性
    - 相关性 (relevance): 与上下文的相关程度
    - 有用性 (helpfulness): 对用户的帮助程度
    - 对话流畅性 (conversation_flow): 对话的自然流畅度
    """
```

### 2. 基于模型的奖励函数 (`compute_model_based_dialogue_score`)

这种方法调用外部评分模型API：

```python
async def compute_model_based_dialogue_score(
    data_source, solution_str, ground_truth, extra_info=None,
    scoring_model_url="http://localhost:8000/v1/chat/completions",
    scoring_model_name="gpt-3.5-turbo"
):
    """
    基于外部模型的对话评分函数
    
    优势：
    - 更准确的评分
    - 可以使用专门训练的评分模型
    - 支持复杂的评分标准
    """
```

## 第三步：配置文件设置

### 交互配置 (`multiturn_dialogue_config.yaml`)
```yaml
interaction:
  - name: "multiturn_dialogue"
    class_name: "verl.interactions.multiturn_dialogue_interaction.MultiturnDialogueInteraction"
    config:
      scoring_model: "dialogue_scorer"
      response_model: "dialogue_responder"
      min_score_threshold: 0.3
      good_score_threshold: 0.7
      max_turns: 8
      turn_weights: [1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6]
```

### 训练配置 (`multiturn_dialogue_grpo.yaml`)
```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  max_prompt_length: 2048
  max_response_length: 2048
  train_batch_size: 256
  return_raw_chat: True

actor_rollout_ref:
  hybrid_engine: True
  rollout:
    name: sglang
    multi_turn:
      enable: True
      max_user_turns: 8
      max_assistant_turns: 8
```

## 第四步：训练脚本

我们提供了两个训练脚本：

### 1. 基础训练脚本 (`run_multiturn_dialogue_custom_reward.sh`)

适用于使用规则基础奖励函数的场景：

```bash
#!/bin/bash
# 多轮对话训练脚本（基于规则的奖励函数）

# 设置模型路径
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_DATA="$HOME/data/multiturn_dialogue/train.parquet"
VAL_DATA="$HOME/data/multiturn_dialogue/test.parquet"

# 运行训练
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multiturn_dialogue_grpo' \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score \
    # ... 其他配置参数
```

### 2. 高级训练脚本 (`run_advanced_multiturn_dialogue.sh`)

适用于使用外部评分模型的场景：

```bash
#!/bin/bash
# 高级多轮对话训练脚本（基于模型的奖励函数）

# 设置模型路径
DIALOGUE_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
SCORING_MODEL_URL="http://localhost:8001/v1/chat/completions"
SCORING_MODEL_NAME="dialogue-scorer"

# 运行训练
python3 -m verl.trainer.main_ppo \
    custom_reward_function.name="compute_model_based_dialogue_score" \
    custom_reward_function.reward_kwargs.scoring_model_url="$SCORING_MODEL_URL" \
    custom_reward_function.reward_kwargs.scoring_model_name="$SCORING_MODEL_NAME" \
    # ... 其他配置参数
```

## 第五步：数据准备

### 数据格式

训练数据应该是Parquet格式，包含以下字段：

```python
{
    "messages": [
        {"role": "user", "content": "用户消息1"},
        {"role": "assistant", "content": "助手回复1"},
        {"role": "user", "content": "用户消息2"}
        # ... 更多轮次
    ],
    "data_source": "dataset_name",
    "ground_truth": "期望的回复（可选）"
}
```

### 数据预处理脚本示例

```python
import pandas as pd
import json

def prepare_dialogue_data(conversations):
    """准备多轮对话数据"""
    data = []
    for conv in conversations:
        # 确保对话格式正确
        formatted_conv = {
            "messages": conv["messages"],
            "data_source": "custom_dialogue",
            "ground_truth": conv.get("expected_response", "")
        }
        data.append(formatted_conv)
    
    df = pd.DataFrame(data)
    df.to_parquet("dialogue_train.parquet")
    return df
```

## 第六步：评分模型设置

### 选项1：使用现有API模型

如果您有访问GPT、Claude等API的权限：

```python
SCORING_MODEL_URL = "https://api.openai.com/v1/chat/completions"
SCORING_MODEL_NAME = "gpt-3.5-turbo"
```

### 选项2：部署自己的评分模型

使用vLLM或其他推理框架部署您的评分模型：

```bash
# 启动评分模型服务
python -m vllm.entrypoints.openai.api_server \
    --model your-scoring-model \
    --port 8001 \
    --host 0.0.0.0
```

### 选项3：使用本地模型

修改奖励函数以直接调用本地模型：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalScoringModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
    
    def score_dialogue(self, dialogue_text):
        # 实现本地评分逻辑
        pass
```

## 运行步骤

### 1. 准备环境

```bash
cd /path/to/verl
pip install -e .
```

### 2. 准备数据

```bash
# 将您的对话数据转换为Parquet格式
python prepare_dialogue_data.py --input conversations.json --output train.parquet
```

### 3. 启动评分模型（如果使用外部模型）

```bash
# 启动评分模型服务
python -m vllm.entrypoints.openai.api_server \
    --model your-scoring-model \
    --port 8001
```

### 4. 运行训练

```bash
# 使用基础脚本
./examples/sglang_multiturn/run_multiturn_dialogue_custom_reward.sh

# 或者使用高级脚本
./examples/sglang_multiturn/run_advanced_multiturn_dialogue.sh
```

## 自定义配置

### 调整评分标准

您可以修改奖励函数中的评分权重：

```python
# 在 multiturn_dialogue.py 中修改权重
weights = {
    'coherence': 0.3,      # 连贯性权重
    'relevance': 0.3,      # 相关性权重  
    'helpfulness': 0.25,   # 有用性权重
    'conversation_flow': 0.15  # 对话流畅性权重
}
```

### 调整对话终止条件

在交互配置中修改：

```yaml
config:
  min_score_threshold: 0.3    # 降低以允许更多轮次
  good_score_threshold: 0.7   # 提高以要求更高质量
  max_turns: 10               # 增加最大轮数
```

### 自定义评分提示

```yaml
config:
  scoring_prompt_template: |
    请根据以下标准对这轮对话进行0.0到1.0的评分：
    1. 回答是否准确和有用
    2. 语言是否自然流畅
    3. 是否保持对话连贯性
    
    对话内容：
    {dialogue}
    
    分数：
```

## 监控和调试

### 使用Wandb监控训练

```bash
export WANDB_PROJECT="multiturn_dialogue_rl"
export WANDB_ENTITY="your-entity"
```

### 查看训练日志

```bash
# 查看控制台输出
tail -f logs/training.log

# 查看评分详情
grep "Dialogue score" logs/training.log
```

### 调试评分函数

```python
# 测试评分函数
from verl.utils.reward_score.multiturn_dialogue import compute_score

test_dialogue = "User: Hello\nAssistant: Hi there! How can I help you?"
score = compute_score("test", test_dialogue, "", {})
print(f"Score: {score}")
```

## 常见问题

### Q: 如何处理评分模型不可用的情况？
A: 系统会自动回退到基于规则的评分方法。确保在 `compute_model_based_dialogue_score` 中实现了适当的错误处理。

### Q: 如何优化训练性能？
A: 
- 调整 `MICRO_BATCH_SIZE` 和 `TRAIN_BATCH_SIZE`
- 使用 `OFFLOAD=True` 来节省GPU内存
- 减少 `max_turns` 来缩短对话长度

### Q: 如何评估模型性能？
A: 
- 监控平均对话分数
- 查看生成的对话样例
- 使用验证集进行定期测试

## 总结

通过以上步骤，您可以：

1. ✅ 设置多轮对话交互系统
2. ✅ 实现自定义奖励函数
3. ✅ 配置评分模型和响应模型  
4. ✅ 运行完整的训练流程
5. ✅ 监控和调试训练过程

这个系统提供了灵活的框架，允许您根据具体需求调整评分标准、对话策略和训练参数。通过迭代优化，您可以训练出高质量的多轮对话模型。