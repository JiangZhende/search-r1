# Search-R1 Multi-Turn GRPO Training

使用 GRPO（Group Relative Policy Optimization）训练具备多轮检索能力的语言模型。模型在回答问题时可主动调用搜索引擎，并根据检索结果进行多轮推理。

## 项目结构

```
train_search_r1_mt.py   # 多轮 GRPO 训练主脚本
grpomt.sh               # 训练启动脚本
batch_eval_mt.py        # 评测脚本（与训练 rollout 完全一致）
search/
  retrieval_server.py   # BM25/Dense 检索服务（FastAPI）
retrieval_launch.sh     # 启动检索服务
datas/
  wiki-18.jsonl         # 语料库（Wikipedia 2018）
  bm25/                 # BM25 索引
```

## 对话格式

训练和推理使用相同的多轮对话格式：

```
user:      <system_prompt> Question: ...
assistant: <thought>...</thought><search>query</search>
user:      <information>\n检索结果\n</information>
assistant: <thought>...</thought><answer>答案</answer>
```

- `<thought>` 内为推理过程
- `<search>` 触发检索，最多调用 5 次
- `<answer>` 输出最终答案
- `<information>` 块由环境注入，**不计入 loss**（`completion_mask=0`）

## 快速开始

### 1. 安装依赖

```bash
pip install trl transformers datasets requests uvicorn fastapi torch
```

### 2. 启动检索服务

```bash
bash retrieval_launch.sh
```

服务默认监听 `http://127.0.0.1:8000/retrieve`。

检索服务参数（见 `retrieval_launch.sh`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--index_path` | `datas/bm25` | BM25 索引目录 |
| `--corpus_path` | `datas/wiki-18.jsonl` | 语料库文件 |
| `--retriever_name` | `bm25` | 检索器类型 |
| `--topk` | 3 | 每次检索返回文档数 |

### 3. 启动训练

```bash
bash grpomt.sh
```

等效命令（完整参数）：

```bash
../grpo/bin/python train_search_r1_mt.py \
    --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
    --output_dir outputs/search-r1-mt2 \
    --max_completion_length 4096 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --max_steps 2000 \
    --beta 0.02 \
    --reward_weights 1.0 1.0 0.5 0.5 0.5 \
    --temperature 1.0 \
    --log_completions False \
    --report_to none
```

训练数据集：[RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)（Natural Questions）

### 4. 评测

```bash
# 单模型评测
../grpo/bin/python batch_eval_mt.py \
    --model_name_or_path outputs/search-r1-mt2 \
    --num_samples 200

# 多 checkpoint 对比
../grpo/bin/python batch_eval_mt.py \
    --model_name_or_path \
        /home/l33500/models/Qwen/Qwen3-0.6B \
        outputs/search-r1-mt2/checkpoint-500 \
        outputs/search-r1-mt2/checkpoint-1000 \
        outputs/search-r1-mt2/checkpoint-1500 \
        outputs/search-r1-mt2/checkpoint-2000 \
    --num_samples 200 \
    --output_file eval_mt.jsonl
```

评测指标：

| 指标 | 说明 |
|------|------|
| `em` | Exact Match（子串匹配，含长度惩罚） |
| `f1` | Token-level F1 |
| `has_answer_rate` | `<answer>` 标签出现率 |
| `mean_search_calls` | 平均检索次数 |
| `mean_model_tokens` | 平均模型生成 token 数（mask=1） |
| `mean_total_tokens` | 平均总 completion token 数 |

## 主要超参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--beta` | `0.02` | KL 散度系数，防止策略崩塌，**最关键参数** |
| `--learning_rate` | `1e-6` | 过大易导致熵崩（建议配合 `constant_with_warmup`） |
| `--num_generations` | `8` | 每条数据采样组数，越大梯度越稳定 |
| `--max_steps` | `2000` | 建议训练至 1500 步左右早停 |
| `--reward_weights` | `1.0 1.0 0.5 0.5 0.5` | 5 个奖励函数权重（见下方） |
| `--temperature` | `1.0` | 训练时生成温度 |
| `--max_completion_length` | `4096` | 包含所有 assistant + information token |

## 奖励函数

训练使用 5 个奖励函数，权重由 `--reward_weights` 指定：

| 权重 | 函数 | 说明 |
|------|------|------|
| 1.0 | `compute_score_em` | 子串 EM（含长度比惩罚，防止塞答案） |
| 1.0 | `compute_score_f1` | Token-level F1 |
| 0.5 | `compute_search_persistence` | 检索行为塑形（惩罚重复查询、奖励高效检索） |
| 0.5 | `compute_retrieval_quality` | 检索结果包含答案时给 +0.20 |
| 0.5 | `compute_answer_quality` | 惩罚过长答案、答案中包含 Doc 引用等格式问题 |

## 实验结果（NQ test，n=200，greedy）

| checkpoint | EM | F1 | has_answer_rate | mean_search_calls |
|---|---|---|---|---|
| Qwen3-0.6B (base) | 0.135 | 0.127 | 0.615 | 1.12 |
| checkpoint-500 | 0.230 | 0.197 | 0.990 | 1.005 |
| checkpoint-1000 | 0.230 | 0.223 | 0.995 | 1.995 |
| **checkpoint-1500** | **0.240** | **0.267** | 1.000 | 2.865 |
| checkpoint-2000 | 0.225 | 0.241 | 1.000 | 2.865 |

推荐使用 **checkpoint-1500**（EM 和 F1 均最优，step-2000 出现轻微退化）。

## 注意事项

- 训练前必须确保检索服务（`retrieval_launch.sh`）已在后台运行
- 评测脚本 `batch_eval_mt.py` 使用与训练完全相同的 `MultiTurnSearchRollout`，保证评测分布一致
- `em_check` 使用长度比惩罚（`gold_len / max(pred_len, gold_len)`），避免模型通过在答案中填充大量无关文字绕过 EM 检查
- `<information>` 块的 token 在训练时 `completion_mask=0`，不计入 loss，但计入 `max_completion_length` 预算
