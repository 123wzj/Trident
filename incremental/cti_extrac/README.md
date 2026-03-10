# CTI 智能提取系统 (v1)

基于 LangGraph 和 LLM 的威胁情报增量式提取系统，支持从长文本中自动提取 TTPs 和 IOCs。

## 功能特性

### 核心能力

- **TTP 提取**：自动识别并提取 MITRE ATT&CK 战术、技术和过程
- **IOC 提取**：提取以下威胁指标：
  - IP 地址（IPv4/IPv6）
  - 域名
  - URL
  - 文件哈希（MD5/SHA1/SHA256）
  - CVE 漏洞编号
- **智能摘要**：生成威胁情报摘要报告

### 技术创新

#### 1. 增量式跨窗口提取
- 将长文档分块处理（默认 8000 字符/块）
- 实体寄存器（Entity Register）维护跨块上下文记忆
- 防止边界实体丢失，支持跨块引用

#### 2. 自洽性推理 (Self-Consistency)
- 每个文本块进行多轮独立推理（默认 3 轮）
- 投票机制聚合结果，置信度阈值过滤（默认 0.66）
- 提高提取准确率

#### 3. 对抗性去偏 (Adversarial Debiasing)
- 使用推理模型（DeepSeek-R1）对提取结果进行审核
- 模拟"魔鬼代言人"角色，识别误报
- 降低假阳性率

#### 4. 正则预提取 + LLM 精炼
- 正则表达式快速识别候选 IOC
- LLM 基于候选和上下文进行精确分类和角色标注

## 安装依赖

```bash
pip install langchain langchain-core langgraph langchain-text-splitters pydantic
```

## 配置说明

### 模型配置

```python
# 主提取模型
llm = deepseek_model

# 审核模型（推荐使用推理能力强的模型）
llm_critic = deepseek_reasoner
```

### 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 8000 | 文本分块大小（字符数） |
| `CHUNK_OVERLAP` | 1000 | 分块重叠大小 |
| `SELF_CONSISTENCY_ROUNDS` | 3 | 自洽性推理轮数 |
| `CONFIDENCE_THRESHOLD` | 0.66 | 置信度过滤阈值 |

### 模型定价配置

系统内置了多个模型的定价配置（单位：美元/百万 Token）：

```python
"deepseek-chat": {"input": 0.29, "output": 0.43}
"deepseek-reasoner": {"input": 0.29, "output": 0.43}
"qwen-max": {"input": 0.46, "output": 1.83}
"claude-opus-4.5": {"input": 5, "output": 25}
# ... 更多模型
```

可在 `TokenCostHandler.pricing` 中自定义或添加新模型。

## 使用方法

### 基本用法

```python
python v1.py
```

### 自定义输入

修改 `input_file_path` 变量：

```python
input_file_path = "./cti/your_report.md"
```

### 输出格式

结果保存为 JSON 文件到 `output/` 目录：

```json
{
  "summary": "威胁情报摘要...",
  "document_stats": {
    "total_length": 15000,
    "num_chunks": 2,
    "chunk_size": 8000,
    "chunk_overlap": 1000
  },
  "innovation_metrics": {
    "self_consistency_rounds": 3,
    "confidence_threshold": 0.66,
    "cross_chunk_references": 5,
    "entity_register_size": 42
  },
  "total_indicators": 15,
  "cost_report": {
    "total_tokens": 12500,
    "prompt_tokens": 9500,
    "completion_tokens": 3000,
    "total_cost": 0.00428,
    "model_breakdown": [...]
  },
  "ttps": [...],
  "iocs": {
    "files": [...],
    "ips": [...],
    "domains": [...],
    "urls": [...],
    "cves": [...]
  }
}
```

## 工作流程

```
输入文本
    ↓
[文本分块] (chunk_text)
    ↓
┌───────────────────────────────────────┐
│  循环处理每个 Chunk                    │
│  ↓                                    │
│  [TTP 提取] (extract_ttp_incremental)  │
│  ↓                                    │
│  [IOC 提取] (extract_ioc_incremental)  │
│  ↓                                    │
│  [下一块] (next_chunk)                │
│  ↓                                    │
│  继续？→ 是 ──────────────────────────┘
│       │
│       否
│       ↓
└───────────────────────────────────────┐
         ↓
[对抗性审核] (adversarial_critic)
         ↓
[聚合结果] (aggregate)
         ↓
输出 JSON
```

## 数据模型

### 提取的实体

#### TTP (战术/技术/过程)
```python
{
  "technique_id": "T1566.001",
  "technique_name": "Spearphishing Attachment",
  "description": "攻击者通过钓鱼邮件发送恶意附件...",
  "confidence_score": 1.0,
  "chunk_id": 0
}
```

#### IOC 基类字段
- `role`: C2 / Payload_Delivery / Victim / Benign / Scanner / Unknown
- `context`: 上下文说明
- `confidence_score`: 置信度
- `chunk_id`: 来源块 ID
- `cross_chunk_reference`: 是否跨块引用

## 成本追踪

系统自动统计每次运行的 Token 使用和成本：

- **总 Token 数**
- **输入/输出 Token 分离统计**
- **按模型分组的详细统计**
- **总成本估算**

报告会在控制台打印并保存到输出 JSON 的 `cost_report` 字段。

## 高级用法

### 自定义 Prompt

修改 `prompts.py` 文件中的以下变量：
- `TTP_EXTRACTION_SYSTEM`
- `TTP_EXTRACTION_USER`
- `IOC_EXTRACTION_SYSTEM`
- `IOC_EXTRACTION_USER`
- `ADVERSARIAL_CRITIC_SYSTEM`
- `ADVERSARIAL_CRITIC_USER`

### 调试模式

取消注释以下行启用调试：

```python
set_debug(True)
```

这将打印完整的 LLM 输入 prompt 和原始输出。

### 自定义白名单

在 `regex_extract_iocs` 函数中修改 `IGNORE_DOMAINS`：

```python
IGNORE_DOMAINS = {
    "example.com",
    "trusted-source.org"
}
```

## 项目结构

```
cti_extrac/
├── v1.py                 # 主程序
├── prompts.py            # Prompt 模板
├── cti/                  # 输入报告目录
│   └── *.md
└── output/               # 输出结果目录
    └── *.json
```

## 注意事项

1. **API Key**：确保已配置相应模型的 API Key
2. **成本控制**：大文件处理会产生较多 API 调用，建议先用小文件测试
3. **内存使用**：超长文档可能需要调整分块大小
4. **模型选择**：审核阶段建议使用推理能力强的模型

## 许可证

MIT License
