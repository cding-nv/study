# AI Radar

每天一条命令,拿到 AI 行业跨源日报 — 从研究前沿 (HuggingFace + arXiv)、推理系统 (vLLM / SGLang / TensorRT-LLM)、厂商方向 (NVIDIA / AMD / DeepMind)、产业内幕 (Reuters / The Information / SemiAnalysis)、真实使用 (OpenRouter),自动汇总并做 LLM 综合,给出 TL;DR + 深入研究建议。

## 使用

```bash
# 1. 装依赖
pip install -r AI_Radar/requirements.txt

# 2. 配好环境变量 (Windows PowerShell 例)
$env:ANTHROPIC_AUTH_TOKEN = "sk-xxx"        # 或 ANTHROPIC_API_KEY
$env:ANTHROPIC_BASE_URL   = "https://..."   # 可选
$env:ANTHROPIC_MODEL      = "claude-sonnet-4-6"   # 可选,默认见 config.yaml
$env:GITHUB_TOKEN         = "github_pat_..."

# 3. 跑
python AI_Radar/run.py
```

代理默认走 `http://127.0.0.1:7078` (在 `config.yaml` 改)。

## 输出

- `AI_Radar/reports/YYYY-MM-DD.md` — 当日日报
- `AI_Radar/cache/YYYY-MM-DD/*.json` — 每源原始数据 (可用 `--from-cache` 不重新抓取重出报告)

## 常用命令

```bash
python AI_Radar/run.py                          # 完整跑
python AI_Radar/run.py --only huggingface arxiv # 只跑指定源
python AI_Radar/run.py --no-llm                 # 跳过 LLM,规则版汇总
python AI_Radar/run.py --from-cache             # 用今日缓存重出报告
python AI_Radar/run.py --from-cache --day 2026-08-17
python AI_Radar/run.py -v                       # 详细日志
```

## 采集器列表

| name (`--only <name>`) | 内容 |
|---|---|
| `huggingface` | Trending Papers + Trending Models |
| `github` | vLLM / SGLang / TRT-LLM 的 releases / merged PRs / hot issues (config.yaml 可改仓库) |
| `vendor_blogs` | NVIDIA Developer / AMD ROCm / DeepMind (RSS,7 天窗口) |
| `arxiv` | cs.AI / cs.LG / cs.CL 最新提交 |
| `openrouter` | 模型上架 + 使用排行 |
| `reuters_tech` | Reuters Technology RSS |
| `theinformation` | AI 频道标题 (付费墙,只列标题) |
| `semianalysis` | Substack RSS (免费预览) |

## 报告结构

```
⚡ TL;DR (LLM 综合的当天要事)
📝 跨源叙事 (研究 / 推理栈 / 厂商 / 产业 / 使用)
🔬 §1 Research Frontier (HF trending + arXiv)
🏗️ §2 Inference Stack (每个仓库 releases/PRs/issues)
🏢 §3 Vendor Blogs
💰 §4 Industry Intel
📊 §5 OpenRouter Usage
🔎 §6 Deep Dive 建议 (LLM 给的今日切入点 + 可复制 prompt)
🧠 附录 · 关键词跨源热度 (MoE / Agentic RL / long-context …)
```

## 容错

- 单源失败不影响整体,失败会在报告顶部列出
- `ANTHROPIC_AUTH_TOKEN` 未设 / LLM 调用失败会自动回落到规则版综合
- 付费源 (The Information / SemiAnalysis) 只保证标题层,不保证正文

## 定制

- 加/去 GitHub 仓库、加关键词分类 → 编辑 `config.yaml`
- 加新数据源 → 在 `collectors/` 新增一个继承 `BaseCollector` 的类,然后加进 `collectors/__init__.py::ALL_COLLECTORS`
