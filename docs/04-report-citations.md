# 04 - 报告引用追溯

## 问题
- 报告 prompt 只传 `title + year + citations`，没有 DOI/URL
- 报告中引用论文无链接、无编号，读者无法溯源
- LLM 可能改写标题

## 方案

### 1. Prompt 传入完整元数据
```python
high_papers_text = "\n".join(
    f"[P{i+1}] {p.title} ({p.year}) DOI:{p.doi or 'N/A'} [citations:{p.citation_count}]"
    for i, p in enumerate(high_papers)
)
```

### 2. 报告要求编号引用
```
REPORT_SYSTEM 增加：
- 引用论文时使用 [P1], [P2] 等编号对应提供的论文列表
- 不要引用未在列表中提供的论文
```

### 3. 报告末尾附参考文献列表
```
## 参考文献
[P1] Title (Year). DOI: xxx
[P2] Title (Year). DOI: xxx
```

## 涉及文件
- `backend/app/agent/prompts.py` — REPORT_SYSTEM / REPORT_USER
- `backend/app/agent/runner.py` — `_generate_report` 传完整元数据

## 验证
- 报告中每个 [Px] 引用都能在参考文献列表中找到
- 参考文献列表中的 DOI 可点击
