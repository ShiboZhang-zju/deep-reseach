"""LLM 探测 + 自动重跑脚本。

逻辑：
1. 探测本地 Qwen LLM 后端是否恢复（一次真实 chat 调用，非仅端口）。
2. 若恢复且尚未触发过重跑（无 _autorerun_done 标记），则通过后端 API 新建并启动一个研究任务。
3. 写标记文件 _autorerun_done，避免后续重复新建任务。
4. 追加一行结果到 _autorerun.log 便于人工回看。

约定：后端服务需已在 http://localhost:8000 运行；本脚本只读探测 + 触发，不做重负载。
"""
import json
import os
import sys
import urllib.request
from datetime import datetime

BACKEND = "http://localhost:8000"
LLM_URL = "http://28.251.176.200:8080/openapi/chat/completions"
LLM_MODEL = "Qwen3.5-397B-A17B-W8A8-P800-Functional-Agent"
DONE_FLAG = os.path.join(os.path.dirname(__file__), "_autorerun_done")
LOG_FILE = os.path.join(os.path.dirname(__file__), "_autorerun.log")

RESEARCH_INPUT = (
    "Retrieval-augmented generation (RAG) for large language models: methods to "
    "improve factual accuracy and reduce hallucination in open-domain question "
    "answering. Focus on retriever-reader architectures, benchmarks, and known "
    "failure modes. Single consumer GPU only, no training of large models."
)


def log(msg: str):
    line = "%s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def llm_healthy() -> bool:
    """用项目自己的 LLM provider 探测，最贴近真实调用路径。"""
    try:
        import asyncio
        from app.llm.factory import get_llm

        async def _probe():
            llm = get_llm()
            resp = await llm.chat(
                [{"role": "user", "content": "What is 2 + 3? Answer with the number only."}]
            )
            return bool(resp and resp.strip())

        return asyncio.run(_probe())
    except Exception as e:
        log("LLM probe failed: %s" % str(e)[:160])
        return False


def create_and_start_task() -> str:
    body = json.dumps({"user_input": RESEARCH_INPUT}).encode("utf-8")
    req = urllib.request.Request(BACKEND + "/api/tasks", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        task = json.loads(r.read().decode("utf-8"))
    tid = task["id"]
    start = urllib.request.Request(BACKEND + "/api/tasks/%s/start" % tid,
                                   data=b"", method="POST")
    with urllib.request.urlopen(start, timeout=20) as r:
        r.read()
    return tid


def main():
    if os.path.exists(DONE_FLAG):
        log("Already re-ran (flag present), skipping.")
        return
    if not llm_healthy():
        log("LLM still down (500 / unreachable), will retry next cycle.")
        return
    log("LLM RECOVERED — creating & starting rerun task...")
    try:
        tid = create_and_start_task()
        with open(DONE_FLAG, "w", encoding="utf-8") as f:
            f.write(tid + "\n")
        log("Rerun task started: %s" % tid)
    except Exception as e:
        log("Failed to create/start task: %s" % str(e)[:200])


if __name__ == "__main__":
    main()
