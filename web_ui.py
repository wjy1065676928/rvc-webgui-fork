import warnings
import logging
import git
import torch

# wjy1214
import subprocess
import threading
import re

# Try to import the Dictionary class in a way compatible with different fairseq versions
try:
    from fairseq.data.dictionary import Dictionary as FairseqDictionary
except ImportError:
    try:
        from fairseq.data import Dictionary as FairseqDictionary
    except ImportError:
        FairseqDictionary = None

if FairseqDictionary is not None:
    torch.serialization.add_safe_globals([FairseqDictionary])

warnings.filterwarnings("ignore")

# Set logging levels for noisy modules
for l in ["httpx", "uvicorn", "httpcore", "urllib3", "PIL", "faiss"]:
    logging.getLogger(l).setLevel(logging.ERROR)

# Now import shared after setting up logging
import shared
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
import uvicorn

from tabs.inference_tab import create_inference_tab
from tabs.train_tab import create_train_tab
from tabs.vocal_tab import create_vocal_tab
from tabs.ckpt_processing_tab import create_ckpt_processing_tab

# Create Gradio app
with gr.Blocks(title="RVC WebUI Fork") as gradio_app:
    try:
        repo = git.Repo(search_parent_directories=True)
        version_info = (
            f"## RVC WebUI Fork ({repo.active_branch}) ({repo.head.object.hexsha[:7]})"
        )
    except Exception:
        version_info = "## RVC WebUI Fork"

    gr.Markdown(version_info)

    with gr.Tabs():
        create_inference_tab(app=gradio_app)  # Pass gradio_app, not app
        create_vocal_tab()
        create_train_tab()
        create_ckpt_processing_tab()

# Handle launch based on environment
def start_cloudflared():
    # 下载 cloudflared（如果没有）
    subprocess.run(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared",
        shell=True
    )
    subprocess.run("chmod +x cloudflared", shell=True)

    # 启动隧道
    process = subprocess.Popen(
        "./cloudflared tunnel --url http://127.0.0.1:7860",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # 解析公网链接
    for line in process.stdout:
        if "trycloudflare.com" in line:
            url = re.search(r"(https://[^\s]+trycloudflare\.com)", line)
            if url:
                print("\n🌐 公网地址：", url.group(1), "\n")


if shared.config.iscolab:
    # 启动 Gradio（本地）
    threading.Thread(target=lambda: gradio_app.queue(max_size=1022).launch(share=False)).start()
    # 启动 CF 隧道
    start_cloudflared()
else:
    # For non-Colab, set up FastAPI with Gradio mounted
    gradio_app.queue(max_size=1022)  # Enable queuing

    # Create FastAPI app
    fastapi_app = FastAPI()

    # Dynamic CORS middleware: reflect incoming Origin and support preflight.
    @fastapi_app.middleware("http")
    async def cors_and_private_network(request, call_next):
        origin = request.headers.get("origin")
        request_method = request.headers.get("access-control-request-method")
        request_headers = request.headers.get("access-control-request-headers")

        # Handle preflight directly so mounted Gradio routes do not need custom handlers.
        if request.method == "OPTIONS":
            response = Response(status_code=204)
        else:
            response = await call_next(request)

        response.headers["Access-Control-Allow-Private-Network"] = "true"

        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = (
                request_method or "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = request_headers or "*"

            vary = response.headers.get("Vary")
            response.headers["Vary"] = f"{vary}, Origin" if vary else "Origin"

        return response

    # Mount Gradio app
    gr.mount_gradio_app(fastapi_app, gradio_app, path="/gradio")

    # Redirect root to /gradio
    @fastapi_app.get("/")
    async def redirect_to_gradio():
        return RedirectResponse(url="/gradio")

    # Configure logging
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    print(f"Listening on http://0.0.0.0:{shared.config.listen_port}")

    # Run the server
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=shared.config.listen_port,
        log_level="warning",
        access_log=False,
    )
