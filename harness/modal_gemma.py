"""Modal app: hold + serve the AU-tuned Gemma 4 26B-A4B.

  modal run modal_gemma.py::prepare   # download base from HF, merge adapter, build GGUFs (CPU)
  modal deploy modal_gemma.py         # deploy tuned/base OpenAI-compatible endpoints (L40S)

Artifacts live in the 'au-gemma4' Volume:
  /adapter            HF PEFT adapter (uploaded from local)
  /merge_adapter.py   merge script (uploaded from local)
  /base               HF snapshot of google/gemma-4-26B-A4B-it
  /merged             merged BF16 HF model (for future vLLM serving)
  /tuned-q8.gguf, /base-q8.gguf
"""
import modal

app = modal.App("au-gemma4")
vol = modal.Volume.from_name("au-gemma4")

prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "safetensors", "numpy", "transformers>=4.58",
                 "sentencepiece", "gguf", "mistral-common",
                 "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands("git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp")
)

serve_image = (modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
               .entrypoint([]))


@app.function(image=prep_image, volumes={"/vol": vol}, cpu=8, memory=98304, timeout=7200)
def prepare(adapter_subdir: str = "adapter", out_prefix: str = "tuned"):
    import subprocess, os
    from huggingface_hub import snapshot_download

    base = snapshot_download("google/gemma-4-26B-A4B-it", local_dir="/vol/base",
                             ignore_patterns=["*.gguf"])
    print("base at", base)
    vol.commit()

    merged = f"/vol/merged-{out_prefix}" if out_prefix != "tuned" else "/vol/merged"
    if not os.path.exists(f"{merged}/model.safetensors.index.json"):
        subprocess.run(["python", "/vol/merge_adapter.py", f"/vol/{adapter_subdir}",
                        "/vol/base", merged], check=True)
        vol.commit()

    for src, out in ((merged, f"/vol/{out_prefix}-q8.gguf"), ("/vol/base", "/vol/base-q8.gguf")):
        if not os.path.exists(out):
            subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", src,
                            "--outfile", out, "--outtype", "q8_0"], check=True)
            vol.commit()
    print("PREPARE_DONE")
    return "ok"


def _server_cmd(gguf):
    import subprocess
    return subprocess.Popen(
        ["/app/llama-server", "-m", gguf, "--host", "0.0.0.0", "--port", "8080",
         "--ctx-size", "24576", "-ngl", "999", "--jinja", "--parallel", "2"])


@app.function(image=serve_image, volumes={"/vol": vol}, gpu="L40S",
              scaledown_window=240, timeout=3600)
@modal.web_server(port=8080, startup_timeout=600)
def tuned():
    _server_cmd("/vol/tuned-q8.gguf")


@app.function(image=serve_image, volumes={"/vol": vol}, gpu="L40S",
              scaledown_window=240, timeout=3600)
@modal.web_server(port=8080, startup_timeout=600)
def base():
    _server_cmd("/vol/base-q8.gguf")


@app.function(image=serve_image, volumes={"/vol": vol}, gpu="L40S",
              scaledown_window=240, timeout=3600)
@modal.web_server(port=8080, startup_timeout=600)
def tuned_v2():
    _server_cmd("/vol/tuned-v2-q8.gguf")


@app.function(image=serve_image, volumes={"/vol": vol}, gpu="L40S",
              scaledown_window=240, timeout=3600)
@modal.web_server(port=8080, startup_timeout=600)
def tuned_v3():
    _server_cmd("/vol/tuned-v3-q8.gguf")


@app.function(image=serve_image, volumes={"/vol": vol}, gpu="L40S",
              scaledown_window=240, timeout=3600)
@modal.web_server(port=8080, startup_timeout=600)
def tuned_v4():
    _server_cmd("/vol/tuned-v4-q8.gguf")


@app.function(image=serve_image, volumes={"/vol": vol}, gpu="L40S",
              scaledown_window=240, timeout=3600)
@modal.web_server(port=8080, startup_timeout=600)
def tuned_v5():
    _server_cmd("/vol/tuned-v5-q8.gguf")


@app.function(image=serve_image, volumes={"/vol": vol}, gpu="L40S",
              scaledown_window=240, timeout=3600)
@modal.web_server(port=8080, startup_timeout=600)
def tuned_v6a():
    _server_cmd("/vol/tuned-v6a-q8.gguf")
