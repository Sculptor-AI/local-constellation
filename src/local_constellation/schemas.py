from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PullModelRequest(BaseModel):
    repo_id: str = Field(description="Hugging Face repo id.")
    filename: str | None = Field(
        default=None,
        description="Specific file to download from the repo. Use this for GGUF repos with multiple files.",
    )
    revision: str = Field(default="main")
    token: str | None = Field(default=None, description="Optional Hugging Face token.")
    prefer_gguf: bool = Field(
        default=True,
        description="Prefer an existing GGUF before falling back to safetensors conversion.",
    )
    outtype: Literal["auto", "f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0"] = Field(
        default="auto",
        description="Base precision for convert_hf_to_gguf.py.",
    )
    quantization: str | None = Field(
        default=None,
        description="Optional llama.cpp quantization type, for example Q4_K_M.",
    )
    keep_intermediate: bool = Field(
        default=False,
        description="Keep the unquantized GGUF when quantization is requested.",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional user-facing label persisted in metadata.",
    )


class PulledModel(BaseModel):
    repo_id: str
    revision: str
    source_type: Literal["gguf", "safetensors"]
    gguf_path: str
    manifest_path: str
    source_path: str | None = None
    quantization: str | None = None
    model_name: str | None = None


class LocalModel(BaseModel):
    name: str
    gguf_path: str
    repo_id: str | None = None
    revision: str | None = None
    source_type: Literal["gguf", "safetensors", "unknown"] = "unknown"
    manifest_path: str | None = None
    quantization: str | None = None
    created_at: str | None = None


class StartServerRequest(BaseModel):
    model: str = Field(description="Absolute GGUF path or a unique file name under the models directory.")
    alias: str | None = Field(default=None)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    ctx_size: int = Field(default=8192)
    n_gpu_layers: int = Field(default=999)
    threads: int | None = Field(default=None)
    batch_size: int | None = Field(default=None)
    ubatch_size: int | None = Field(default=None)
    parallel: int | None = Field(default=None)
    tensor_split: str | None = Field(default=None)
    metrics: bool = Field(default=True)
    extra_args: list[str] = Field(default_factory=list)


class ServerState(BaseModel):
    running: bool
    pid: int | None = None
    model_path: str | None = None
    command: list[str] = Field(default_factory=list)
    host: str | None = None
    port: int | None = None
    api_base: str | None = None
    started_at: str | None = None
    log_path: str | None = None


class StopServerResponse(BaseModel):
    stopped: bool
    pid: int | None = None
