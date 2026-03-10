from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCAL_CONSTELLATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    manager_host: str = "0.0.0.0"
    manager_port: int = 9000
    inference_host: str = "0.0.0.0"
    inference_port: int = 8080
    default_ctx_size: int = 8192
    default_n_gpu_layers: int = 999
    project_root: Path = REPO_ROOT
    llama_cpp_root: Path = REPO_ROOT / "vendor" / "llama.cpp"
    model_root: Path = REPO_ROOT / "data" / "models"
    runtime_root: Path = REPO_ROOT / "data" / "runtime"
    hf_cache_root: Path = REPO_ROOT / "data" / "hf-cache"
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    @property
    def state_path(self) -> Path:
        return self.runtime_root / "llama-server.json"

    @property
    def log_path(self) -> Path:
        return self.runtime_root / "llama-server.log"

    @property
    def manifests_root(self) -> Path:
        return self.model_root / ".manifests"

    def ensure_directories(self) -> None:
        for directory in (
            self.model_root,
            self.runtime_root,
            self.hf_cache_root,
            self.manifests_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_llama_server_binary(self) -> Path:
        candidates = (
            self.llama_cpp_root / "build" / "bin" / "llama-server",
            self.llama_cpp_root / "build" / "bin" / "server",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not find llama-server. Build llama.cpp first.")

    def resolve_quantize_binary(self) -> Path:
        candidates = (
            self.llama_cpp_root / "build" / "bin" / "llama-quantize",
            self.llama_cpp_root / "build" / "bin" / "quantize",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not find the llama.cpp quantize binary. Build llama.cpp first."
        )

    def resolve_convert_script(self) -> Path:
        candidate = self.llama_cpp_root / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            "Could not find convert_hf_to_gguf.py. Clone llama.cpp into vendor/llama.cpp first."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
