from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .schemas import (
    LocalModel,
    PullModelRequest,
    PulledModel,
    ServerState,
    StartServerRequest,
    StopServerResponse,
)
from .settings import Settings


class ManagerError(RuntimeError):
    pass


class LlamaControlPlane:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_directories()

    def list_models(self) -> list[LocalModel]:
        manifests = self._load_manifests()
        models: list[LocalModel] = []
        seen_paths: set[Path] = set()

        for manifest_path, payload in manifests.items():
            gguf_path = Path(payload["gguf_path"])
            seen_paths.add(gguf_path)
            models.append(
                LocalModel(
                    name=gguf_path.name,
                    gguf_path=str(gguf_path),
                    repo_id=payload.get("repo_id"),
                    revision=payload.get("revision"),
                    source_type=payload.get("source_type", "unknown"),
                    manifest_path=str(manifest_path),
                    quantization=payload.get("quantization"),
                    created_at=payload.get("created_at"),
                )
            )

        for gguf_path in sorted(self.settings.model_root.rglob("*.gguf")):
            if gguf_path in seen_paths:
                continue
            models.append(LocalModel(name=gguf_path.name, gguf_path=str(gguf_path)))

        models.sort(key=lambda item: item.name.lower())
        return models

    def pull_model(self, request: PullModelRequest) -> PulledModel:
        repo_slug = self._repo_slug(request.repo_id)
        target_root = self.settings.model_root / repo_slug / request.revision
        target_root.mkdir(parents=True, exist_ok=True)

        token = request.token or self.settings.hf_token
        api = HfApi(token=token)
        info = api.model_info(repo_id=request.repo_id, revision=request.revision, token=token)
        siblings = sorted(item.rfilename for item in info.siblings)

        selected_gguf = self._select_gguf_file(
            siblings=siblings,
            filename=request.filename,
            prefer_gguf=request.prefer_gguf,
        )
        if selected_gguf:
            gguf_path = self._download_gguf(
                repo_id=request.repo_id,
                revision=request.revision,
                token=token,
                target_root=target_root,
                filename=selected_gguf,
            )
            manifest_path = self._write_manifest(
                target_root=target_root,
                gguf_path=gguf_path,
                repo_id=request.repo_id,
                revision=request.revision,
                source_type="gguf",
                quantization=None,
                model_name=request.model_name,
                source_path=None,
            )
            return PulledModel(
                repo_id=request.repo_id,
                revision=request.revision,
                source_type="gguf",
                gguf_path=str(gguf_path),
                manifest_path=str(manifest_path),
                model_name=request.model_name,
            )

        source_root = target_root / "source"
        source_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=request.repo_id,
            revision=request.revision,
            token=token,
            local_dir=source_root,
            cache_dir=self.settings.hf_cache_root,
        )

        converted_path = self._convert_safetensors(
            source_root=source_root,
            target_root=target_root,
            request=request,
        )
        final_path = converted_path
        if request.quantization:
            final_path = self._quantize_gguf(
                source_path=converted_path,
                target_root=target_root,
                quantization=request.quantization,
            )
            if not request.keep_intermediate and converted_path.exists():
                converted_path.unlink()

        manifest_path = self._write_manifest(
            target_root=target_root,
            gguf_path=final_path,
            repo_id=request.repo_id,
            revision=request.revision,
            source_type="safetensors",
            quantization=request.quantization,
            model_name=request.model_name,
            source_path=source_root,
        )
        return PulledModel(
            repo_id=request.repo_id,
            revision=request.revision,
            source_type="safetensors",
            gguf_path=str(final_path),
            manifest_path=str(manifest_path),
            source_path=str(source_root),
            quantization=request.quantization,
            model_name=request.model_name,
        )

    def server_status(self) -> ServerState:
        state = self._read_state()
        if not state:
            return ServerState(running=False)

        pid = state.get("pid")
        if not pid or not self._is_running(pid):
            self._clear_state()
            return ServerState(running=False)

        return ServerState(
            running=True,
            pid=pid,
            model_path=state.get("model_path"),
            command=state.get("command", []),
            host=state.get("host"),
            port=state.get("port"),
            api_base=state.get("api_base"),
            started_at=state.get("started_at"),
            log_path=state.get("log_path"),
        )

    def start_server(self, request: StartServerRequest) -> ServerState:
        current = self.server_status()
        if current.running:
            raise ManagerError(f"llama-server is already running with pid {current.pid}.")

        model_path = self._resolve_model_path(request.model)
        llama_server = self.settings.resolve_llama_server_binary()
        command = [
            str(llama_server),
            "-m",
            str(model_path),
            "--host",
            request.host,
            "--port",
            str(request.port),
            "-c",
            str(request.ctx_size),
            "-ngl",
            str(request.n_gpu_layers),
        ]

        if request.alias:
            command.extend(["--alias", request.alias])
        if request.threads is not None:
            command.extend(["-t", str(request.threads)])
        if request.batch_size is not None:
            command.extend(["-b", str(request.batch_size)])
        if request.ubatch_size is not None:
            command.extend(["-ub", str(request.ubatch_size)])
        if request.parallel is not None:
            command.extend(["-np", str(request.parallel)])
        if request.tensor_split:
            command.extend(["-ts", request.tensor_split])
        if request.metrics:
            command.append("--metrics")
        if request.extra_args:
            command.extend(request.extra_args)

        log_handle = self.settings.log_path.open("a", encoding="utf-8")
        log_handle.write(
            f"\n[{datetime.now(tz=UTC).isoformat()}] starting llama-server for {model_path}\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=self.settings.llama_cpp_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()

        if not self._wait_for_port(host=request.host, port=request.port, pid=process.pid):
            process.terminate()
            raise ManagerError(
                f"llama-server failed to bind to {request.host}:{request.port}. Check {self.settings.log_path}."
            )

        api_host = "127.0.0.1" if request.host in {"0.0.0.0", "::"} else request.host

        state = {
            "pid": process.pid,
            "model_path": str(model_path),
            "command": command,
            "host": request.host,
            "port": request.port,
            "api_base": f"http://{api_host}:{request.port}/v1",
            "started_at": datetime.now(tz=UTC).isoformat(),
            "log_path": str(self.settings.log_path),
        }
        self.settings.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return self.server_status()

    def stop_server(self) -> StopServerResponse:
        state = self._read_state()
        if not state or not state.get("pid"):
            self._clear_state()
            return StopServerResponse(stopped=False)

        pid = int(state["pid"])
        if not self._is_running(pid):
            self._clear_state()
            return StopServerResponse(stopped=False, pid=pid)

        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not self._is_running(pid):
                self._clear_state()
                return StopServerResponse(stopped=True, pid=pid)
            time.sleep(0.25)

        os.kill(pid, signal.SIGKILL)
        self._clear_state()
        return StopServerResponse(stopped=True, pid=pid)

    def _select_gguf_file(
        self,
        siblings: list[str],
        filename: str | None,
        prefer_gguf: bool,
    ) -> str | None:
        ggufs = [item for item in siblings if item.lower().endswith(".gguf")]
        if filename:
            if filename not in siblings:
                raise ManagerError(f"`{filename}` is not present in the repo.")
            if not filename.lower().endswith(".gguf"):
                return None
            return filename

        if not prefer_gguf or not ggufs:
            return None
        if len(ggufs) > 1:
            raise ManagerError(
                "This repo exposes multiple GGUF files. Specify `filename` to choose one."
            )
        return ggufs[0]

    def _download_gguf(
        self,
        repo_id: str,
        revision: str,
        token: str | None,
        target_root: Path,
        filename: str,
    ) -> Path:
        download_root = target_root / "raw"
        download_root.mkdir(parents=True, exist_ok=True)
        gguf_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                token=token,
                local_dir=download_root,
                cache_dir=self.settings.hf_cache_root,
            )
        )
        return gguf_path.resolve()

    def _convert_safetensors(
        self,
        source_root: Path,
        target_root: Path,
        request: PullModelRequest,
    ) -> Path:
        convert_script = self.settings.resolve_convert_script()
        gguf_root = target_root / "gguf"
        gguf_root.mkdir(parents=True, exist_ok=True)

        model_basename = request.model_name or source_root.parent.parent.name
        output_path = gguf_root / f"{model_basename}-{request.outtype}.gguf"
        command = [
            sys.executable,
            str(convert_script),
            str(source_root),
            "--outfile",
            str(output_path),
            "--outtype",
            request.outtype,
        ]
        if request.model_name:
            command.extend(["--model-name", request.model_name])

        self._run_command(
            command=command,
            cwd=self.settings.llama_cpp_root,
            error_context="convert safetensors to GGUF",
        )
        return output_path.resolve()

    def _quantize_gguf(
        self,
        source_path: Path,
        target_root: Path,
        quantization: str,
    ) -> Path:
        quantize_bin = self.settings.resolve_quantize_binary()
        gguf_root = target_root / "gguf"
        gguf_root.mkdir(parents=True, exist_ok=True)
        output_path = gguf_root / f"{source_path.stem}-{quantization.lower()}.gguf"
        command = [str(quantize_bin), str(source_path), str(output_path), quantization]
        self._run_command(
            command=command,
            cwd=self.settings.llama_cpp_root,
            error_context=f"quantize GGUF with {quantization}",
        )
        return output_path.resolve()

    def _run_command(self, command: list[str], cwd: Path, error_context: str) -> None:
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or "No output captured."
            raise ManagerError(f"Failed to {error_context}: {detail}")

    def _resolve_model_path(self, model: str) -> Path:
        candidate = Path(model)
        if candidate.exists():
            return candidate.resolve()

        direct_candidate = (self.settings.model_root / model).resolve()
        if direct_candidate.exists():
            return direct_candidate

        matches = [item for item in self.settings.model_root.rglob("*.gguf") if item.name == model]
        if not matches:
            raise ManagerError(f"Could not find model `{model}`.")
        if len(matches) > 1:
            raise ManagerError(f"`{model}` matches multiple GGUF files. Use an absolute path instead.")
        return matches[0].resolve()

    def _write_manifest(
        self,
        target_root: Path,
        gguf_path: Path,
        repo_id: str,
        revision: str,
        source_type: str,
        quantization: str | None,
        model_name: str | None,
        source_path: Path | None,
    ) -> Path:
        manifest_name = f"{self._repo_slug(repo_id)}-{revision}.json"
        manifest_path = self.settings.manifests_root / manifest_name
        payload: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision,
            "source_type": source_type,
            "gguf_path": str(gguf_path),
            "quantization": quantization,
            "model_name": model_name,
            "target_root": str(target_root),
            "created_at": datetime.now(tz=UTC).isoformat(),
        }
        if source_path:
            payload["source_path"] = str(source_path)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path

    def _load_manifests(self) -> dict[Path, dict[str, Any]]:
        manifests: dict[Path, dict[str, Any]] = {}
        for manifest_path in sorted(self.settings.manifests_root.glob("*.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            manifests[manifest_path] = payload
        return manifests

    def _read_state(self) -> dict[str, Any] | None:
        if not self.settings.state_path.exists():
            return None
        try:
            return json.loads(self.settings.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _clear_state(self) -> None:
        if self.settings.state_path.exists():
            self.settings.state_path.unlink()

    def _repo_slug(self, repo_id: str) -> str:
        return repo_id.replace("/", "--")

    def _is_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _wait_for_port(self, host: str, port: int, pid: int) -> bool:
        deadline = time.monotonic() + 30
        host_to_check = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        while time.monotonic() < deadline:
            if not self._is_running(pid):
                return False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                if sock.connect_ex((host_to_check, port)) == 0:
                    return True
            time.sleep(0.25)
        return False
