from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from .manager import LlamaControlPlane, ManagerError
from .schemas import LocalModel, PullModelRequest, PulledModel, ServerState, StartServerRequest, StopServerResponse
from .settings import Settings, get_settings


def get_manager(settings: Settings = Depends(get_settings)) -> LlamaControlPlane:
    return LlamaControlPlane(settings)


app = FastAPI(
    title="local-constellation",
    version="0.1.0",
    summary="Control plane for downloading models from Hugging Face and serving them with llama.cpp.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models", response_model=list[LocalModel])
def list_models(manager: LlamaControlPlane = Depends(get_manager)) -> list[LocalModel]:
    return manager.list_models()


@app.post("/models/pull", response_model=PulledModel)
def pull_model(
    request: PullModelRequest,
    manager: LlamaControlPlane = Depends(get_manager),
) -> PulledModel:
    try:
        return manager.pull_model(request)
    except ManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/server", response_model=ServerState)
def server_status(manager: LlamaControlPlane = Depends(get_manager)) -> ServerState:
    return manager.server_status()


@app.post("/server/start", response_model=ServerState)
def start_server(
    request: StartServerRequest,
    manager: LlamaControlPlane = Depends(get_manager),
) -> ServerState:
    try:
        return manager.start_server(request)
    except ManagerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/server/stop", response_model=StopServerResponse)
def stop_server(manager: LlamaControlPlane = Depends(get_manager)) -> StopServerResponse:
    return manager.stop_server()
