from __future__ import annotations

import uvicorn

from .settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "local_constellation.app:app",
        host=settings.manager_host,
        port=settings.manager_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
