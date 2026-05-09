from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; PSB liquidity data pipeline)"
CHUNK_SIZE = 1024 * 1024


def download_file(url: str, output_path: Path) -> None:
    """Скачивает файл по ссылке и сохраняет его в указанное место"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        with temporary_path.open("wb") as file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)

    temporary_path.replace(output_path)
