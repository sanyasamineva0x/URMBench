from __future__ import annotations

import asyncio
from typing import Any

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def endpoints(model: str) -> list[dict[str, Any]]:
    response = httpx.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
    response.raise_for_status()
    return response.json()["data"]["endpoints"]


class OpenRouterError(RuntimeError):
    pass


class OpenRouter:
    def __init__(self, api_key: str, attempts: int = 5) -> None:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(300),
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "URMBench"},
        )
        self._attempts = attempts

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        error = ""
        for attempt in range(self._attempts):
            try:
                response = await self._http.post(URL, json=body)
            except httpx.TransportError as e:
                error = repr(e)
            else:
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices") or [{}]
                    if "error" not in data and "error" not in choices[0] and "message" in choices[0]:
                        return data
                    error = str(data.get("error") or choices[0].get("error") or data)
                elif response.status_code in RETRY_STATUS:
                    error = f"{response.status_code}: {response.text[:500]}"
                else:
                    raise OpenRouterError(f"{response.status_code}: {response.text[:2000]}")
            await asyncio.sleep(2**attempt)
        raise OpenRouterError(error)

    async def aclose(self) -> None:
        await self._http.aclose()
