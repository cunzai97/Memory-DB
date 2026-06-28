"""Embedding client — calls local llama.cpp /v1/embeddings API."""

import os

import httpx

# Default: your llama.cpp embedding server on localhost:8081
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8081/v1/embeddings")


async def encode(text: str, url: str | None = None, max_retries: int = 2) -> list[float]:
    """Encode a single string into a vector via the local embedding API."""
    api_url = url or EMBEDDING_API_URL
    # trust_env=False: don't let system proxy env vars break localhost connections
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                resp = await client.post(
                    api_url,
                    json={"input": [text], "model": ""},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["data"][0]["embedding"]
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < max_retries:
                continue  # retry on transient errors
            raise
        except Exception:
            raise  # non-transient errors should fail fast
    raise RuntimeError(f"Failed to encode after {max_retries + 1} attempts: {last_error}")


async def health_check(url: str | None = None) -> tuple[bool, str]:
    """Check if embedding API is reachable. Returns (ok, message)."""
    api_url = url or EMBEDDING_API_URL
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            resp = await client.post(
                api_url,
                json={"input": ["test"], "model": ""},
            )
            resp.raise_for_status()
            data = resp.json()
            dim = len(data["data"][0]["embedding"])
            return True, f"OK (dimension={dim})"
    except httpx.ConnectError as e:
        return False, f"Connection failed: {e}"
    except httpx.TimeoutException:
        return False, "Timeout (5s)"
    except Exception as e:
        return False, f"Error: {e}"
