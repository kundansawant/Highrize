"""
highrize.middleware — FastAPI / Starlette middleware for automatic compression.

Intercepts any POST request that has a JSON body containing a "messages"
key (OpenAI-style or Anthropic-style), compresses the content in-place,
and forwards the modified request to your route handler.

Usage:
    from fastapi import FastAPI
    from highrize.middleware import HighRizeMiddleware

    app = FastAPI()
    app.add_middleware(
        HighRizeMiddleware,
        model="gpt-4o",
        provider="openai",
        log_savings=True,     # prints savings per request to stdout
    )

    # Your routes are unchanged — compression is automatic
    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()  # already compressed
        messages = body["messages"]
        ...

Also includes a /highrize/stats endpoint you can mount for a live dashboard.
"""

import json
import time
import logging
from typing import Callable, Optional

logger = logging.getLogger("highrize.middleware")


class HighRizeMiddleware:
    """
    ASGI middleware that auto-compresses AI request payloads.

    Targets any request where the JSON body contains a "messages" key.
    Leaves all other requests untouched.

    Args:
        app:          Your ASGI app.
        model:        Model name for cost estimation.
        provider:     "openai" | "anthropic" | "gemini"
        log_savings:  Print savings per request.
        skip_paths:   List of path prefixes to skip (e.g. ["/health", "/metrics"])
        min_tokens:   Only compress if original exceeds this token count.
        cache:        Pass a CompressionCache instance for caching.
        on_compress:  Optional async callback(original_tokens, compressed_tokens, path)
    """

    def __init__(
        self,
        app,
        model: str = "default",
        provider: str = "openai",
        log_savings: bool = True,
        skip_paths: Optional[list] = None,
        min_tokens: int = 50,
        cache=None,
        on_compress: Optional[Callable] = None,
        **highrize_kwargs,
    ):
        self.app = app
        self.log_savings = log_savings
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.min_tokens = min_tokens
        self.cache = cache
        self.on_compress = on_compress

        # Lazy import to avoid hard dep on FastAPI at module level
        from highrize.core import HighRize
        self.tp = HighRize(model=model, provider=provider, **highrize_kwargs)

        # Lifetime stats
        self._total_requests = 0
        self._compressed_requests = 0
        self._total_tokens_original = 0
        self._total_tokens_compressed = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        # Skip non-POST or excluded paths
        if method != "POST" or any(path.startswith(p) for p in self.skip_paths):
            await self.app(scope, receive, send)
            return

        # Buffer the request body
        body_chunks = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)

        body_bytes = b"".join(body_chunks)

        # Try to parse JSON and find messages
        try:
            payload = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON — pass through unchanged
            await self._replay(scope, receive, send, body_bytes)
            return

        if "messages" not in payload:
            await self._replay(scope, receive, send, body_bytes)
            return

        # --- Compress ---
        t0 = time.perf_counter()
        self._total_requests += 1

        original_messages = payload["messages"]
        system = payload.get("system")  # Anthropic-style system prompt

        if self.cache:
            # Use cache for each message's content
            from highrize.models import Modality
            compressed_messages = []
            for msg in original_messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    result = self.cache.get_or_compress(self.tp, content, Modality.TEXT)
                    compressed_messages.append({**msg, "content": result.compressed})
                else:
                    compressed_messages.append(msg)
        else:
            compressed_messages, _ = self.tp.compress_messages(original_messages)

        if system:
            from highrize.models import Modality
            sys_result = self.tp.compress(system, Modality.TEXT)
            payload["system"] = sys_result.compressed

        payload["messages"] = compressed_messages
        new_body = json.dumps(payload).encode("utf-8")

        # Track stats
        orig_t = self.tp.report.total_original_tokens
        comp_t = self.tp.report.total_compressed_tokens

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._compressed_requests += 1

        if self.log_savings:
            pct = self.tp.report.savings_pct
            logger.info(
                f"highrize [{path}] "
                f"{self.tp.report.tokens_saved:,} tokens saved "
                f"({pct}%) in {elapsed_ms:.1f}ms"
            )

        if self.on_compress:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.on_compress):
                    await self.on_compress(orig_t, comp_t, path)
                else:
                    self.on_compress(orig_t, comp_t, path)
            except Exception:
                pass

        # Rebuild receive with compressed body
        async def new_receive():
            return {"type": "http.request", "body": new_body, "more_body": False}

        # Update Content-Length header
        headers = list(scope.get("headers", []))
        headers = [
            (k, str(len(new_body)).encode() if k == b"content-length" else v)
            for k, v in headers
        ]
        scope = {**scope, "headers": headers}

        await self.app(scope, new_receive, send)

    async def _replay(self, scope, receive, send, body: bytes):
        """Replay original body unchanged."""
        async def _receive():
            return {"type": "http.request", "body": body, "more_body": False}
        await self.app(scope, _receive, send)

    def get_stats(self) -> dict:
        """Returns lifetime compression stats."""
        return {
            "total_requests": self._total_requests,
            "compressed_requests": self._compressed_requests,
            **{
                "tokens_original": self.tp.report.total_original_tokens,
                "tokens_compressed": self.tp.report.total_compressed_tokens,
                "tokens_saved": self.tp.report.tokens_saved,
                "savings_pct": self.tp.report.savings_pct,
                "cost_saved_usd": self.tp.report.cost_saved_usd,
            },
        }


def mount_stats_route(app, middleware_instance, path: str = "/highrize/stats"):
    """
    Mount a live stats endpoint on your FastAPI app.

    Usage:
        from highrize.middleware import HighRizeMiddleware, mount_stats_route

        mw = HighRizeMiddleware(app, model="gpt-4o", log_savings=True)
        app.add_middleware(lambda a: mw)  # or app.middleware_stack
        mount_stats_route(app, mw)

        # GET /highrize/stats → JSON savings dashboard
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        @app.get(path, tags=["highrize"])
        async def highrize_stats():
            """Live token compression savings dashboard."""
            return JSONResponse(middleware_instance.get_stats())

    except ImportError:
        logger.warning("FastAPI not installed — stats route not mounted.")
