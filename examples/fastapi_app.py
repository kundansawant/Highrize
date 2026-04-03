"""
Example: HighRize FastAPI middleware

Run with: uvicorn examples.fastapi_app:app --reload

Then POST to http://localhost:8000/chat with any OpenAI-style body.
GET http://localhost:8000/highrize/stats to see live savings.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from highrize.middleware import HighRizeMiddleware, mount_stats_route
from highrize.cache import CompressionCache

app = FastAPI(title="HighRize Demo")

cache = CompressionCache(backend="disk", path="./highrize_cache")

# Option A: Standard middleware addition
# app.add_middleware(GiveriseMiddleware, model="gpt-4o", cache=cache)

# Option B: Manual instance (useful if you want to call get_stats() directly)
middleware = HighRizeMiddleware(
    app,
    model="gpt-4o",
    provider="openai",
    log_savings=True,
    min_tokens=30,
    cache=cache,
)

# We use the instance as a standard ASGI app wrapper
app.middleware_stack = middleware

mount_stats_route(app, middleware, path="/highrize/stats")


@app.post("/chat")
async def chat(request: Request):
    """
    Your normal route — body is already compressed by middleware.
    Forward to OpenAI (or any provider) from here.
    """
    body = await request.json()
    messages = body.get("messages", [])

    # Example: just echo back the compressed messages + stats
    return JSONResponse({
        "compressed_messages": messages,
        "compression_stats": middleware.get_stats(),
    })


@app.get("/")
async def root():
    return {"message": "HighRize demo running. POST /chat to compress messages."}
