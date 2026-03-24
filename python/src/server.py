import asyncio
import json
import logging
import queue
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

logger = logging.getLogger("cactus.server")


# Pydantic models for exchange
class Message(BaseModel):
    model_config = {"extra": "allow"}

    role: str
    content: Optional[Union[str, list]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: str = "function"
    function: ToolChoiceFunction


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoiceObject]] = None
    stream_options: Optional[dict] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list] = None


class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[Usage] = None

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "cactus"
    context_window: Optional[int] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Queue sentinels
class _TokenChunk:
    __slots__ = ('text',)
    def __init__(self, text: str):
        self.text = text


class _StreamDone:
    __slots__ = ('result_json',)
    def __init__(self, result_json: str):
        self.result_json = result_json


class _StreamError:
    __slots__ = ('error',)
    def __init__(self, error: Exception):
        self.error = error


# Helpers
def _flatten_message(msg: Message) -> dict:
    """Normalize a message for the C library."""
    d = {"role": msg.role}
    content = msg.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        d["content"] = "\n".join(parts)
    elif content is not None:
        d["content"] = content
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls is not None:
        d["tool_calls"] = msg.tool_calls
    return d


def translate_tools(
    tools: Optional[List[Tool]],
    tool_choice: Optional[Union[str, ToolChoiceObject]],
) -> tuple:
    """Convert OpenAI-format tools/tool_choice to cactus format.

    Returns (cactus_tools, force_tools).
    """
    if tools is None:
        return None, False

    if isinstance(tool_choice, str) and tool_choice == "none":
        return None, False

    if isinstance(tool_choice, ToolChoiceObject):
        target = tool_choice.function.name
        for t in tools:
            if t.function.name == target:
                return [_tool_to_cactus(t)], True
        return None, False

    cactus_tools = [_tool_to_cactus(t) for t in tools]
    force = isinstance(tool_choice, str) and tool_choice == "required"
    return cactus_tools, force


def _tool_to_cactus(tool: Tool) -> dict:
    """Convert to OpenAI nested format expected by the C library's parse_tools_json."""
    return {
        "type": "function",
        "function": {
            "name": tool.function.name,
            "description": tool.function.description or "",
            "parameters": tool.function.parameters or {},
        },
    }


def make_tool_calls(function_calls: list) -> List[ToolCall]:
    """Convert C library function_calls to OpenAI-format ToolCall objects."""
    out = []
    for fc in function_calls:
        args = fc.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        out.append(ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            function=ToolCallFunction(
                name=fc.get("name", ""),
                arguments=args_str,
            ),
        ))
    return out


def build_options(request: ChatCompletionRequest, force_tools: bool = False) -> dict:
    options = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    if request.stop:
        options["stop"] = request.stop
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if force_tools:
        options["force_tools"] = True
    return options


def _parse_usage(result_data: dict) -> Usage:
    """Extract usage from C library response (field names differ from OpenAI)."""
    return Usage(
        prompt_tokens=result_data.get("prefill_tokens", 0),
        completion_tokens=result_data.get("decode_tokens", 0),
        total_tokens=result_data.get("total_tokens", 0),
    )


def _build_response(result_data: dict, completion_id: str, created: int, model: str) -> ChatCompletionResponse:
    """Build a ChatCompletionResponse from C library result JSON."""
    function_calls = result_data.get("function_calls", [])
    has_tool_calls = len(function_calls) > 0

    message = ChatMessage(
        role="assistant",
        content=result_data.get("response", "") if not has_tool_calls else None,
        tool_calls=make_tool_calls(function_calls) if has_tool_calls else None,
    )

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason="tool_calls" if has_tool_calls else "stop",
            )
        ],
        usage=_parse_usage(result_data),
    )


# Streaming generator
async def _stream_completion(app, body, completion_id, created, cactus_tools, force_tools, include_usage=False):
    """Async generator yielding SSE lines from a streaming completion."""
    token_queue = queue.Queue()
    model = app.state.model
    model_name = app.state.model_name

    messages = [_flatten_message(m) for m in body.messages]
    messages_json = json.dumps(messages)
    options = build_options(body, force_tools)

    def _run_streaming():
        model.reset()
        model.complete_streaming(messages_json, token_queue, options, cactus_tools)

    loop = asyncio.get_event_loop()
    gen_task = loop.run_in_executor(None, _run_streaming)

    try:
        # First chunk: role
        first_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
            )]
        )
        yield first_chunk.to_sse()

        # Poll with sleep to keep event loop responsive
        while True:
            try:
                item = token_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if isinstance(item, _TokenChunk):
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=item.text),
                    )]
                )
                yield chunk.to_sse()

            elif isinstance(item, _StreamDone):
                result_data = json.loads(item.result_json, strict=False)
                if not result_data.get("success", False):
                    raise HTTPException(status_code=500, detail=result_data.get("error", "completion failed"))
                function_calls = result_data.get("function_calls", [])
                has_tool_calls = len(function_calls) > 0

                if has_tool_calls:
                    tool_calls_raw = [
                        {
                            "index": i,
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc["arguments"]) if isinstance(fc.get("arguments"), dict) else str(fc.get("arguments", "")),
                            },
                        }
                        for i, fc in enumerate(function_calls)
                    ]
                    tool_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[StreamChoice(
                            index=0,
                            delta=DeltaMessage(tool_calls=tool_calls_raw),
                        )]
                    )
                    yield tool_chunk.to_sse()

                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[StreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="tool_calls" if has_tool_calls else "stop",
                    )]
                )
                yield final_chunk.to_sse()

                if include_usage:
                    usage_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[],
                        usage=_parse_usage(result_data),
                    )
                    yield usage_chunk.to_sse()

                yield "data: [DONE]\n\n"
                break

            elif isinstance(item, _StreamError):
                raise item.error

    except (asyncio.CancelledError, GeneratorExit):
        model.stop()
        try:
            await asyncio.wait_for(asyncio.shield(gen_task), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass
        raise

    finally:
        while not token_queue.empty():
            try:
                token_queue.get_nowait()
            except queue.Empty:
                break


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    model = getattr(app.state, "model", None)
    if model is not None and hasattr(model, "_handle"):
        from .cactus import cactus_destroy
        cactus_destroy(model._handle)


def create_app(model_name: str, model, context_length: int = 0) -> FastAPI:
    app = FastAPI(
        title="Cactus API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.model = model
    app.state.model_name = model_name
    app.state.context_length = context_length
    app.state.inference_lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        ctx = app.state.context_length or None
        return ModelList(data=[
            ModelInfo(
                id=model_name,
                owned_by="cactus",
                created=int(time.time()),
                context_window=ctx,
            )
        ])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest):
        effective_model = app.state.model_name

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        cactus_tools, force_tools = translate_tools(body.tools, body.tool_choice)

        include_usage = bool(body.stream_options and body.stream_options.get("include_usage"))

        if body.stream:
            async def locked_stream():
                async with app.state.inference_lock:
                    async for chunk in _stream_completion(
                        app, body, completion_id, created, cactus_tools, force_tools, include_usage
                    ):
                        yield chunk

            return StreamingResponse(
                locked_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming path
        try:
            messages = [_flatten_message(m) for m in body.messages]
            messages_json = json.dumps(messages)
            options = build_options(body, force_tools)
            async with app.state.inference_lock:
                await asyncio.to_thread(app.state.model.reset)
                result = await asyncio.to_thread(
                    app.state.model.complete, messages_json,
                    options=options, tools=cactus_tools,
                )
            result_data = json.loads(result, strict=False)
            if not result_data.get("success", False):
                raise HTTPException(status_code=500, detail=result_data.get("error", "completion failed"))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return _build_response(result_data, completion_id, created, effective_model)

    return app


def run_server(model_name: str, model, host: str, port: int, context_length: int = 0):
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app(model_name, model, context_length)

    print(f"Server running at http://{host}:{port}")
    print(f"Health: http://{host}:{port}/health")
    print(f"API: http://{host}:{port}/v1/chat/completions")

    uvicorn.run(app, host=host, port=port)
