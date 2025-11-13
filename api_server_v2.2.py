import os
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import time
import soundfile as sf
from typing import List, Optional

from loguru import logger
logger.add("logs/api_server_v2.1.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return Response(
            status_code=503,
            content='{"status": "unhealthy", "message": "TTS model not initialized"}',
            media_type="application/json"
        )

    return Response(
        status_code=200,
        content=f'{{"status": "healthy", "message": "Service is running", "timestamp": {time.time()}}}',
        media_type="application/json"
    )

async def audio_chunk_generator(data: dict):
    """生成音频流的生成器"""
    try:
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts

        # 情感控制参数处理
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value

        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        elif emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                yield b'{"error": "情感向量之和不能超过1.5，请调整后重试。"}'
                return
        else:
            vec = None

        # 模拟流式音频生成 - 实际需要修改 IndexTTS2.infer_stream()
        # 这里假设 infer_stream 返回音频块生成器
        sr, wav_chunks = await tts.infer_stream(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=None,
            emo_audio_prompt=emo_ref_path,
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3),
            emo_text=emo_text,
            use_random=emo_random,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence)
        )

        # 流式返回音频块
        for wav_chunk in wav_chunks:
            with io.BytesIO() as chunk_buffer:
                sf.write(chunk_buffer, wav_chunk, sr, format='WAV')
                yield chunk_buffer.getvalue()

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        yield f'{{"error": "{tb_str}"}}'.encode()

@app.post("/tts_url_stream", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_stream(request: Request):
    """流式TTS推理端点"""
    try:
        data = await request.json()

        # 检查是否是错误响应
        async for chunk in audio_chunk_generator(data):
            if chunk.startswith(b'{"error":'):
                return Response(
                    status_code=500,
                    content=chunk,
                    media_type="application/json"
                )
            yield chunk

        return StreamingResponse(
            audio_chunk_generator(data),
            media_type="audio/wav"
        )

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return Response(
            status_code=500,
            content=f'{{"error": "{tb_str}"}}',
            media_type="application/json"
        )

@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    """非流式TTS推理端点 - 保持向后兼容"""
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return Response(
                    status_code=500,
                    content='{"status": "error", "error": "情感向量之和不能超过1.5，请调整后重试。"}',
                    media_type="application/json"
                )
        else:
            vec = None

        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=None,
            emo_audio_prompt=emo_ref_path,
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3),
            emo_text=emo_text,
            use_random=emo_random,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence)
        )

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return Response(
            status_code=500,
            content=f'{{"error": "{tb_str}"}}',
            media_type="application/json"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM")
    parser.add_argument("--is_fp16", action="store_true", default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    args = parser.parse_args()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)