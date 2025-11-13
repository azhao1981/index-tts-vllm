import os
import io
import traceback
import json
import struct
from typing import Optional, List
import numpy as np

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn
import argparse
import time
import soundfile as sf

from loguru import logger
logger.add("logs/api_server_v2.1.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

# Import your TTS model
from indextts.infer_vllm_v2 import IndexTTS2

class TTSRequest(BaseModel):
    """TTS请求参数模型"""
    text: str
    spk_audio_path: str
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 1.0
    emo_vec: List[float] = [0] * 8
    emo_text: Optional[str] = None
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('文本内容不能为空')
        return v.strip()
    
    @validator('emo_vec')
    def validate_emo_vec(cls, v, values):
        if values.get('emo_control_method') == 2:
            vec_sum = sum(v)
            if vec_sum > 1.5:
                raise ValueError('情感向量之和不能超过1.5，请调整后重试。')
        return v
    
    @validator('emo_control_method')
    def validate_emo_control_method(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError('情感控制方法必须是0, 1, 2, 3中的一个')
        return v

class TTSStreamRequest(BaseModel):
    """流式TTS请求参数模型"""
    text: str
    spk_audio_path: str
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 1.0
    emo_vec: List[float] = [0] * 8
    emo_text: Optional[str] = None
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('文本内容不能为空')
        return v.strip()

class TTSApp:
    def __init__(self):
        self.tts = None
        self.args = None
    
    def parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=6006)
        parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM")
        parser.add_argument("--is_fp16", action="store_true", default=False)
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
        parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
        self.args = parser.parse_args()
        return self.args
    
    def initialize_model(self):
        """初始化TTS模型"""
        try:
            logger.info("正在初始化TTS模型...")
            self.tts = IndexTTS2(
                model_dir=self.args.model_dir,
                is_fp16=self.args.is_fp16,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                qwenemo_gpu_memory_utilization=self.args.qwenemo_gpu_memory_utilization,
            )
            logger.info("TTS模型初始化完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise
    
    def create_wav_header(self, sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
        """创建WAV文件头"""
        # 计算数据大小（未知，设为最大值）
        data_size = 0xFFFFFFFF  
        file_size = data_size + 36
        
        # WAV文件头结构
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            file_size,
            b'WAVE',
            b'fmt ',
            16,  # fmt chunk大小
            1,   # PCM格式
            num_channels,
            sample_rate,
            sample_rate * num_channels * bits_per_sample // 8,  # 字节率
            num_channels * bits_per_sample // 8,  # 块对齐
            bits_per_sample,
            b'data',
            data_size
        )
        return header
    
    def process_emo_params(self, request: TTSRequest) -> dict:
        """处理情感控制参数"""
        emo_control_method = request.emo_control_method
        
        if emo_control_method == 0:
            return {
                "emo_audio_prompt": None,
                "emo_alpha": 1.0,
                "emo_vector": None,
                "use_emo_text": False,
                "emo_text": None
            }
        elif emo_control_method == 1:
            return {
                "emo_audio_prompt": request.emo_ref_path,
                "emo_alpha": request.emo_weight,
                "emo_vector": None,
                "use_emo_text": False,
                "emo_text": None
            }
        elif emo_control_method == 2:
            return {
                "emo_audio_prompt": None,
                "emo_alpha": request.emo_weight,
                "emo_vector": request.emo_vec,
                "use_emo_text": False,
                "emo_text": None
            }
        elif emo_control_method == 3:
            return {
                "emo_audio_prompt": None,
                "emo_alpha": request.emo_weight,
                "emo_vector": None,
                "use_emo_text": True,
                "emo_text": request.emo_text
            }
        else:
            raise ValueError(f"不支持的情感控制方法: {emo_control_method}")
    
    def create_error_response(self, message: str, status_code: int = 500) -> Response:
        """创建统一的错误响应"""
        error_data = {
            "error": message,
            "timestamp": time.time(),
            "status": "error"
        }
        return Response(
            status_code=status_code,
            content=json.dumps(error_data, ensure_ascii=False),
            media_type="application/json"
        )

# 创建应用实例
tts_app = TTSApp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    args = tts_app.parse_args()
    
    # 创建输出目录
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    # 初始化模型
    tts_app.initialize_model()
    
    try:
        yield
    finally:
        # 清理资源
        if hasattr(tts_app.tts, 'cleanup'):
            logger.info("正在清理TTS模型资源...")
            await tts_app.tts.cleanup()

app = FastAPI(lifespan=lifespan, title="IndexTTS2 API", version="2.1")

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if tts_app.tts is None:
        return Response(
            status_code=503,
            content=json.dumps({
                "status": "unhealthy", 
                "message": "TTS模型未初始化",
                "timestamp": time.time()
            }),
            media_type="application/json"
        )

    return Response(
        status_code=200,
        content=json.dumps({
            "status": "healthy", 
            "message": "服务运行正常",
            "timestamp": time.time()
        }),
        media_type="application/json"
    )

async def audio_chunk_generator(request: TTSStreamRequest):
    """生成音频流的生成器"""
    try:
        logger.info(f"开始生成音频流，文本长度: {len(request.text)}")
        
        # 处理情感参数
        emo_params = tts_app.process_emo_params(request)
        
        # 调用流式推理
        sr, wav_chunks = await tts_app.tts.infer_stream(
            spk_audio_prompt=request.spk_audio_path,
            text=request.text,
            output_path=None,
            use_random=request.emo_random,
            max_text_tokens_per_sentence=int(request.max_text_tokens_per_sentence),
            **emo_params
        )
        
        # 先发送WAV文件头
        wav_header = tts_app.create_wav_header(sr)
        yield wav_header
        
        # 流式返回音频数据块
        chunk_count = 0
        for wav_chunk in wav_chunks:
            # 确保音频数据是正确格式
            if isinstance(wav_chunk, np.ndarray):
                # 转换为字节
                audio_bytes = wav_chunk.tobytes()
                yield audio_bytes
                chunk_count += 1
            else:
                logger.warning(f"跳过非numpy数组的音频块: {type(wav_chunk)}")
        
        logger.info(f"音频流生成完成，共发送 {chunk_count} 个数据块")
        
    except Exception as ex:
        logger.error(f"音频流生成失败: {str(ex)}")
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        error_response = tts_app.create_error_response(f"音频生成错误: {str(ex)}")
        yield error_response.body

@app.post("/tts_url_stream")
async def tts_api_stream(request: TTSStreamRequest):
    """流式TTS推理端点"""
    try:
        logger.info(f"收到流式TTS请求，文本: {request.text[:50]}...")
        
        return StreamingResponse(
            audio_chunk_generator(request),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_audio.wav",
                "X-Audio-Sample-Rate": "22050"  # 根据实际情况调整
            }
        )
        
    except Exception as ex:
        logger.error(f"流式TTS API错误: {str(ex)}")
        return tts_app.create_error_response(f"服务内部错误: {str(ex)}")

@app.post("/tts_url")
async def tts_api_url(request: TTSRequest):
    """非流式TTS推理端点 - 保持向后兼容"""
    try:
        logger.info(f"收到TTS请求，文本: {request.text[:50]}...")
        
        # 处理情感参数
        emo_params = tts_app.process_emo_params(request)
        
        # 调用推理
        sr, wav = await tts_app.tts.infer(
            spk_audio_prompt=request.spk_audio_path,
            text=request.text,
            output_path=None,
            use_random=request.emo_random,
            max_text_tokens_per_sentence=int(request.max_text_tokens_per_sentence),
            **emo_params
        )

        # 生成WAV文件
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        logger.info("TTS推理完成")
        return Response(content=wav_bytes, media_type="audio/wav")

    except ValueError as ve:
        logger.warning(f"参数验证失败: {str(ve)}")
        return tts_app.create_error_response(f"参数错误: {str(ve)}", 400)
    except Exception as ex:
        logger.error(f"TTS API错误: {str(ex)}")
        return tts_app.create_error_response(f"服务内部错误: {str(ex)}")

if __name__ == "__main__":
    # 解析参数并启动服务
    args = tts_app.parse_args()
    
    uvicorn.run(
        app=app, 
        host=args.host, 
        port=args.port,
        log_config=None  # 使用loguru进行日志管理
    )