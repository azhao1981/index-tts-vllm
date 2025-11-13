# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

IndexTTS-vLLM 是基于 index-tts 的 TTS (Text-to-Speech) 推理加速项目，使用 vLLM 库重新实现了 GPT 模型的推理。该项目支持 Index-TTS v1.0、v1.5 和 v2.0 版本，显著提升了推理速度和并发性能。

## 核心架构

### 主要模块结构
- `indextts/` - 核心推理模块
  - `infer_vllm.py` / `infer_vllm_v2.py` - 基于 vLLM 的推理引擎 (v1/v1.5 和 v2.0)
  - `gpt/` - GPT 模型相关代码，包含 vLLM 适配版本
  - `s2mel/` - Speech-to-Melody 转换模块，包含 BigVGAN、DAC 等子模块
  - `BigVGAN/` - 声码器模块，用于生成最终音频
  - `utils/` - 工具函数，包括文本处理、特征提取、检查点加载等

### 推理流程
1. 文本预处理和规范化 (`utils/front.py`)
2. GPT 模型推理生成 mel-spectrogram (使用 vLLM 加速)
3. S2Mel 模块转换为 mel-spectrogram
4. BigVGAN 声码器生成最终音频

## 常用开发命令

### 环境设置
```bash
# 安装依赖
pip install -r requirements.txt

# 下载模型权重（推荐使用 ModelScope）
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

### 启动服务

#### WebUI
```bash
# Index-TTS v1.0
python webui.py

# Index-TTS v1.5
python webui.py --version 1.5

# Index-TTS v2.0
python webui_v2.py

# 指定模型目录和其他参数
python webui.py --model_dir /path/to/model --host 0.0.0.0 --port 6006 --gpu_memory_utilization 0.25
```

#### API Server
```bash
# Index-TTS v1.0/1.5
python api_server.py --model_dir /path/to/model --host 0.0.0.0 --port 6006 --gpu_memory_utilization 0.25

# Index-TTS v2.0
python api_server_v2.py --model_dir /path/to/model --host 0.0.0.0 --port 6006 --gpu_memory_utilization 0.25
```

### 模型转换
```bash
# 转换官方权重格式为 vLLM 兼容格式
bash convert_hf_format.sh /path/to/your/model_dir

# 使用 Python 脚本转换
python convert_hf_format.py /path/to/your/model_dir
```

## API 接口

### 主要端点
- `/audio/speech` - OpenAI 兼容的 TTS 接口
- `/audio/voices` - 获取可用的语音角色列表
- `/v1/infer` - 自定义推理接口 (v1/v1.5)
- `/v2/infer` - 自定义推理接口 (v2.0)

### 请求示例
- `api_example.py` - v1/v1.5 API 使用示例
- `api_example_v2.py` - v2.0 API 使用示例

## 重要配置

### GPU 内存设置
- `gpu_memory_utilization` 参数控制 vLLM GPU 内存占用率
- 推荐值：0.25 (约5GB显存)，可支持约16个并发请求
- 可根据硬件配置调整

### 支持的模型版本
- **Index-TTS v1.0**: 使用 `checkpoints/Index-TTS-vLLM`
- **Index-TTS v1.5**: 使用 `checkpoints/Index-TTS-1.5-vLLM`
- **Index-TTS v2.0**: 使用 `checkpoints/IndexTTS-2-vLLM`

### 关键依赖
- `vllm==0.10.2` - GPT 模型推理加速
- `descript-audiotools==0.7.2` - 音频处理
- `gradio` - WebUI 界面
- `fastapi` - API 服务器
- `torchaudio` - 音频处理
- `modelscope` - 模型下载

## 性能特性

### 加速效果 (RTX 4090)
- 单个请求 RTF：≈0.3 → ≈0.1
- GPT 模型 decode 速度：≈90 token/s → ≈280 token/s
- 并发量：约16个并发请求 (5GB显存)

### 特性支持
- 多角色音频混合 (v1/v1.5)
- OpenAI API 兼容接口
- 流式推理支持
- 多语言支持 (中文、英文)

## 开发注意事项

1. **首次启动较慢**：BigVGAN 需要 CUDA 核编译
2. **显存管理**：根据 GPU 内存调整 `gpu_memory_utilization` 参数
3. **模型权重路径**：确保模型文件下载到正确的 `checkpoints/` 目录
4. **版本兼容性**：不同版本使用不同的推理脚本和 WebUI
5. **性能优化**：当前 v2.0 的 S2Mel 推理仍为串行，未来有优化空间