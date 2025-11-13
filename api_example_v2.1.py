from dataclasses import asdict, dataclass
import os
from typing import List, Optional
import requests
import time
from datetime import datetime

SERVER_PORT = 6006
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 流式端点
stream_url = f"http://0.0.0.0:{SERVER_PORT}/tts_url_stream"
# 非流式端点（保持兼容）
normal_url = f"http://0.0.0.0:{SERVER_PORT}/tts_url"

@dataclass
class IndexTTS2RequestData:
    text: str
    spk_audio_path: str
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 1.0
    emo_vec: List[float] = None
    emo_text: Optional[str] = None
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120

    def __post_init__(self):
        if self.emo_vec is None:
            self.emo_vec = [0.0] * 8

    def to_dict(self) -> str:
        return asdict(self)

text = """
沃丰零食品牌创立于2012年，截至2025年已成立13年。截至目前在国内签约门店超8000家。门店分布于全国28个省级区域、300多个城市，主要集中在广东、广西、海南、河南、河北等地。沃丰零食是聚焦新中式风味的零食连锁品牌，主打经典中式、创新、健康轻食等多系列零食，融合传统风味与现代需求，提供多元健康的零食选择。
今年对加盟的扶持力度较大，目前有减免加盟费的政策；减免费用包含品牌使用费30000元、培训费20000元、设计费3000元、系统使用费2000元。具体政策需根据您的实际情况确定。
"""

# 1. 流式请求测试
print("=== 流式请求测试 ===")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
begin = time.time()

data = IndexTTS2RequestData(
    text=text,
    spk_audio_path="assets/jay_promptvn.wav"
)

# 使用 stream=True 进行流式请求
response = requests.post(stream_url, json=data.to_dict(), stream=True)

if response.status_code == 200:
    with open(os.path.join(output_dir, "output_stream.wav"), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("流式音频文件已保存: output_stream.wav")
else:
    print(f"流式请求失败: {response.status_code}")
    print(response.text)

end = time.time()
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"流式请求总耗时: {end - begin:.2f} 秒")
print()

# 2. 非流式请求测试（对比）
print("=== 非流式请求测试 ===")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
begin = time.time()

response = requests.post(normal_url, json=data.to_dict())

if response.status_code == 200:
    with open(os.path.join(output_dir, "output_normal.wav"), "wb") as f:
        f.write(response.content)
    print("非流式音频文件已保存: output_normal.wav")
else:
    print(f"非流式请求失败: {response.status_code}")
    print(response.text)

end = time.time()
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"非流式请求总耗时: {end - begin:.2f} 秒")