import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from rich.console import Console
from rich import print as rprint
from demucs.pretrained import get_model
from demucs.audio import save_audio
from torch.cuda import is_available as is_cuda_available
from typing import Optional
from demucs.api import Separator
from demucs.apply import BagOfModels
import gc

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = os.path.join(AUDIO_DIR, "raw.mp3")
BACKGROUND_AUDIO_FILE = os.path.join(AUDIO_DIR, "background.mp3")
VOCAL_AUDIO_FILE = os.path.join(AUDIO_DIR, "vocal.mp3")

class PreloadedSeparator(Separator):
    def __init__(self, model: BagOfModels, shifts: int = 1, overlap: float = 0.25,
                 split: bool = True, segment: Optional[int] = None, jobs: int = 0):
        self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
        device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                            segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)

def demucs_main():
    """这个函数主要把人声和背景音乐分开"""
    if os.path.exists(VOCAL_AUDIO_FILE) and os.path.exists(BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {VOCAL_AUDIO_FILE} and {BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return
    
    console = Console()
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    console.print("🤖 Loading <htdemucs> model...")
    # 加载预训练的htdemucs模型
    model = get_model('htdemucs')
    # 初始化一个PreloadedSeparator实例，使用加载的模型，并设置shifts为1，overlap为0.25
    separator = PreloadedSeparator(model=model, shifts=1, overlap=0.25)

    # 打印正在分离音频的消息
    console.print("🎵 Separating audio...")
    # 使用separator对象处理原始音频文件，得到分离后的音频输出
    _, outputs = separator.separate_audio_file(RAW_AUDIO_FILE)

    # 定义保存音频时使用的参数字典
    kwargs = {"samplerate": model.samplerate, "bitrate": 64, "preset": 2,
              "clip": "rescale", "as_float": False, "bits_per_sample": 16}

    # 打印正在保存人声音轨的消息
    console.print("🎤 Saving vocals track...")
    # 将分离出的人声音频保存到指定文件
    save_audio(outputs['vocals'].cpu(), VOCAL_AUDIO_FILE, **kwargs)

    # 打印正在保存背景音乐的消息
    console.print("🎹 Saving background music...")
    # 计算所有非人声源的音频总和作为背景音乐
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    # 将计算得到的背景音乐保存到指定文件
    save_audio(background.cpu(), BACKGROUND_AUDIO_FILE, **kwargs)

    # Clean up memory
    del outputs, background, model, separator
    gc.collect()
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_main()
