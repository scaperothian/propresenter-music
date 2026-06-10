"""Live audio capture: microphone (sounddevice) and file-based (for testing).

Both sources produce fixed-size chunks of float32 mono audio at TARGET_SR.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchaudio

from .config import CHUNK_SEC, TARGET_SR


class AudioCapture:
    """Abstract base: iterates over fixed-size PCM chunks."""

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """Yield (chunk_f32_mono_24kHz, chunk_start_sec_wallclock)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class MicCapture(AudioCapture):
    """
    Capture live microphone audio via sounddevice.

    The stream opens at the device's native sample rate and each block is
    resampled to TARGET_SR — many input devices refuse non-native rates.

    Backpressure: if alignment falls behind the capture rate, every block
    queued in the meantime is drained and yielded as ONE combined chunk.
    The aligner's audio ring buffer accepts arbitrary chunk sizes, so this
    costs one skipped update instead of permanently growing latency.

    Args:
        chunk_sec:   Duration of each capture block in seconds.
        device:      sounddevice device index or name (None = default input).
    """

    def __init__(self, chunk_sec: float = CHUNK_SEC, device=None) -> None:
        self.chunk_sec = chunk_sec
        self.device = device
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._status_warned = False

    def _callback(self, indata, frames, time_info, status):
        if status and not self._status_warned:
            print(f"[mic] stream status: {status}", flush=True)
            self._status_warned = True
        self._q.put(indata[:, 0].copy())  # take first channel (mono)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        import sounddevice as sd

        dev_info = sd.query_devices(self.device, kind="input")
        native_sr = int(dev_info["default_samplerate"])
        blocksize = int(self.chunk_sec * native_sr)
        print(f"[mic] {dev_info['name']}  {native_sr} Hz → {TARGET_SR} Hz, "
              f"{self.chunk_sec * 1000:.0f}ms blocks")

        t_start = time.monotonic()
        samples_out = 0
        with sd.InputStream(
            samplerate=native_sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=self.device,
            callback=self._callback,
        ):
            while True:
                parts = [self._q.get()]
                while True:  # drain everything that accumulated while busy
                    try:
                        parts.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                block = np.concatenate(parts) if len(parts) > 1 else parts[0]
                if native_sr != TARGET_SR:
                    block = torchaudio.functional.resample(
                        torch.from_numpy(block), native_sr, TARGET_SR
                    ).numpy()
                wall_t = t_start + samples_out / TARGET_SR
                samples_out += len(block)
                yield block.astype(np.float32, copy=False), wall_t


class FileCapture(AudioCapture):
    """
    Simulate real-time audio capture from a WAV/FLAC/MP3 file.

    Reads the file at TARGET_SR, then yields chunks at the real-time
    rate (*realtime=True*) or as fast as possible (*realtime=False*).
    The *start_offset_sec* parameter simulates beginning playback
    mid-song (for testing the cold-start alignment scenario).

    Args:
        audio_path:       path to audio file
        chunk_sec:        chunk duration in seconds
        realtime:         if True, sleep between chunks to match wall time
        start_offset_sec: start this many seconds into the file
    """

    def __init__(
        self,
        audio_path: Path,
        chunk_sec: float = CHUNK_SEC,
        realtime: bool = True,
        start_offset_sec: float = 0.0,
    ) -> None:
        self.audio_path = Path(audio_path)
        self.chunk_sec = chunk_sec
        self.realtime = realtime
        self.start_offset_sec = start_offset_sec

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        wav, sr = torchaudio.load(str(self.audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        samples = wav.squeeze(0).numpy().astype(np.float32)

        start_sample = int(self.start_offset_sec * TARGET_SR)
        samples = samples[start_sample:]
        chunk_samples = int(self.chunk_sec * TARGET_SR)

        t_wall_start = time.monotonic()
        idx = 0
        while idx + chunk_samples <= len(samples):
            chunk = samples[idx : idx + chunk_samples]
            song_t = self.start_offset_sec + idx / TARGET_SR
            if self.realtime:
                target_wall = t_wall_start + (idx / TARGET_SR)
                now = time.monotonic()
                sleep_s = target_wall - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
            yield chunk, song_t
            idx += chunk_samples
