"""Core utilities for RPG Chronicler.

This module is intentionally independent from Tkinter so audio preparation,
session persistence and exports can be tested without opening the GUI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from typing import Any, Iterable
import urllib.parse
from urllib.parse import urlparse
import urllib.request
import zipfile
import base64
import html
import http.server
import ssl

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal


SENSITIVE_CONFIG_KEYS = {"lm_studio_api_key", "openai_account_id"}


def normalize_openai_base_url(raw_url: str) -> str:
    value = (raw_url or "http://127.0.0.1:1234/v1").strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    parsed = urlparse(value)
    if parsed.path in {"", "/"}:
        value = f"{value}/v1"
    return value


def is_local_endpoint(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def cloud_destination_fingerprint(url: str, model: str = "") -> tuple[str, str, int | None, str]:
    """Stable identity used to scope informed consent to one remote destination."""
    parsed = urlparse(normalize_openai_base_url(url))
    return (parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port, model.strip())


def describe_model_access(
    base_url: str, model_id: str, metadata: dict[str, Any] | None = None
) -> dict[str, str]:
    """Describe model maker, effective API provider and evidenced billing mode."""
    parsed = urlparse(normalize_openai_base_url(base_url))
    host = (parsed.hostname or "").lower()
    model_id = model_id.strip()
    maker_slug = model_id.split("/", 1)[0] if "/" in model_id else ""
    maker_names = {
        "openai": "OpenAI", "google": "Google", "deepseek": "DeepSeek",
        "anthropic": "Anthropic", "meta-llama": "Meta", "mistralai": "Mistral AI",
        "cohere": "Cohere", "stepfun": "StepFun", "poolside": "Poolside",
        "tencent": "Tencent", "upstage": "Upstage", "meituan": "Meituan",
        "x-ai": "xAI", "z-ai": "Z.ai", "moonshotai": "Moonshot AI",
        "thinkingmachines": "Thinking Machines", "kwaipilot": "KwaiPilot",
        "sao10k": "Sao10K",
    }
    maker = maker_names.get(maker_slug, maker_slug or "Não informado")

    if host in {"localhost", "127.0.0.1", "::1"} and parsed.port == 1234:
        return {"maker": maker if maker_slug else "Modelo local", "provider": "LM Studio local", "billing": "Local / sem créditos de API"}
    if host in {"localhost", "127.0.0.1", "::1"} and parsed.port == 8645:
        provider = "Hermes / Nous Portal"
    elif host == "api.openai.com":
        provider = "OpenAI API"
        maker = "OpenAI"
    elif "generativelanguage.googleapis.com" in host:
        provider = "Google Gemini API"
        maker = "Google"
    elif host == "api.deepseek.com":
        provider = "DeepSeek API"
        maker = "DeepSeek"
    else:
        provider = host or "Não informado"

    access_mode = str((metadata or {}).get("access_mode", "")).lower()
    pricing = (metadata or {}).get("pricing") or {}
    numeric_prices = []
    for key, value in pricing.items():
        if key == "original" or isinstance(value, (dict, list)):
            continue
        try:
            numeric_prices.append(float(value))
        except (TypeError, ValueError):
            pass
    if access_mode == "subscription":
        billing = "Assinatura / franquia de tokens"
    elif model_id.endswith(":free") or (numeric_prices and max(numeric_prices) == 0):
        billing = "Grátis (pode ter limites)"
    elif numeric_prices:
        billing = "Créditos / pagamento por uso"
    elif provider == "Google Gemini API":
        billing = "Cota grátis ou pagamento por uso"
    else:
        billing = "Não informado pelo provedor"
    return {"maker": maker, "provider": provider, "billing": billing}


def model_catalog_sort_key(base_url: str, metadata: dict[str, Any]) -> tuple[str, int, str]:
    """Group catalog entries by maker, then access mode and model id."""
    model_id = str(metadata.get("id", ""))
    info = describe_model_access(base_url, model_id, metadata)
    billing_rank = {
        "Assinatura / franquia de tokens": 0,
        "Grátis (pode ter limites)": 1,
        "Créditos / pagamento por uso": 2,
    }.get(info["billing"], 3)
    return (info["maker"].casefold(), billing_rank, model_id.casefold())


def format_model_catalog_choice(base_url: str, metadata: dict[str, Any]) -> str:
    model_id = str(metadata.get("id", ""))
    info = describe_model_access(base_url, model_id, metadata)
    if info["billing"].startswith("Assinatura"):
        badge = "ASSINATURA"
    elif info["billing"].startswith("Grátis"):
        badge = "GRÁTIS"
    elif info["provider"] == "Hermes / Nous Portal":
        badge = "CRÉDITOS NOUS"
    elif info["billing"].startswith("Créditos"):
        badge = "CRÉDITOS"
    else:
        badge = "NÃO INFORMADO"
    return f"{info['maker']} | {badge} | {model_id}"


def atomic_write_text(path: Path | str, content: str) -> Path:
    """Write UTF-8 text atomically in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    return atomic_write_text(
        path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy safe for logs, manifests and example files."""
    return {
        key: (
            "<redacted>"
            if value and (key in SENSITIVE_CONFIG_KEYS or "key" in key.lower() or "token" in key.lower())
            else value
        )
        for key, value in config.items()
    }


@dataclass(frozen=True)
class AudioMetadata:
    path: str
    size_bytes: int
    duration_seconds: float
    codec: str
    sample_rate: int
    channels: int


@dataclass
class PreparedAudio:
    original_path: Path
    processing_path: Path
    metadata: AudioMetadata
    temporary_directory: Path | None = None

    def cleanup(self) -> None:
        if self.temporary_directory:
            shutil.rmtree(self.temporary_directory, ignore_errors=True)
            self.temporary_directory = None


def probe_audio(path: Path | str) -> AudioMetadata:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=30
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFprobe não está instalado ou não está no PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "arquivo de áudio inválido").strip()
        raise ValueError(f"Não foi possível ler o áudio: {detail}") from exc
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError("O arquivo não contém uma faixa de áudio.")
    stream = streams[0]
    return AudioMetadata(
        path=str(source),
        size_bytes=source.stat().st_size,
        duration_seconds=float((payload.get("format") or {}).get("duration") or 0.0),
        codec=str(stream.get("codec_name") or "unknown"),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
    )


def prepare_audio(path: Path | str, sample_rate: int = 16000) -> PreparedAudio:
    """Return a mono PCM WAV suitable for Whisper and scipy.wavfile.

    Original files are never modified. Compatible WAVs are reused; every other
    format is converted inside an isolated temporary directory.
    """
    source = Path(path).resolve()
    metadata = probe_audio(source)
    is_compatible = (
        source.suffix.lower() == ".wav"
        and metadata.codec.startswith("pcm_")
        and metadata.sample_rate == sample_rate
        and metadata.channels == 1
    )
    if is_compatible:
        return PreparedAudio(source, source, metadata)

    temporary_directory = Path(tempfile.mkdtemp(prefix="rpg-chronicler-audio-"))
    converted = temporary_directory / f"{source.stem}-mono-{sample_rate}.wav"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(converted),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
        converted_metadata = probe_audio(converted)
    except FileNotFoundError as exc:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise RuntimeError("FFmpeg não está instalado ou não está no PATH.") from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        detail = (exc.stderr or exc.stdout or "falha desconhecida").strip()
        raise ValueError(f"Falha ao converter o áudio: {detail}") from exc
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return PreparedAudio(source, converted, converted_metadata, temporary_directory)


def format_timestamp(seconds: float, *, decimal: str = ",") -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def segments_to_srt(segments: Iterable[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        speaker = str(segment.get("speaker") or "").strip()
        text = str(segment.get("text") or "").strip()
        line = f"[{speaker}] {text}" if speaker else text
        blocks.append(
            f"{index}\n{format_timestamp(segment['start'])} --> "
            f"{format_timestamp(segment['end'])}\n{line}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def segments_to_vtt(segments: Iterable[dict[str, Any]]) -> str:
    blocks = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "").strip()
        text = str(segment.get("text") or "").strip()
        line = f"<v {speaker}>{text}" if speaker else text
        blocks.append(
            f"{format_timestamp(segment['start'], decimal='.')} --> "
            f"{format_timestamp(segment['end'], decimal='.')}\n{line}"
        )
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")


def format_timestamp_ass(seconds: float) -> str:
    """Converte segundos para timestamp ASS (H:MM:SS.cs com centissegundos)."""
    cs_total = max(0, round(float(seconds) * 100))
    hours, rem = divmod(cs_total, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def segments_to_karaoke_ass(segments: Iterable[dict[str, Any]], title: str = "RPG Chronicler Karaoke") -> str:
    """
    Gera legendas em formato ASS (Advanced SubStation Alpha) com tags de karaokê ({\\k<cs>})
    para cada palavra falada, ideal para reels, shorts e players de vídeo que suportam karaokê dinâmico.
    """
    header = (
        "[Script Info]\n"
        f"Title: {title}\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: KaraokeDefault,Arial,48,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,40,40,160,1\n"
        "Style: SpeakerTag,Arial,36,&H0000D7FF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,40,40,240,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for segment in segments:
        start_t = format_timestamp_ass(segment.get("start", 0.0))
        end_t = format_timestamp_ass(segment.get("end", 0.0))
        speaker = str(segment.get("speaker") or "").strip()
        words = segment.get("words") or []

        if words:
            # Formata palavras com tags de karaokê {\k<centissegundos>}
            karaoke_text_parts = []
            for w in words:
                w_text = str(w.get("word") or "").strip()
                if not w_text:
                    continue
                w_dur_cs = max(1, round((float(w.get("end", 0.0)) - float(w.get("start", 0.0))) * 100))
                karaoke_text_parts.append(f"{{\\k{w_dur_cs}}}{w_text} ")
            spoken_line = "".join(karaoke_text_parts).strip()
        else:
            total_dur_cs = max(1, round((float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))) * 100))
            spoken_line = f"{{\\k{total_dur_cs}}}{segment.get('text', '').strip()}"

        prefix = f"{{\\c&H0000D7FF&}}[{speaker}] {{\\r}} " if speaker else ""
        events.append(f"Dialogue: 0,{start_t},{end_t},KaraokeDefault,,0,0,0,,{prefix}{spoken_line}")

    return header + "\n".join(events) + ("\n" if events else "")


class SessionRun:
    """Versioned, auditable on-disk checkpoint record of one processing run."""

    def __init__(self, directory: Path, manifest: dict[str, Any]):
        self.directory = directory
        self.manifest_path = directory / "manifest.json"
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        runs_directory: Path | str,
        metadata: AudioMetadata,
        config: dict[str, Any],
    ) -> "SessionRun":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        directory = Path(runs_directory) / timestamp
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": timestamp,
            "created_at": datetime.now().isoformat(),
            "status": "started",
            "audio": asdict(metadata),
            "config": public_config(config),
            "stages": {},
        }
        run = cls(directory, manifest)
        run._save_manifest()
        return run

    def _save_manifest(self) -> None:
        atomic_write_json(self.manifest_path, self.manifest)

    def mark_stage(self, name: str, status: str, **details: Any) -> None:
        self.manifest["stages"][name] = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            **details,
        }
        self.manifest["status"] = "failed" if status == "failed" else "running"
        self._save_manifest()

    def write_text_stage(self, name: str, filename: str, content: str) -> Path:
        destination = atomic_write_text(self.directory / filename, content)
        self.mark_stage(name, "completed", file=filename)
        return destination

    def write_transcript(self, segments: list[dict[str, Any]]) -> None:
        atomic_write_json(self.directory / "transcript.json", {"segments": segments})
        markdown = "# Transcrição\n\n" + "\n".join(
            f"- **{format_timestamp(s['start'], decimal='.')} → "
            f"{format_timestamp(s['end'], decimal='.')}** "
            f"{('[' + s['speaker'] + '] ') if s.get('speaker') else ''}{s.get('text', '')}"
            for s in segments
        ) + "\n"
        atomic_write_text(self.directory / "transcript.md", markdown)
        atomic_write_text(self.directory / "transcript.srt", segments_to_srt(segments))
        atomic_write_text(self.directory / "transcript.vtt", segments_to_vtt(segments))
        atomic_write_text(self.directory / "transcript.ass", segments_to_karaoke_ass(segments))
        self.mark_stage(
            "transcription",
            "completed",
            files=["transcript.json", "transcript.md", "transcript.srt", "transcript.vtt", "transcript.ass"],
            segments=len(segments),
        )

    def finish(self) -> None:
        self.manifest["status"] = "completed"
        self.manifest["completed_at"] = datetime.now().isoformat()
        self._save_manifest()

    def fail(self, stage: str, error: Exception | str) -> None:
        self.mark_stage(stage, "failed", error=str(error))


# =========================================================================
# DSP: SUPRESSOR DE RUÍDOS & MELHORADOR VOCAL DINÂMICO
# =========================================================================

def apply_highpass_filter(
    audio_data: np.ndarray, sample_rate: int = 16000, cutoff: float = 80.0
) -> np.ndarray:
    """Filtro passa-altas Butterworth (4ª ordem) para remover rumble, batidas na mesa e 60Hz."""
    if len(audio_data) < 16:
        return audio_data.astype(np.float32)
    nyquist = sample_rate / 2.0
    normalized_cutoff = min(0.95, max(0.005, cutoff / nyquist))
    b, a = signal.butter(4, normalized_cutoff, btype="highpass")
    filtered = signal.filtfilt(b, a, audio_data.astype(np.float64))
    return filtered.astype(np.float32)


def apply_spectral_noise_suppression(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    noise_reduction_ratio: float = 0.85,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Supressor de ruído por Spectral Gating / Subtração Espectral Suave.
    Rastreia o piso de ruído estacionário e atenua frequências sem voz, eliminando chiado.
    """
    if len(audio_data) < n_fft * 2:
        return audio_data.astype(np.float32)

    # Janela Hann
    window = np.hanning(n_fft)
    num_samples = len(audio_data)
    num_frames = max(1, int(np.floor((num_samples - n_fft) / hop_length)) + 1)

    pad_len = (num_frames - 1) * hop_length + n_fft
    padded = np.zeros(pad_len, dtype=np.float32)
    padded[:num_samples] = audio_data

    # STFT
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(num_frames, n_fft),
        strides=(padded.strides[0] * hop_length, padded.strides[0]),
    )
    windowed_frames = frames * window
    stft = np.fft.rfft(windowed_frames, n=n_fft)

    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimação adaptativa do perfil de ruído (percentil 15 das menores energias em cada banda)
    noise_estimate = np.percentile(magnitude, 15, axis=0, keepdims=True)
    noise_estimate = np.maximum(noise_estimate, 1e-6)

    # Over-subtraction com piso espectral suave para evitar artefatos de "musical noise"
    alpha = min(2.5, max(1.0, 1.2 + noise_reduction_ratio))
    beta = 0.05  # Spectral floor

    subtracted = magnitude - alpha * (noise_estimate * noise_reduction_ratio)
    gain = np.maximum(subtracted / np.maximum(magnitude, 1e-6), beta)
    gain = np.clip(gain, beta, 1.0)

    # Reconstrução com ISTFT e Overlap-Add
    clean_mag = magnitude * gain
    clean_stft = clean_mag * np.exp(1j * phase)
    reconstructed_frames = np.fft.irfft(clean_stft, n=n_fft) * window

    output = np.zeros(pad_len, dtype=np.float32)
    window_sum = np.zeros(pad_len, dtype=np.float32)
    win_sq = window ** 2

    for i in range(num_frames):
        pos = i * hop_length
        output[pos : pos + n_fft] += reconstructed_frames[i]
        window_sum[pos : pos + n_fft] += win_sq

    nonzero_mask = window_sum > 1e-6
    output[nonzero_mask] /= window_sum[nonzero_mask]
    return output[:num_samples].astype(np.float32)


def apply_speech_vocal_enhancer(
    audio_data: np.ndarray, sample_rate: int = 16000
) -> np.ndarray:
    """
    Realce de formantes e inteligibilidade de fala humana.
    Aplica equalização de presença (1.5 kHz a 4.0 kHz) para destacar a clareza dos timbres.
    """
    if len(audio_data) < 32:
        return audio_data.astype(np.float32)

    # Filtro peaking de presença em 2.8 kHz com Q=1.2 e ganho de +2.5 dB
    center_freq = 2800.0
    gain_db = 2.5
    q_factor = 1.2

    w0 = 2.0 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2.0 * q_factor)
    a_gain = 10.0 ** (gain_db / 40.0)

    b0 = 1.0 + alpha * a_gain
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a_gain
    a0 = 1.0 + alpha / a_gain
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a_gain

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

    enhanced = signal.filtfilt(b, a, audio_data.astype(np.float64))
    return enhanced.astype(np.float32)


def apply_dynamic_gain_control(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    target_rms_db: float = -20.0,
    max_gain_db: float = 14.0,
) -> np.ndarray:
    """
    AGC Adaptativo (Automatic Gain Control) com nivelamento temporal por blocos,
    suavização de ataque/relaxamento (attack/release envelope) e soft-knee limiter.
    Permite que vozes distantes de jogadores na mesa sejam amplificadas sem estourar
    a voz de quem fala perto do microfone.
    """
    if len(audio_data) < 128:
        return audio_data.astype(np.float32)

    audio = audio_data.astype(np.float32).copy()

    # Tamanho do bloco de análise (~50ms) e hop (~12.5ms)
    frame_len = max(64, int(sample_rate * 0.05))
    hop_len = max(16, int(sample_rate * 0.0125))

    num_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
    gain_envelope = np.ones(num_frames, dtype=np.float32)

    # Piso de ruído / silêncio (-48 dB) para não amplificar chiado em silêncio absoluto
    silence_threshold_linear = 10.0 ** (-48.0 / 20.0)

    for i in range(num_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        local_rms = float(np.sqrt(np.mean(frame**2) + 1e-9))

        if local_rms < silence_threshold_linear:
            # Em silêncio absoluto ou pausa, mantém ganho neutro
            gain_envelope[i] = 1.0
        else:
            local_db = 20.0 * np.log10(local_rms)
            target_gain_db = np.clip(target_rms_db - local_db, -6.0, max_gain_db)
            gain_envelope[i] = 10.0 ** (target_gain_db / 20.0)

    # Suavização do envelope de ganho (Attack/Release smoothing)
    # Attack rápido (~20ms) para conter picos, Release mais lento (~150ms) para naturalidade
    smooth_gains = np.empty_like(gain_envelope)
    smooth_gains[0] = gain_envelope[0]
    alpha_attack = 0.35
    alpha_release = 0.08

    for i in range(1, num_frames):
        target = gain_envelope[i]
        curr = smooth_gains[i - 1]
        if target < curr:
            smooth_gains[i] = alpha_attack * target + (1.0 - alpha_attack) * curr
        else:
            smooth_gains[i] = alpha_release * target + (1.0 - alpha_release) * curr

    # Interpolação linear da curva de ganho ao longo de cada amostra de áudio
    frame_centers = np.arange(num_frames) * hop_len + (frame_len // 2)
    sample_indices = np.arange(len(audio))
    interpolated_gain = np.interp(sample_indices, frame_centers, smooth_gains).astype(np.float32)

    boosted = audio * interpolated_gain

    # Compressão não-linear suave (soft-knee limiter) para prevenir distorção
    threshold = 0.75
    over_mask = np.abs(boosted) > threshold
    if np.any(over_mask):
        excess = np.abs(boosted[over_mask]) - threshold
        compressed = threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold + 1e-6))
        boosted[over_mask] = np.sign(boosted[over_mask]) * compressed

    # Normalização de teto segura para -1.0 dBFS (~0.89 linear)
    peak = float(np.max(np.abs(boosted)))
    if peak > 0.89:
        boosted = boosted * (0.89 / peak)

    return boosted.astype(np.float32)


def apply_transient_suppression(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    threshold_factor: float = 3.5,
) -> np.ndarray:
    """
    Suprime impactos secos e ruídos transientes rápidos típicos de mesa de RPG
    (rolagem de dados, copos, toques e batidas acidentais na mesa).
    Utiliza limitação de taxa de variação (slew-rate) e detecção de pico impulsivo.
    """
    if len(audio_data) < 32:
        return audio_data.astype(np.float32)

    data = audio_data.astype(np.float32).copy()
    diff = np.diff(data, prepend=data[0])
    abs_diff = np.abs(diff)

    # Mediana móvel da taxa de variação
    med_diff = float(np.median(abs_diff)) + 1e-6
    transient_mask = abs_diff > (med_diff * threshold_factor * 2.5)

    if np.any(transient_mask):
        # Suavização adaptativa apenas nos instantes de transiente
        indices = np.where(transient_mask)[0]
        for idx in indices:
            start_i = max(0, idx - 2)
            end_i = min(len(data), idx + 3)
            # Substitui o pico impulsivo por interpolação local
            data[idx] = float(np.median(data[start_i:end_i]))

    return data


def apply_dereverberation(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    decay_reverb_ratio: float = 0.40,
) -> np.ndarray:
    """
    Desreverberação Espectral para Salas Médias e Grandes (Microfone de Conferência).
    Atenua reflexões tardias e caudas de eco (reverb) calculando a resposta de decaimento
    espectral entre janelas adjacentes de STFT.
    """
    if len(audio_data) < 512:
        return audio_data.astype(np.float32)

    n_fft = 512
    hop_length = 128
    window = np.hanning(n_fft)

    # STFT
    num_frames = max(1, 1 + (len(audio_data) - n_fft) // hop_length)
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = audio_data[start : start + n_fft] * window

    stft_matrix = np.fft.rfft(frames, axis=1)
    mag = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Modelo de decaimento de cauda reverberante (late reflections estimator)
    reverb_est = np.zeros_like(mag)
    alpha = 0.75  # Constante de tempo de decaimento de sala (~200ms)
    
    for t in range(1, num_frames):
        # O eco acumulado é proporcional à energia dos quadros anteriores
        reverb_est[t] = alpha * reverb_est[t - 1] + (1.0 - alpha) * mag[t - 1]

    # Subtração da componente reverberante estimada
    suppressed_mag = np.maximum(mag - (decay_reverb_ratio * reverb_est), 0.15 * mag)
    clean_stft = suppressed_mag * np.exp(1j * phase)

    # ISTFT (Overlap-Add)
    reconstructed_frames = np.fft.irfft(clean_stft, axis=1) * window
    output = np.zeros(num_frames * hop_length + n_fft, dtype=np.float32)
    norm_window = np.zeros(num_frames * hop_length + n_fft, dtype=np.float32)
    window_sq = window**2

    for i in range(num_frames):
        start = i * hop_length
        output[start : start + n_fft] += reconstructed_frames[i]
        norm_window[start : start + n_fft] += window_sq

    nonzero = norm_window > 1e-4
    output[nonzero] /= norm_window[nonzero]

    result = output[: len(audio_data)]
    if len(result) < len(audio_data):
        result = np.pad(result, (0, len(audio_data) - len(result)))

    return result.astype(np.float32)


def enhance_audio_pipeline(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    enable_denoise: bool = True,
    enable_enhance: bool = True,
    enable_agc: bool = True,
    enable_conference_mode: bool = False,
    enable_plosive_suppression: bool = True,
) -> np.ndarray:
    """
    Pipeline mestre de áudio com suporte a Microfone de Conferência & Sala Grande:
    1. Passa-altas (Rumble cut < 80Hz)
    2. Supressão de Plosivas / De-Popper (< 75Hz)
    3. Supressão de Transientes (Rolagem de dados na mesa / impactos)
    4. Desreverberação de Sala (Eliminação de eco oco) [se conference_mode]
    5. Supressão espectral de ruído e chiado
    6. Realce de formantes de fala (Presença vocal)
    7. AGC e Normalização de ganho de longa distância (+18dB para participantes distantes)
    """
    if audio_data is None or len(audio_data) == 0:
        return np.array([], dtype=np.float32)

    data = audio_data.astype(np.float32)
    # Converte inteiros para escala float normalizada [-1.0, 1.0] se necessário
    if np.issubdtype(audio_data.dtype, np.integer):
        data = data / float(np.iinfo(audio_data.dtype).max)
    elif np.max(np.abs(data)) > 1.0:
        data = data / float(np.max(np.abs(data)))

    # 1. Filtro Passa-Altas
    data = apply_highpass_filter(data, sample_rate=sample_rate, cutoff=80.0)

    # 1.1 Supressão de Plosivas (De-Popper para pops de P/B/T)
    if enable_plosive_suppression:
        data = apply_plosive_suppression(data, sample_rate=sample_rate, cutoff_hz=75.0)

    # 2. Supressão de Ruídos Transientes (Mesa / Dados)
    if enable_conference_mode:
        data = apply_transient_suppression(data, sample_rate=sample_rate)

    # 3. Desreverberação Espectral (Eco de Sala Grande)
    if enable_conference_mode:
        data = apply_dereverberation(data, sample_rate=sample_rate, decay_reverb_ratio=0.45)

    # 4. Supressor de Ruído Espectral
    if enable_denoise:
        denoise_ratio = 0.85 if enable_conference_mode else 0.80
        data = apply_spectral_noise_suppression(data, sample_rate=sample_rate, noise_reduction_ratio=denoise_ratio)

    # 5. Realce Vocal
    if enable_enhance:
        data = apply_speech_vocal_enhancer(data, sample_rate=sample_rate)

    # 6. AGC e Nivelamento Dinâmico (com Ganho Estendido para Sala Grande)
    if enable_agc or enable_conference_mode:
        target_db = -18.0 if enable_conference_mode else -19.0
        max_gain = 18.0 if enable_conference_mode else 12.0
        data = apply_dynamic_gain_control(data, sample_rate=sample_rate, target_rms_db=target_db, max_gain_db=max_gain)

    return data


def clean_and_enhance_audio_file(
    source_path: Path | str,
    output_path: Path | str | None = None,
    sample_rate: int = 16000,
    enable_denoise: bool = True,
    enable_enhance: bool = True,
    enable_agc: bool = True,
    enable_conference_mode: bool = False,
    enable_plosive_suppression: bool = True,
) -> Path:
    """Lê um arquivo WAV, aplica o pipeline de melhoria e salva um novo WAV cristalino."""
    src = Path(source_path).resolve()
    sr, data = wavfile.read(src)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)

    cleaned = enhance_audio_pipeline(
        data,
        sample_rate=sr,
        enable_denoise=enable_denoise,
        enable_enhance=enable_enhance,
        enable_agc=enable_agc,
        enable_conference_mode=enable_conference_mode,
        enable_plosive_suppression=enable_plosive_suppression,
    )

    max_v = float(np.max(np.abs(cleaned)))
    if max_v > 0:
        cleaned_int16 = (cleaned * 32767).astype(np.int16)
    else:
        cleaned_int16 = cleaned.astype(np.int16)

    if output_path is None:
        out_p = src.parent / f"{src.stem}_enhanced.wav"
    else:
        out_p = Path(output_path).resolve()

    wavfile.write(out_p, sr, cleaned_int16)
    return out_p


def apply_deesser(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    frequency: float = 6500.0,
    bandwidth: float = 2000.0,
    threshold_ratio: float = 0.35,
    max_attenuation_db: float = 6.0,
) -> np.ndarray:
    """
    De-Esser Dinâmico de Estúdio para Voz Humana.
    Detecta e atenua sibilâncias excessivas (sons de 'S', 'CH', 'Z') na faixa de 5.5 kHz a 8.5 kHz,
    evitando que a equalização de presença torne o áudio estridente ou cansativo ao ouvinte.
    """
    if len(audio_data) < 64:
        return audio_data.astype(np.float32)

    nyquist = sample_rate / 2.0
    low_freq = max(20.0, frequency - bandwidth / 2.0)
    high_freq = min(nyquist - 50.0, frequency + bandwidth / 2.0)

    if low_freq >= high_freq or high_freq >= nyquist:
        return audio_data.astype(np.float32)

    b_bp, a_bp = signal.butter(2, [low_freq / nyquist, high_freq / nyquist], btype="bandpass")
    sibilance_band = signal.filtfilt(b_bp, a_bp, audio_data.astype(np.float64)).astype(np.float32)

    env_b, env_a = signal.butter(1, min(0.95, 50.0 / nyquist), btype="lowpass")
    sibilance_env = signal.filtfilt(env_b, env_a, np.abs(sibilance_band).astype(np.float64)).astype(np.float32)

    broad_env = signal.filtfilt(env_b, env_a, np.abs(audio_data).astype(np.float64)).astype(np.float32)
    broad_env = np.maximum(broad_env, 1e-5)

    sib_ratio = sibilance_env / broad_env
    excess_mask = sib_ratio > threshold_ratio

    if not np.any(excess_mask):
        return audio_data.astype(np.float32)

    overshoot = np.clip((sib_ratio - threshold_ratio) / (threshold_ratio + 1e-4), 0.0, 1.0)
    max_gain_reduction = 1.0 - (10.0 ** (-abs(max_attenuation_db) / 20.0))
    attenuation_factor = 1.0 - (overshoot * max_gain_reduction)

    ducked_sibilance = sibilance_band * (1.0 - attenuation_factor)
    cleaned = audio_data.astype(np.float32) - ducked_sibilance
    return cleaned.astype(np.float32)


def apply_podcast_equalizer(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    low_cut_hz: float = 80.0,
    mud_cut_hz: float = 300.0,
    mud_gain_db: float = -2.5,
    presence_hz: float = 3000.0,
    presence_gain_db: float = 2.5,
    air_hz: float = 10000.0,
    air_gain_db: float = 1.8,
) -> np.ndarray:
    """
    Equalizador de Estúdio de 4 Bandas Otimizado para Podcast e Voz Falada:
    1. Low Cut (Highpass Butterworth 4ª ordem em 80 Hz): remove vibrações e estrondos graves.
    2. De-Mudding Peaking (-2.5 dB em 300 Hz, Q=1.0): limpa ressonâncias de sala e 'som oco/encanado'.
    3. Presença Vocal Peaking (+2.5 dB em 3.0 kHz, Q=1.2): realça inteligibilidade e consoantes.
    4. Air / Brilho High-Shelf (+1.8 dB em 10 kHz): confere proximidade e calor profissional.
    """
    if len(audio_data) < 32:
        return audio_data.astype(np.float32)

    data = audio_data.astype(np.float64)
    nyquist = sample_rate / 2.0

    data = apply_highpass_filter(data.astype(np.float32), sample_rate=sample_rate, cutoff=low_cut_hz).astype(np.float64)

    def _apply_peaking(audio_signal: np.ndarray, center_f: float, gain_d: float, q_f: float) -> np.ndarray:
        if center_f >= nyquist * 0.95 or abs(gain_d) < 0.1:
            return audio_signal
        w0 = 2.0 * np.pi * center_f / sample_rate
        alpha = np.sin(w0) / (2.0 * q_f)
        a_gain = 10.0 ** (gain_d / 40.0)
        b0 = 1.0 + alpha * a_gain
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * a_gain
        a0 = 1.0 + alpha / a_gain
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / a_gain
        b_coeff = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a_coeff = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return signal.filtfilt(b_coeff, a_coeff, audio_signal)

    data = _apply_peaking(data, mud_cut_hz, mud_gain_db, 1.0)
    data = _apply_peaking(data, presence_hz, presence_gain_db, 1.2)

    if air_hz < nyquist * 0.9:
        data = _apply_peaking(data, air_hz, air_gain_db, 0.7)

    return data.astype(np.float32)


def calculate_integrated_lufs(audio_data: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Calcula o Loudness Integrado em LUFS seguindo as especificações ITU-R BS.1770-4 / EBU R128.
    Inclui filtro de ponderação K (High-Shelf estágio 1 + Highpass RLB estágio 2) e janelamento com gating.
    """
    if len(audio_data) < sample_rate * 0.4:
        rms = float(np.sqrt(np.mean(audio_data**2) + 1e-12))
        return float(20.0 * np.log10(rms)) if rms > 1e-5 else -70.0

    nyquist = sample_rate / 2.0
    data = audio_data.astype(np.float64)

    shelf_freq = min(1500.0, nyquist * 0.5)
    w0 = 2.0 * np.pi * shelf_freq / sample_rate
    a_gain = 10.0 ** (4.0 / 40.0)
    alpha = np.sin(w0) / 2.0 * np.sqrt((a_gain + 1.0 / a_gain) * (1.0 / 0.707 - 1.0) + 2.0)
    cos_w0 = np.cos(w0)
    b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w0 + 2.0 * np.sqrt(a_gain) * alpha)
    b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0)
    b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w0 - 2.0 * np.sqrt(a_gain) * alpha)
    a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 + 2.0 * np.sqrt(a_gain) * alpha
    a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0)
    a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 - 2.0 * np.sqrt(a_gain) * alpha

    b_hs = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a_hs = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    y_stage1 = signal.lfilter(b_hs, a_hs, data)

    b_rlb, a_rlb = signal.butter(2, min(0.9, 100.0 / nyquist), btype="highpass")
    y_k = signal.lfilter(b_rlb, a_rlb, y_stage1)

    block_size = int(sample_rate * 0.40)
    hop_size = int(sample_rate * 0.10)
    num_blocks = max(1, (len(y_k) - block_size) // hop_size + 1)

    powers = []
    for i in range(num_blocks):
        start = i * hop_size
        block = y_k[start : start + block_size]
        z_i = float(np.mean(block**2))
        powers.append(z_i)

    powers = np.array(powers, dtype=np.float64)
    valid = powers > 1e-12
    if not np.any(valid):
        return -70.0

    loudness_blocks = -0.691 + 10.0 * np.log10(powers[valid])

    abs_mask = loudness_blocks > -70.0
    if not np.any(abs_mask):
        return -70.0

    z_abs = np.mean(powers[valid][abs_mask])
    gamma_rel = -0.691 + 10.0 * np.log10(z_abs) - 10.0

    rel_mask = loudness_blocks > gamma_rel
    if not np.any(rel_mask):
        return float(round(-0.691 + 10.0 * np.log10(z_abs), 1))

    z_rel = np.mean(powers[valid][rel_mask])
    integrated_lufs = -0.691 + 10.0 * np.log10(max(1e-12, z_rel))
    return float(round(integrated_lufs, 1))


def apply_loudness_normalization(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    target_lufs: float = -16.0,
    true_peak_limit_db: float = -1.0,
) -> np.ndarray:
    """
    Normalização de Loudness Padrão Broadcast / Podcast (EBU R128).
    - Podcast estéreo: alvo típico -16 LUFS (-19 LUFS em mono).
    - Aplica ganho unificado e um limiter True-Peak com soft-knee em -1.0 dBFS para evitar clipping em MP3/AAC.
    """
    if len(audio_data) < sample_rate * 0.2:
        return audio_data.astype(np.float32)

    current_lufs = calculate_integrated_lufs(audio_data, sample_rate=sample_rate)
    if current_lufs <= -68.0:
        return audio_data.astype(np.float32)

    gain_db = np.clip(target_lufs - current_lufs, -24.0, 24.0)
    gain_linear = 10.0 ** (gain_db / 20.0)

    leveled = audio_data.astype(np.float32) * float(gain_linear)

    peak_ceiling = 10.0 ** (true_peak_limit_db / 20.0)
    threshold = peak_ceiling * 0.85

    over_mask = np.abs(leveled) > threshold
    if np.any(over_mask):
        excess = np.abs(leveled[over_mask]) - threshold
        limit_range = peak_ceiling - threshold
        compressed = threshold + limit_range * np.tanh(excess / (limit_range + 1e-6))
        leveled[over_mask] = np.sign(leveled[over_mask]) * compressed

    max_p = float(np.max(np.abs(leveled)))
    if max_p > peak_ceiling:
        leveled = leveled * (peak_ceiling / max_p)

    return leveled.astype(np.float32)


def detect_overlapping_speech(
    audio_segment: np.ndarray,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """
    Detector Heurístico de Sobreposição de Falas (Overlapping Speech Detection - OSD).
    Analisa multiplicidade de picos harmônicos na autocorrelação e dispersão espectral.
    Útil em sessões de RPG onde múltiplos jogadores reagem ou discutem simultaneamente.
    """
    if len(audio_segment) < sample_rate * 0.3:
        return {"is_overlapping": False, "confidence": 0.0, "multi_pitch_score": 0.0, "spectral_entropy": 0.0}

    segment = audio_segment[: sample_rate * 4].astype(np.float32)
    max_val = np.max(np.abs(segment))
    if max_val > 1e-4:
        segment = segment / max_val

    corr = signal.fftconvolve(segment, segment[::-1], mode="full")
    corr = corr[len(segment) - 1 :]
    corr = corr / (corr[0] + 1e-8)

    min_lag = int(sample_rate / 350.0)
    max_lag = int(sample_rate / 70.0)
    search_corr = corr[min_lag:max_lag]

    peaks, properties = signal.find_peaks(search_corr, height=0.20, distance=int(sample_rate / 500.0))
    peak_heights = properties.get("peak_heights", [])

    multi_pitch_score = 0.0
    if len(peak_heights) >= 2:
        sorted_heights = np.sort(peak_heights)[::-1]
        ratio = sorted_heights[1] / max(1e-4, sorted_heights[0])
        if ratio > 0.60:
            multi_pitch_score = float(ratio)

    fft_mag = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    prob_dist = fft_mag / (np.sum(fft_mag) + 1e-8)
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-12))
    norm_entropy = float(entropy / (np.log2(len(prob_dist)) + 1e-8))

    is_overlapping = (multi_pitch_score > 0.65 and norm_entropy > 0.60) or (len(peak_heights) >= 3 and norm_entropy > 0.70)
    conf = float(np.clip((multi_pitch_score * 0.6 + norm_entropy * 0.4), 0.0, 1.0))

    return {
        "is_overlapping": is_overlapping,
        "confidence": round(conf, 2),
        "multi_pitch_score": round(multi_pitch_score, 2),
        "spectral_entropy": round(norm_entropy, 2),
    }


def master_podcast_audio(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    target_lufs: float = -16.0,
    enable_denoise: bool = True,
    enable_deesser: bool = True,
    enable_eq: bool = True,
    enable_conference_mode: bool = True,
) -> np.ndarray:
    """
    Cadeia Completa de Masterização Profissional para Podcasts e Audiovisual:
    1. Pipeline Base de Áudio (Passa-altas + Transientes + Desreverberação + Denoise Espectral)
    2. Equalizador de Estúdio de 4 Bandas
    3. De-Esser Dinâmico de Precisão
    4. Normalização Final de Loudness EBU R128 (-16 LUFS estéreo / -19 LUFS mono) com True Peak Limiter.
    """
    if audio_data is None or len(audio_data) == 0:
        return np.array([], dtype=np.float32)

    processed = enhance_audio_pipeline(
        audio_data,
        sample_rate=sample_rate,
        enable_denoise=enable_denoise,
        enable_enhance=True,
        enable_agc=True,
        enable_conference_mode=enable_conference_mode,
    )

    if enable_eq:
        processed = apply_podcast_equalizer(processed, sample_rate=sample_rate)

    if enable_deesser:
        processed = apply_deesser(processed, sample_rate=sample_rate)

    processed = apply_loudness_normalization(
        processed, sample_rate=sample_rate, target_lufs=target_lufs, true_peak_limit_db=-1.0
    )

    return processed.astype(np.float32)


def master_podcast_audio_file(
    source_path: Path | str,
    output_path: Path | str | None = None,
    sample_rate: int = 16000,
    target_lufs: float = -16.0,
    enable_denoise: bool = True,
    enable_deesser: bool = True,
    enable_eq: bool = True,
    enable_conference_mode: bool = True,
) -> Path:
    """Processa um arquivo WAV e salva a versão masterizada pronta para publicação em plataformas de áudio."""
    src = Path(source_path).resolve()
    sr, data = wavfile.read(src)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)

    mastered = master_podcast_audio(
        data,
        sample_rate=sr,
        target_lufs=target_lufs,
        enable_denoise=enable_denoise,
        enable_deesser=enable_deesser,
        enable_eq=enable_eq,
        enable_conference_mode=enable_conference_mode,
    )

    max_v = float(np.max(np.abs(mastered)))
    if max_v > 0:
        mastered_int16 = (mastered * 32767).astype(np.int16)
    else:
        mastered_int16 = mastered.astype(np.int16)

    if output_path is None:
        out_p = src.parent / f"{src.stem}_podcast_master.wav"
    else:
        out_p = Path(output_path).resolve()

    wavfile.write(out_p, sr, mastered_int16)
    return out_p


def apply_adaptive_speaker_eq(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    pitch_f0: float = 150.0,
    spectral_centroid: float = 1800.0,
) -> np.ndarray:
    """
    Equalizador Adaptativo Sensível ao Locutor (Auto-EQ per Speaker):
    Ajusta dinamicamente a curva de corte de graves, redução de ressonâncias e realce
    de inteligibilidade com base na frequência fundamental (F0) do jogador.
    - Vozes graves (F0 < 115 Hz): Corte de rumble mais alto (95 Hz) e desobstrução de 250 Hz (-3.5 dB).
    - Vozes agudas (F0 > 190 Hz): Corte de graves mais musical (70 Hz), atenuação de aspereza em 4.5 kHz.
    - Vozes médias (115 - 190 Hz): Curva de transmissão balanceada.
    """
    if audio_data is None or len(audio_data) == 0:
        return np.array([], dtype=np.float32)

    data = audio_data.astype(np.float64)
    nyquist = sample_rate / 2.0

    if pitch_f0 > 0 and pitch_f0 < 115.0:
        hp_cutoff = min(95.0, nyquist * 0.45)
        mud_freq = 240.0
        mud_gain = -3.5
        presence_freq = 3200.0
        presence_gain = 2.5
    elif pitch_f0 > 190.0:
        hp_cutoff = min(70.0, nyquist * 0.45)
        mud_freq = 380.0
        mud_gain = -2.0
        presence_freq = 2700.0
        presence_gain = 1.8
    else:
        hp_cutoff = min(80.0, nyquist * 0.45)
        mud_freq = 300.0
        mud_gain = -2.5
        presence_freq = 3000.0
        presence_gain = 2.0

    # 1. Filtro passa-altas adaptativo
    b_hp, a_hp = signal.butter(2, hp_cutoff / nyquist, btype="highpass")
    data = signal.filtfilt(b_hp, a_hp, data)

    # 2. Peaking EQ paramétrico para lama/ressonância e presença
    def _apply_peaking(audio_sig: np.ndarray, f0_hz: float, gain_db: float, q_val: float) -> np.ndarray:
        if f0_hz >= nyquist * 0.95 or f0_hz <= 10.0:
            return audio_sig
        w0 = 2.0 * np.pi * f0_hz / sample_rate
        alpha = np.sin(w0) / (2.0 * q_val)
        a_gain = 10.0 ** (gain_db / 40.0)
        b0 = 1.0 + alpha * a_gain
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * a_gain
        a0 = 1.0 + alpha / a_gain
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / a_gain
        b_coeff = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a_coeff = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return signal.filtfilt(b_coeff, a_coeff, audio_sig)

    data = _apply_peaking(data, mud_freq, mud_gain, 1.2)
    data = _apply_peaking(data, presence_freq, presence_gain, 1.3)

    return data.astype(np.float32)


def apply_adaptive_spectral_gate(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    threshold_db: float = -42.0,
    attack_ms: float = 6.0,
    release_ms: float = 50.0,
) -> np.ndarray:
    """
    Portão Espectral Adaptativo (Adaptive Spectral Gate):
    Atenua ruídos de fundo entre pausas de falas (como rolagem de dados na mesa,
    cliques de caneta, rangidos de cadeira e ar-condicionado) preservando a dinâmica vocal.
    """
    if audio_data is None or len(audio_data) < 256:
        return np.array(audio_data, dtype=np.float32)

    frame_length = int(sample_rate * 0.020)  # 20ms
    hop_length = int(sample_rate * 0.010)    # 10ms
    if frame_length % 2 != 0:
        frame_length += 1

    window = np.hanning(frame_length)
    n_frames = max(1, (len(audio_data) - frame_length) // hop_length + 1)
    
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    attack_coeff = float(np.exp(-1.0 / (attack_ms * sample_rate / 1000.0 / hop_length + 1e-6)))
    release_coeff = float(np.exp(-1.0 / (release_ms * sample_rate / 1000.0 / hop_length + 1e-6)))

    output_audio = np.zeros(len(audio_data), dtype=np.float64)
    window_sum = np.zeros(len(audio_data), dtype=np.float64)
    current_gain = 1.0

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio_data):
            break
        frame = audio_data[start:end] * window
        frame_rms = float(np.sqrt(np.mean(frame**2) + 1e-12))

        target_gain = 1.0 if frame_rms >= threshold_linear else 0.15 + 0.85 * (frame_rms / (threshold_linear + 1e-9))
        target_gain = float(np.clip(target_gain, 0.05, 1.0))

        if target_gain > current_gain:
            current_gain = attack_coeff * current_gain + (1.0 - attack_coeff) * target_gain
        else:
            current_gain = release_coeff * current_gain + (1.0 - release_coeff) * target_gain

        output_audio[start:end] += frame * current_gain
        window_sum[start:end] += window

    valid_mask = window_sum > 1e-4
    output_audio[valid_mask] /= window_sum[valid_mask]
    output_audio[~valid_mask] = audio_data[~valid_mask]

    return output_audio.astype(np.float32)


def split_multichannel_audio(audio_matrix: np.ndarray) -> list[np.ndarray]:
    """
    Desmembra áudio multi-canal (estéreo, 4 canais de interface de mesa, virtual cable)
    em faixas mono independentes prontas para diarização física por canal de microfone.
    """
    if audio_matrix is None or len(audio_matrix) == 0:
        return []

    arr = np.asarray(audio_matrix)
    if arr.ndim == 1:
        return [arr.astype(np.float32)]

    # Se a matriz tiver shape (samples, channels)
    if arr.shape[0] > arr.shape[1]:
        channels = [arr[:, ch].astype(np.float32) for ch in range(arr.shape[1])]
    else:  # shape (channels, samples)
        channels = [arr[ch, :].astype(np.float32) for ch in range(arr.shape[0])]

    return channels


def extract_campaign_vocabulary(bible_text: str) -> dict[str, list[str]]:
    """
    Extrai entidades canônicas, NPCs, cidades, magias e facções a partir da Bíblia de Campanha.
    """
    if not bible_text:
        return {"characters": [], "npcs": [], "locations": [], "factions": [], "terms": []}

    categories: dict[str, list[str]] = {
        "characters": [],
        "npcs": [],
        "locations": [],
        "factions": [],
        "terms": []
    }

    current_cat = "terms"
    lines = bible_text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            header = stripped[3:].lower()
            if "personagen" in header or "jogador" in header:
                current_cat = "characters"
            elif "npc" in header:
                current_cat = "npcs"
            elif "cenário" in header or "loca" in header or "cidade" in header:
                current_cat = "locations"
            elif "facç" in header or "ordem" in header or "guilda" in header:
                current_cat = "factions"
            else:
                current_cat = "terms"
        elif stripped.startswith("### "):
            name = stripped[4:].split("(")[0].split("-")[0].strip()
            if name and len(name) <= 40:
                if name not in categories[current_cat]:
                    categories[current_cat].append(name)
        elif stripped.startswith("- **"):
            match = re.match(r"-\s*\*\*([^*]+)\*\*", stripped)
            if match:
                item_name = match.group(1).split(":")[0].strip()
                if item_name and len(item_name) <= 40 and item_name not in categories[current_cat]:
                    categories[current_cat].append(item_name)

    return categories


def build_whisper_prompt_bias(bible_text: str, max_words: int = 150) -> str:
    """
    Constrói um initial_prompt denso e contextual para guiar o modelo Whisper a transcrever
    corretamente nomes próprios, monstros e terminologia da campanha em vez de alucinar termos comuns.
    """
    vocab = extract_campaign_vocabulary(bible_text)
    all_terms = []
    for cat in ["characters", "npcs", "locations", "factions", "terms"]:
        all_terms.extend(vocab.get(cat, []))

    # Adiciona termos fundamentais de RPG para reforço fonético
    rpg_base_terms = ["d20", "crítico", "iniciativa", "CA", "PV", "salvaguarda", "Tiefling", "Grimório", "Taumaturgia"]
    for t in rpg_base_terms:
        if t not in all_terms:
            all_terms.append(t)

    prompt_str = "Vocabulário da sessão de RPG: " + ", ".join(all_terms[:max_words]) + "."
    return prompt_str


def normalize_rpg_transcript_mechanics(text: str, custom_entities: list[str] | None = None) -> str:
    """
    Pós-processador Fonético & Mecânico de RPG:
    Converte expressões faladas e gírias de mesa em notação formal de RPG.
    """
    if not text:
        return ""

    out = text

    # Normalização de dados por extenso: "d vinte", "d seis", etc.
    words_to_dice = {
        "vinte": "20",
        "doze": "12",
        "dez": "10",
        "oito": "8",
        "seis": "6",
        "quatro": "4",
        "cem": "100",
    }
    for word, num in words_to_dice.items():
        out = re.sub(rf"\b[dD]\s+{word}\b", f"d{num}", out, flags=re.IGNORECASE)

    # Expressões com artigo: "um d 20", "1 d 20", "4 d 6"
    out = re.sub(r"\bum\s+[dD]\s*([0-9]{1,3})\b", r"1d\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\b([0-9]{1,2})\s+[dD]\s+([0-9]{1,3})\b", r"\1d\2", out)
    # Espaçamento em notação de dados: "d 20" -> "d20"
    out = re.sub(r"\b[dD]\s+([0-9]{1,3})\b", r"d\1", out)

    # Normalização de 20 e 1 natural
    out = re.sub(r"\b(?:vinte|20)\s+natural\b", "20 natural", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:um|1)\s+natural\b", "1 natural", out, flags=re.IGNORECASE)

    # Rolagem de d20: "tirei 19 no d20" -> "d20: 19"
    out = re.sub(r"\btirei\s+([0-9]{1,2})\s+no\s+d20\b", r"d20: \1", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:deu|foi)\s+([0-9]{1,2})\s+no\s+dado\b", r"dado: \1", out, flags=re.IGNORECASE)

    # Termos técnicos de atributos e regras
    out = re.sub(r"\bclasse\s+de\s+armadura\b", "CA", out, flags=re.IGNORECASE)
    out = re.sub(r"\bpontos\s+de\s+vida\b", "PV", out, flags=re.IGNORECASE)
    out = re.sub(r"\bclasse\s+de\s+dificuldade\b", "CD", out, flags=re.IGNORECASE)

    # Entidades customizadas
    if custom_entities:
        for ent in custom_entities:
            if ent:
                pat = re.compile(r"\b" + re.escape(ent) + r"\b", re.IGNORECASE)
                out = pat.sub(ent, out)

    return out


def apply_whisper_sensitive_vad(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    threshold_db: float = -42.0,
    hangover_ms: float = 300.0,
    return_mask: bool = False,
) -> np.ndarray | dict:
    """
    VAD de Alta Sensibilidade com Hangover Dinâmico:
    Preserva sussurros, respirações suaves e caudas de sílabas átonas frequentes em roleplay furtivo,
    eliminando ruído estático de fundo sem truncar consoantes finais.
    """
    if audio_data is None or len(audio_data) == 0:
        if return_mask:
            return {
                "audio": np.array([], dtype=np.float32),
                "voiced_mask": np.array([], dtype=bool),
                "speech_ratio": 0.0,
                "speech_intervals": [],
            }
        return np.array([], dtype=np.float32)

    audio = np.asarray(audio_data, dtype=np.float32)
    frame_len = int(sample_rate * 0.02)  # 20 ms
    hop_len = int(sample_rate * 0.01)    # 10 ms
    if len(audio) < frame_len:
        if return_mask:
            return {
                "audio": audio,
                "voiced_mask": np.ones(len(audio), dtype=bool),
                "speech_ratio": 1.0,
                "speech_intervals": [(0.0, len(audio) / float(sample_rate))],
            }
        return audio

    num_frames = (len(audio) - frame_len) // hop_len + 1
    energies_db = np.empty(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        energies_db[i] = 20.0 * np.log10(rms + 1e-12)

    speech_active = energies_db > threshold_db

    # Hangover time para proteger caudas de fala
    hangover_frames = int((hangover_ms / 1000.0) * (sample_rate / hop_len))
    active_mask = np.zeros(num_frames, dtype=bool)
    countdown = 0

    for i in range(num_frames):
        if speech_active[i]:
            active_mask[i] = True
            countdown = hangover_frames
        elif countdown > 0:
            active_mask[i] = True
            countdown -= 1

    # Interpolação suave de ganho
    gains = active_mask.astype(np.float32)
    sample_indices = np.arange(len(audio))
    frame_centers = np.arange(num_frames) * hop_len + (frame_len // 2)
    smooth_gain = np.interp(sample_indices, frame_centers, gains, left=0.0, right=0.0).astype(np.float32)

    # Suavização com filtro passa-baixas para evitar cliques de corte
    smooth_gain = np.clip(smooth_gain, 0.0, 1.0)
    b, a = signal.butter(1, min(0.45, 40.0 / (sample_rate / 2.0)), btype='low')
    smooth_gain = signal.filtfilt(b, a, smooth_gain.astype(np.float64)).astype(np.float32)
    smooth_gain = np.clip(smooth_gain, 0.0, 1.0)

    # Piso de atenuação de -24 dB em silêncio (não muta 100% para não soar artificial ao Whisper)
    floor_gain = 0.063
    final_gain = floor_gain + (1.0 - floor_gain) * smooth_gain
    processed_audio = (audio * final_gain).astype(np.float32)

    if return_mask:
        # Detecta intervalos de fala (start_sec, end_sec)
        intervals = []
        if len(active_mask) > 0:
            diffs = np.diff(active_mask.astype(np.int8))
            starts = np.where(diffs == 1)[0] + 1
            if active_mask[0]:
                starts = np.insert(starts, 0, 0)
            ends = np.where(diffs == -1)[0] + 1
            if active_mask[-1]:
                ends = np.append(ends, len(active_mask))
            for s, e in zip(starts, ends):
                intervals.append((float(s * hop_len / sample_rate), float(e * hop_len / sample_rate)))

        return {
            "audio": processed_audio,
            "voiced_mask": smooth_gain > 0.1,
            "speech_ratio": float(np.mean(active_mask)) if len(active_mask) > 0 else 0.0,
            "speech_intervals": intervals,
        }

    return processed_audio


def apply_plosive_suppression(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    cutoff_hz: float = 75.0,
    threshold: float = 0.6,
) -> np.ndarray:
    """
    Filtro Anti-Plosivas (De-Popper / Plosive Tamer):
    Atenua picos infra-acústicos (< 75 Hz) causados por golpes de ar na cápsula do microfone em 'p', 'b' e 't',
    preservando as frequências fundamentais da voz através de processamento em float64 de alta estabilidade.
    """
    if audio_data is None or len(audio_data) == 0:
        return np.array([], dtype=np.float32)

    audio = np.asarray(audio_data, dtype=np.float32)
    if len(audio) < int(sample_rate * 0.05):
        return audio

    # Filtro passa-altas de 2ª ordem estritamente em float64
    nyquist = sample_rate / 2.0
    norm_cutoff = min(0.95, max(0.001, cutoff_hz / nyquist))
    b, a = signal.butter(2, norm_cutoff, btype='highpass')

    # Detecta regiões com alta energia em sub-graves
    sub_b, sub_a = signal.butter(2, norm_cutoff, btype='lowpass')
    sub_bass = signal.filtfilt(sub_b, sub_a, audio.astype(np.float64))
    
    peak_val = np.max(np.abs(audio)) + 1e-6
    sub_energy_ratio = np.abs(sub_bass) / peak_val

    # Aplica atenuação suave apenas onde há transiente de sopro/plosiva
    highpassed = signal.filtfilt(b, a, audio.astype(np.float64)).astype(np.float32)
    plosive_weight = np.clip((sub_energy_ratio - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0).astype(np.float32)

    out = (1.0 - plosive_weight) * audio + plosive_weight * highpassed
    return out.astype(np.float32)


def apply_auto_ducking(
    voice_audio: np.ndarray,
    background_music: np.ndarray,
    sample_rate: int = 16000,
    ducking_db: float = -14.0,
    attack_ms: float = 40.0,
    release_ms: float = 400.0,
    return_mix: bool = True,
) -> np.ndarray:
    """
    Auto-Ducking Sidechain Dinâmico:
    Atenua a trilha sonora de fundo em ducking_db (-14 dB) com lookahead quando houver voz humana detectada,
    restaurando a música suavemente durante os silêncios e pausas.
    """
    if voice_audio is None or len(voice_audio) == 0:
        return np.asarray(background_music if background_music is not None else [], dtype=np.float32)
    if background_music is None or len(background_music) == 0:
        return np.asarray(voice_audio, dtype=np.float32)

    voice = np.asarray(voice_audio, dtype=np.float32)
    music = np.asarray(background_music, dtype=np.float32)

    length = max(len(voice), len(music))
    if len(voice) < length:
        voice = np.pad(voice, (0, length - len(voice)))
    if len(music) < length:
        music = np.pad(music, (0, length - len(music)))

    hop_len = int(sample_rate * 0.01)   # 10 ms
    frame_len = int(sample_rate * 0.03) # 30 ms
    num_frames = (length - frame_len) // hop_len + 1
    if num_frames <= 0:
        return (voice + music).astype(np.float32)

    voice_levels = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        st = i * hop_len
        chunk = voice[st:st + frame_len]
        voice_levels[i] = np.sqrt(np.mean(chunk ** 2) + 1e-12)

    threshold_rms = 0.02
    attenuation_factor = 10.0 ** (ducking_db / 20.0)

    attack_coeff = np.exp(-1.0 / max(1, int((attack_ms / 1000.0) * (sample_rate / hop_len))))
    release_coeff = np.exp(-1.0 / max(1, int((release_ms / 1000.0) * (sample_rate / hop_len))))

    ducking_gains = np.ones(num_frames, dtype=np.float32)
    current_gain = 1.0
    for i in range(num_frames):
        target = attenuation_factor if voice_levels[i] > threshold_rms else 1.0
        if target < current_gain:
            current_gain = attack_coeff * current_gain + (1.0 - attack_coeff) * target
        else:
            current_gain = release_coeff * current_gain + (1.0 - release_coeff) * target
        ducking_gains[i] = current_gain

    sample_indices = np.arange(length)
    frame_centers = np.arange(num_frames) * hop_len + (frame_len // 2)
    smooth_duck_curve = np.interp(sample_indices, frame_centers, ducking_gains, left=1.0, right=1.0).astype(np.float32)

    ducked_music = (music * smooth_duck_curve).astype(np.float32)
    if not return_mix:
        return ducked_music

    mixed = voice + ducked_music
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    return mixed.astype(np.float32)


def apply_cross_bleed_cancellation(
    channel_matrix: np.ndarray | list[np.ndarray],
    sample_rate: int = 16000,
    bleed_suppression_db: float = -12.0,
) -> np.ndarray | list[np.ndarray]:
    """
    Cancelamento de Vazamento Cruzado (Cross-Bleed & Cross-Talk Suppression):
    Em gravações presenciais com múltiplos microfones, detecta qual canal detém a voz primária
    em cada janela temporal e atenua o vazamento acústico nos microfones vizinhos.
    """
    channels = split_multichannel_audio(channel_matrix)
    num_channels = len(channels)
    if num_channels <= 1:
        return channel_matrix

    min_len = min(len(ch) for ch in channels)
    if min_len < int(sample_rate * 0.05):
        return channel_matrix

    channels = [ch[:min_len] for ch in channels]
    frame_len = int(sample_rate * 0.03)  # 30 ms
    hop_len = int(sample_rate * 0.015)   # 15 ms
    num_frames = (min_len - frame_len) // hop_len + 1

    attenuation_factor = 10.0 ** (bleed_suppression_db / 20.0)

    channel_energies = np.zeros((num_channels, num_frames), dtype=np.float32)
    for ch_idx in range(num_channels):
        for f in range(num_frames):
            st = f * hop_len
            frame = channels[ch_idx][st:st + frame_len]
            channel_energies[ch_idx, f] = np.sum(frame ** 2)

    dominant_channel = np.argmax(channel_energies, axis=0)
    cleaned_channels = []

    frame_centers = np.arange(num_frames) * hop_len + (frame_len // 2)
    sample_indices = np.arange(min_len)

    for ch_idx in range(num_channels):
        gains = np.ones(num_frames, dtype=np.float32)
        for f in range(num_frames):
            dom_idx = dominant_channel[f]
            if dom_idx != ch_idx:
                dom_e = channel_energies[dom_idx, f]
                my_e = channel_energies[ch_idx, f]
                if dom_e > my_e * 2.0:
                    gains[f] = attenuation_factor

        smooth_gain = np.interp(sample_indices, frame_centers, gains, left=1.0, right=1.0).astype(np.float32)
        cleaned_channels.append((channels[ch_idx] * smooth_gain).astype(np.float32))

    if isinstance(channel_matrix, np.ndarray) and channel_matrix.ndim == 2:
        return np.stack(cleaned_channels, axis=1 if channel_matrix.shape[1] == num_channels else 0).astype(np.float32)
    return cleaned_channels


def acting_invariant_feature_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Distância Acústica Tolerante a Interpretação Vocal (Acting-Invariant Distance):
    Calcula a distância entre duas impressões digitais de voz reduzindo a penalidade
    de desvios em F0 (pitch) e enfatizando a ressonância anatômica do trato vocal (MFCCs).
    """
    a1 = np.asarray(feat1, dtype=np.float32)
    a2 = np.asarray(feat2, dtype=np.float32)
    if len(a1) != len(a2) or len(a1) == 0:
        return 1.0

    weights = np.ones_like(a1)
    if len(a1) == 38:
        weights[:20] = 1.6   # MFCCs
        weights[20:30] = 1.0 # Centroid, Bandwidth, Contrast
        weights[-1] = 0.20   # Pitch / F0 reduzido para tolerar interpretação

    norm1 = np.linalg.norm(a1 * weights) + 1e-7
    norm2 = np.linalg.norm(a2 * weights) + 1e-7
    weighted_cos = float(np.dot(a1 * weights, a2 * weights) / (norm1 * norm2))
    
    dist = max(0.0, min(1.0, 1.0 - weighted_cos))
    return float(dist)


def list_audio_input_devices(sd_module=None) -> list[dict]:
    """
    Lista todos os dispositivos de áudio físicos e virtuais com capacidade de entrada (microfone),
    incluindo índice, nome amigável, canais de entrada, taxa nativa e flag de dispositivo padrão.
    """
    if sd_module is None:
        try:
            import sounddevice as sd_module
        except Exception:
            return [{"index": 0, "name": "Dispositivo Padrão", "channels": 1, "samplerate": 16000, "is_default": True, "display": "[0] Dispositivo Padrão"}]

    try:
        devices = sd_module.query_devices()
    except Exception:
        return [{"index": 0, "name": "Dispositivo Padrão", "channels": 1, "samplerate": 16000, "is_default": True, "display": "[0] Dispositivo Padrão"}]

    hostapis = []
    try:
        if hasattr(sd_module, "query_hostapis"):
            hostapis = sd_module.query_hostapis()
    except Exception:
        hostapis = []

    default_in = None
    try:
        if hasattr(sd_module, "default") and hasattr(sd_module.default, "device"):
            default_in = sd_module.default.device[0]
    except Exception:
        default_in = None

    input_list = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            api_name = ""
            try:
                if hostapis and dev.get("hostapi", -1) < len(hostapis):
                    api_name = hostapis[dev["hostapi"]]["name"]
            except Exception:
                pass
            is_default = (default_in is not None and idx == default_in) or (default_in is None and idx == 0)
            input_list.append({
                "index": idx,
                "name": dev["name"],
                "api": api_name,
                "channels": dev["max_input_channels"],
                "samplerate": int(dev.get("default_samplerate", 16000)),
                "is_default": is_default,
                "display": f"{idx}: {dev['name']}" + (f" ({api_name})" if api_name else "") + (" ★ [Padrão]" if is_default else ""),
            })

    if not input_list:
        input_list.append({
            "index": 0,
            "name": "Dispositivo Padrão",
            "channels": 1,
            "samplerate": 16000,
            "is_default": True,
            "display": "0: Dispositivo Padrão"
        })
    return input_list


def play_audio_slice(
    audio_path: Path | str,
    start_sec: float,
    end_sec: float,
    sd_module=None,
    block: bool = False,
) -> bool:
    """
    Reproduz de forma assíncrona um trecho fatiado de áudio [start_sec, end_sec],
    permitindo ao usuário revisar imediatamente falas específicas clicadas na transcrição.
    """
    src = Path(audio_path).resolve()
    if not src.exists():
        return False

    if sd_module is None:
        try:
            import sounddevice as sd_module
        except Exception:
            return False

    try:
        sr, data = wavfile.read(src)
        start_idx = max(0, int(start_sec * sr))
        end_idx = min(len(data), int(end_sec * sr))
        if end_idx <= start_idx:
            return False

        slice_data = data[start_idx:end_idx]
        if hasattr(sd_module, "stop"):
            try:
                sd_module.stop()
            except Exception:
                pass
        sd_module.play(slice_data, samplerate=sr)
        if block and hasattr(sd_module, "wait"):
            sd_module.wait()
        return True
    except Exception:
        return False


def generate_interactive_lore_graph_html(
    graph_dict: dict,
    mermaid_code: str = "",
    campaign_name: str = "Campanha de RPG",
    output_path: Path | str | None = None,
    **kwargs,
) -> Path:
    """
    Gera um visualizador HTML interativo, autocontido e responsivo do Grafo de Lore & Relações,
    permitindo zoom, pan, filtros por tipo (heróis, npcs, itens, locais, facções) e busca textual.
    """
    effective_campaign = kwargs.get("campanha_name", campaign_name)
    nodes = list(graph_dict.get("nodes", [])) if isinstance(graph_dict, dict) else []
    edges = list(graph_dict.get("edges", [])) if isinstance(graph_dict, dict) else []

    if not nodes and isinstance(graph_dict, dict):
        for c in graph_dict.get("characters", []):
            nodes.append({"id": str(c), "label": str(c), "type": "player" if "Jogador" in str(c) else "npc"})
        for f in graph_dict.get("factions", []):
            nodes.append({"id": str(f), "label": str(f), "type": "faction"})
        for loc in graph_dict.get("locations", []):
            nodes.append({"id": str(loc), "label": str(loc), "type": "location"})
        for it in graph_dict.get("items", []):
            nodes.append({"id": str(it), "label": str(it), "type": "item"})

    if not edges and isinstance(graph_dict, dict):
        for r in graph_dict.get("relationships", []):
            if isinstance(r, dict):
                edges.append(r)

    if not mermaid_code and (nodes or edges):
        # Gera mermaid básico caso venha apenas o dict
        lines = ["graph TD"]
        for n in nodes:
            nid = n.get("id", "node")
            label = n.get("label", nid)
            lines.append(f'    {nid}["{label}"]')
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            lbl = e.get("relation", "")
            if src and tgt:
                lines.append(f'    {src} -->|"{lbl}"| {tgt}' if lbl else f'    {src} --> {tgt}')
        mermaid_code = "\n".join(lines)
    elif not mermaid_code:
        mermaid_code = "graph TD\n    A[Sem Dados de Grafo]"

    # Estatísticas de entidades
    counts = {}
    for n in nodes:
        nt = n.get("type", "outro")
        counts[nt] = counts.get(nt, 0) + 1

    esc_campaign = html.escape(effective_campaign)
    esc_mermaid = html.escape(mermaid_code)
    nodes_json = json.dumps(nodes, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grafo de Lore & Relações — {esc_campaign}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --gold: #f59e0b;
            --accent: #8b5cf6;
            --text: #f0f6fc;
            --text-muted: #8b949e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 24px;
            min-height: 100vh;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 20px;
        }}
        h1 {{ font-size: 1.5rem; color: var(--gold); }}
        .badge-bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }}
        .badge {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        .badge strong {{ color: var(--gold); }}
        .controls {{
            display: flex;
            gap: 12px;
            align-items: center;
            margin-bottom: 16px;
        }}
        .btn {{
            background: var(--accent);
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        .btn:hover {{ opacity: 0.85; }}
        .btn-secondary {{ background: var(--card-bg); border: 1px solid var(--border); color: var(--text); }}
        .canvas-container {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            overflow: auto;
            min-height: 500px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .mermaid {{ width: 100%; text-align: center; }}
        .entities-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 12px;
            margin-top: 24px;
        }}
        .entity-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 12px 16px;
            border-radius: 8px;
        }}
        .entity-type {{ font-size: 0.75rem; text-transform: uppercase; color: var(--gold); font-weight: bold; }}
        .entity-name {{ font-size: 1rem; font-weight: 600; margin-top: 4px; }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>🗺️ Grafo de Lore & Relações de Campanha</h1>
            <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 4px;">{esc_campaign}</p>
        </div>
        <div class="controls">
            <button class="btn btn-secondary" onclick="window.print()">🖨️ Imprimir / Salvar PDF</button>
            <button class="btn" onclick="location.reload()">🔄 Atualizar</button>
        </div>
    </header>

    <div class="badge-bar">
        <div class="badge">Total de Nós: <strong>{len(nodes)}</strong></div>
        <div class="badge">Conexões: <strong>{len(edges)}</strong></div>
        <div class="badge">⚔️ Jogadores: <strong>{counts.get('player', 0)}</strong></div>
        <div class="badge">👤 NPCs: <strong>{counts.get('npc', 0)}</strong></div>
        <div class="badge">💎 Itens: <strong>{counts.get('item', 0)}</strong></div>
        <div class="badge">🏰 Locais: <strong>{counts.get('location', 0)}</strong></div>
        <div class="badge">🛡️ Fações: <strong>{counts.get('faction', 0)}</strong></div>
    </div>

    <div class="canvas-container" id="graph-container">
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>

    <h2 style="margin-top: 24px; font-size: 1.1rem; color: var(--gold);">📋 Entidades & Elementos Identificados</h2>
    <div class="entities-grid" id="entities-grid"></div>

    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            themeVariables: {{
                primaryColor: '#7c3aed',
                primaryTextColor: '#ffffff',
                lineColor: '#f59e0b',
                secondaryColor: '#1e293b',
                tertiaryColor: '#0f172a'
            }}
        }});

        const nodesData = {nodes_json};
        const grid = document.getElementById('entities-grid');
        if (nodesData.length === 0) {{
            grid.innerHTML = '<p style="color: var(--text-muted);">Nenhuma entidade mapeada ainda.</p>';
        }} else {{
            nodesData.forEach(node => {{
                const card = document.createElement('div');
                card.className = 'entity-card';
                card.innerHTML = `
                    <div class="entity-type">${{node.type || 'Elemento'}}</div>
                    <div class="entity-name">${{node.label || node.id}}</div>
                `;
                grid.appendChild(card);
            }});
        }}
    </script>
</body>
</html>
"""
    if output_path is None:
        out_p = Path(tempfile.gettempdir()) / f"rpg_lore_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    else:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

    out_p.write_text(html_content, encoding="utf-8")
    return out_p


def export_session_publishing_bundle(
    session_dir: Path | str,
    campaign_info: dict | None = None,
    output_zip: Path | str | None = None,
) -> Path:
    """
    Exportador de Pacote Completo da Sessão (Publishing Hub):
    Empacota áudio masterizado, transcrição, legendas karaoke, relatórios de combate,
    show notes, cortes virais e portal web index.html em um arquivo .ZIP estruturado.
    """
    sdir = Path(session_dir).resolve()
    if not sdir.exists():
        raise FileNotFoundError(f"Diretório da sessão não encontrado: {sdir}")

    info = campaign_info or {}
    campanha = info.get("campanha", "Campanha de RPG")
    sessao = info.get("sessao", "Sessão")

    if output_zip is None:
        zip_name = f"{sdir.stem}_pacote_publicacao.zip"
        zip_path = sdir / zip_name
    else:
        zip_path = Path(output_zip).resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Cria diretório de staging temporário
    with tempfile.TemporaryDirectory(prefix="rpg-bundle-") as temp_stage_str:
        stage_dir = Path(temp_stage_str)
        (stage_dir / "audio").mkdir(exist_ok=True)
        (stage_dir / "subtitles").mkdir(exist_ok=True)
        (stage_dir / "documents").mkdir(exist_ok=True)
        (stage_dir / "interactive").mkdir(exist_ok=True)

        # 1. Áudios
        for f in sdir.glob("*.wav"):
            shutil.copy2(f, stage_dir / "audio" / f.name)
        for f in sdir.glob("*.mp3"):
            shutil.copy2(f, stage_dir / "audio" / f.name)
        for f in sdir.glob("*.mp4"):
            shutil.copy2(f, stage_dir / "audio" / f.name)

        # 2. Legendas
        for f in sdir.glob("*.srt"):
            shutil.copy2(f, stage_dir / "subtitles" / f.name)
        for f in sdir.glob("*.vtt"):
            shutil.copy2(f, stage_dir / "subtitles" / f.name)
        for f in sdir.glob("*.ass"):
            shutil.copy2(f, stage_dir / "subtitles" / f.name)

        # 3. Documentos
        for f in sdir.glob("*.md"):
            shutil.copy2(f, stage_dir / "documents" / f.name)
        for f in sdir.glob("*.json"):
            if f.name not in {"config.json"}:
                shutil.copy2(f, stage_dir / "documents" / f.name)
        for f in sdir.glob("*.mmd"):
            shutil.copy2(f, stage_dir / "documents" / f.name)

        # 4. Grafo Interativo
        lore_graph_json = sdir / "lore_graph.json"
        lore_mmd = sdir / "lore_graph.mmd"
        mmd_content = lore_mmd.read_text(encoding="utf-8") if lore_mmd.exists() else ""
        graph_dict = {}
        if lore_graph_json.exists():
            try:
                graph_dict = json.loads(lore_graph_json.read_text(encoding="utf-8"))
            except Exception:
                pass
        generate_interactive_lore_graph_html(
            graph_dict,
            mermaid_code=mmd_content,
            campaign_name=f"{campanha} — {sessao}",
            output_path=stage_dir / "interactive" / "lore_graph.html",
        )

        # 5. Portal da Sessão (index.html)
        notes_txt = ""
        show_notes_file = sdir / "show_notes.md"
        if show_notes_file.exists():
            notes_txt = show_notes_file.read_text(encoding="utf-8")

        portal_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(campanha)} — {html.escape(sessao)}</title>
    <style>
        body {{
            background: #0d1117;
            color: #f0f6fc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 32px 20px;
            line-height: 1.6;
        }}
        h1 {{ color: #ffd166; font-size: 2rem; margin-bottom: 8px; }}
        h2 {{ color: #7c3aed; margin-top: 32px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
        .player-box {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        audio {{ width: 100%; border-radius: 8px; }}
        .btn-link {{
            display: inline-block;
            background: #7c3aed;
            color: #fff;
            padding: 10px 18px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: bold;
            margin-top: 10px;
        }}
        .file-list {{ list-style: none; padding: 0; }}
        .file-list li {{
            background: #161b22;
            padding: 10px 14px;
            margin: 6px 0;
            border-radius: 6px;
            border: 1px solid #30363d;
        }}
        .file-list a {{ color: #58a6ff; text-decoration: none; font-weight: 500; }}
    </style>
</head>
<body>
    <h1>🎲 {html.escape(campanha)}</h1>
    <h3 style="color: #8b949e; margin-bottom: 24px;">{html.escape(sessao)}</h3>

    <div class="player-box">
        <h3>🎙️ Áudio Masterizado da Sessão</h3>
        <audio controls src="audio/podcast_master.wav"></audio>
        <p style="font-size: 0.85rem; color: #8b949e;">Normalizado em -16 LUFS (EBU R128) com De-Popper, De-Esser e EQ de Podcast.</p>
    </div>

    <div>
        <a class="btn-link" href="interactive/lore_graph.html" target="_blank">🗺️ Visualizar Grafo de Lore & Relações</a>
    </div>

    <h2>📁 Arquivos & Artefatos Incluídos</h2>
    <ul class="file-list">
        <li>📜 <strong>Documentos:</strong> <a href="documents/show_notes.md">Show Notes</a> | <a href="documents/combat_stats.md">Estatísticas de Combate</a> | <a href="documents/viral_clips.md">Roteiro de Cortes Virais</a></li>
        <li>💬 <strong>Legendas Sincronizadas:</strong> <a href="subtitles/">Arquivos .SRT, .VTT e .ASS Karaoke</a></li>
        <li>🌐 <strong>Interativo:</strong> <a href="interactive/lore_graph.html">Grafo Visual de Campanha (HTML)</a></li>
    </ul>
</body>
</html>
"""
        (stage_dir / "index.html").write_text(portal_html, encoding="utf-8")

        # Cria o arquivo ZIP
        with zipfile.ZipFile(str(zip_path), 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for root_dir, _, files in os.walk(stage_dir):
                for f in files:
                    full_p = Path(root_dir) / f
                    rel_p = full_p.relative_to(stage_dir)
                    zf.write(full_p, arcname=str(rel_p))

    return zip_path


def detect_ollama_local_models(endpoint: str = "http://localhost:11434", timeout: float = 1.5) -> list[str]:
    """
    Detecta modelos instalados no serviço Ollama local com verificação rápida e timeout curto.
    Retorna lista de nomes de modelos disponíveis ou vazia caso o serviço não esteja ativo.
    """
    url = endpoint.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RPGChronicler/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            st = getattr(resp, "status", getattr(resp, "code", 200))
            if st in (200, None):
                payload = resp.read().decode("utf-8")
                data = json.loads(payload)
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return sorted(models)
    except Exception:
        pass
    return []


# =====================================================================
# 1. GERENCIADOR DE MARCADORES DE LINHA DO TEMPO (HIGHLIGHTS / HOTKEYS)
# =====================================================================

@dataclass
class TimelineHighlight:
    timestamp: float
    category: str = "epic"  # 'epic', 'critical', 'twist', 'note'
    description: str = ""
    formatted_time: str = ""

    def __post_init__(self):
        if not self.formatted_time:
            self.formatted_time = format_timestamp(max(0.0, float(self.timestamp)), decimal=".")[:8]


class TimelineHighlightManager:
    """Gerencia marcadores em tempo real durante a gravação da sessão de RPG.

    Permite salvar momentos épicos, acertos/erros críticos e reviravoltas de roteiro
    com timestamps precisos para orientar cortes virais e edição de podcast.
    """

    def __init__(self):
        self.highlights: list[TimelineHighlight] = []

    def add_highlight(self, timestamp: float, category: str = "epic", description: str = "") -> TimelineHighlight:
        timestamp = max(0.0, float(timestamp))
        category = (category or "epic").strip().lower()
        hl = TimelineHighlight(timestamp=timestamp, category=category, description=description)
        self.highlights.append(hl)
        self.highlights.sort(key=lambda h: h.timestamp)
        return hl

    def to_dict(self) -> list[dict]:
        return [
            {
                "timestamp": round(h.timestamp, 2),
                "category": h.category,
                "description": h.description,
                "formatted_time": h.formatted_time,
            }
            for h in self.highlights
        ]

    def from_dict(self, data: list[dict]):
        self.highlights = [
            TimelineHighlight(
                timestamp=float(d.get("timestamp", 0.0)),
                category=str(d.get("category", "epic")),
                description=str(d.get("description", "")),
                formatted_time=str(d.get("formatted_time", "")),
            )
            for d in data
        ]
        self.highlights.sort(key=lambda h: h.timestamp)

    def export_markers_txt(self, output_path: str | Path) -> Path:
        """Exporta marcadores compatíveis com Audacity label track e texto de capítulos."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for h in self.highlights:
            label = f"[{h.category.upper()}] {h.description}".strip()
            lines.append(f"{h.timestamp:.3f}\t{h.timestamp + 5.0:.3f}\t{label}")
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def clear(self):
        self.highlights.clear()


# =====================================================================
# 2. COMPRESSOR INTELIGENTE DE ARMAZENAMENTO DE ÁUDIO (FLAC/OPUS/MP3)
# =====================================================================

def compress_audio_archive(
    audio_path: str | Path,
    target_format: str = "flac",
    output_path: str | Path | None = None,
    remove_source: bool = False,
) -> dict:
    """Comprime gravações de áudio WAV para formatos lossless (FLAC) ou lossy eficientes (Opus/MP3).

    Utiliza FFmpeg nativo do sistema para máxima fidelidade e velocidade.
    Retorna métricas detalhadas de compressão e bytes economizados.
    """
    src = Path(audio_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {src}")

    target_format = target_format.lower().strip()
    if target_format not in ("flac", "opus", "mp3"):
        raise ValueError(f"Formato de compressão não suportado: {target_format}. Use 'flac', 'opus' ou 'mp3'.")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg não encontrado no PATH do sistema. Instale o FFmpeg para compressão de áudio.")

    if output_path is None:
        dst = src.with_suffix(f".{target_format}")
    else:
        dst = Path(output_path)

    dst.parent.mkdir(parents=True, exist_ok=True)

    orig_size = src.stat().st_size

    cmd = [ffmpeg_bin, "-y", "-i", str(src)]
    if target_format == "flac":
        cmd.extend(["-c:a", "flac", "-compression_level", "8"])
    elif target_format == "opus":
        cmd.extend(["-c:a", "libopus", "-b:a", "96k"])
    elif target_format == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-q:a", "2"])

    cmd.append(str(dst))

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if res.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Falha na compressão FFmpeg ({res.returncode}): {res.stderr[-300:] if res.stderr else ''}")

    new_size = dst.stat().st_size
    saved_bytes = max(0, orig_size - new_size)
    saved_ratio = (saved_bytes / max(1, orig_size)) * 100.0

    if remove_source and src.resolve() != dst.resolve():
        try:
            src.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "source_path": str(src),
        "target_path": str(dst),
        "format": target_format,
        "orig_size_bytes": orig_size,
        "new_size_bytes": new_size,
        "saved_bytes": saved_bytes,
        "compression_ratio_pct": round(saved_ratio, 2),
    }


# =====================================================================
# 3. EXPORTADORES OBSIDIAN (COM WIKILINKS) & FOUNDRY VTT (JOURNAL ENTRY)
# =====================================================================

def export_to_obsidian_vault(
    session_data: dict,
    output_vault_dir: str | Path,
    lore_entities: list[str] | None = None,
) -> dict:
    """Exporta sessão de RPG em formato Obsidian Vault com suporte a Wikilinks [[Entidade]].

    Cria pastas 'Sessoes', 'Entidades' e 'Destaques' com frontmatter YAML canônico.
    """
    vault = Path(output_vault_dir)
    sessoes_dir = vault / "Sessoes"
    entidades_dir = vault / "Entidades"
    destaques_dir = vault / "Destaques"
    for d in (sessoes_dir, entidades_dir, destaques_dir):
        d.mkdir(parents=True, exist_ok=True)

    title = session_data.get("title", "Sessão Sem Título")
    date_str = session_data.get("date", datetime.now().strftime("%Y-%m-%d"))
    duration = session_data.get("duration", "N/A")
    summary = session_data.get("summary", "")
    segments = session_data.get("segments", [])
    highlights = session_data.get("highlights", [])
    entities = list(lore_entities or session_data.get("lore_entities", []))

    # Wikilink replacement helper (longer entities first to prevent nested substrings)
    sorted_entities = sorted(set(e.strip() for e in entities if len(e.strip()) >= 2), key=len, reverse=True)

    def inject_wikilinks(text: str) -> str:
        if not sorted_entities or not text:
            return text
        result = text
        for ent in sorted_entities:
            pattern = re.compile(rf"(?<!\[\[)\b({re.escape(ent)})\b(?!\]\])", re.IGNORECASE)
            result = pattern.sub(rf"[[\1]]", result)
        return result

    # 1. Nota da Sessão
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title).strip() or "Sessao"
    session_file = sessoes_dir / f"{safe_title}.md"

    frontmatter = f"""---
title: "{title}"
date: {date_str}
duration: "{duration}"
tags:
  - rpg/sessao
  - chronicler
---
"""
    body = f"""# {title}

**Data:** {date_str}  
**Duração:** {duration}  

## 📜 Crônica da Sessão
{inject_wikilinks(summary)}

## 🎙️ Transcrição por Diálogo
"""
    for seg in segments:
        spk = seg.get("speaker", "Voz")
        t0 = seg.get("start", 0.0)
        t_fmt = format_timestamp(t0, decimal=".")[:8] if isinstance(t0, (int, float)) else str(t0)
        text_line = inject_wikilinks(seg.get("text", ""))
        body += f"- **[{t_fmt}] {spk}:** {text_line}\n"

    session_file.write_text(frontmatter + "\n" + body, encoding="utf-8")

    # 2. Notas de Entidades
    created_entities = 0
    for ent in sorted_entities:
        safe_ent = re.sub(r'[\\/*?:"<>|]', "", ent).strip()
        if safe_ent:
            ent_file = entidades_dir / f"{safe_ent}.md"
            if not ent_file.exists():
                ent_content = f"""---
name: "{safe_ent}"
type: entity
tags:
  - rpg/entidade
---

# {safe_ent}

*Mencionado inicialmente na sessão [[{safe_title}]].*

## Descrição e Lore
*(Adicione notas sobre antecedentes, objetivos e alianças de {safe_ent} aqui)*

## Sessões com Aparições
- [[{safe_title}]]
"""
                ent_file.write_text(ent_content, encoding="utf-8")
                created_entities += 1

    # 3. Nota de Destaques
    if highlights:
        hl_file = destaques_dir / f"Destaques - {safe_title}.md"
        hl_content = f"""# ⭐ Destaques Épicos - {title}

| Tempo | Categoria | Descrição |
|---|---|---|
"""
        for h in highlights:
            t_fmt = h.get("formatted_time") or (format_timestamp(h.get("timestamp", 0.0), decimal=".")[:8] if isinstance(h.get("timestamp"), (int, float)) else "00:00:00")
            cat = h.get("category", "epic")
            desc = h.get("description", "")
            hl_content += f"| `{t_fmt}` | **{cat.upper()}** | {inject_wikilinks(desc)} |\n"
        hl_file.write_text(hl_content, encoding="utf-8")

    return {
        "vault_dir": str(vault),
        "session_file": str(session_file),
        "entities_created": created_entities,
        "has_highlights": bool(highlights),
    }


def export_to_foundry_vtt(session_data: dict, output_file: str | Path) -> Path:
    """Exporta sessão de RPG em formato JournalEntry do Foundry VTT (v10+).

    Gera um arquivo JSON importável diretamente na aba de Diário (Journals) do Foundry.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = session_data.get("title", "Sessão de RPG")
    date_str = session_data.get("date", datetime.now().strftime("%Y-%m-%d"))
    summary = session_data.get("summary", "")
    segments = session_data.get("segments", [])
    highlights = session_data.get("highlights", [])

    # Html summary page
    summary_html = f"<h2>Resumo da Sessão ({date_str})</h2>"
    for p in summary.split("\n\n"):
        if p.strip():
            summary_html += f"<p>{html.escape(p.strip())}</p>"

    # Html highlights page
    highlights_html = "<h2>⭐ Destaques e Momentos Épicos</h2><ul>"
    if highlights:
        for h in highlights:
            t_fmt = h.get("formatted_time", "00:00:00")
            cat = h.get("category", "epic")
            desc = h.get("description", "")
            highlights_html += f"<li><strong>[{html.escape(t_fmt)}] ({html.escape(cat.upper())}):</strong> {html.escape(desc)}</li>"
    else:
        highlights_html += "<li>Nenhum destaque manual registrado.</li>"
    highlights_html += "</ul>"

    # Html transcription page
    transcription_html = "<h2>🎙️ Transcrição Completa</h2><div class='dialogue-log'>"
    for seg in segments:
        spk = seg.get("speaker", "Voz")
        t0 = seg.get("start", 0.0)
        t_fmt = format_timestamp(t0, decimal=".")[:8] if isinstance(t0, (int, float)) else str(t0)
        text_line = seg.get("text", "")
        transcription_html += f"<p><span style='color: #4a90e2; font-weight: bold;'>[{t_fmt}] {html.escape(spk)}:</span> {html.escape(text_line)}</p>"
    transcription_html += "</div>"

    journal_entry = {
        "name": f"{title} ({date_str})",
        "pages": [
            {
                "name": "Crônica & Resumo",
                "type": "text",
                "text": {
                    "content": summary_html,
                    "format": 1,  # CONST.JOURNAL_ENTRY_PAGE_FORMATS.HTML
                },
                "title": {"show": True, "level": 1},
            },
            {
                "name": "Momentos Épicos",
                "type": "text",
                "text": {
                    "content": highlights_html,
                    "format": 1,
                },
                "title": {"show": True, "level": 1},
            },
            {
                "name": "Registro de Diálogo",
                "type": "text",
                "text": {
                    "content": transcription_html,
                    "format": 1,
                },
                "title": {"show": True, "level": 1},
            },
        ],
        "flags": {
            "rpg_chronicler": {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
            }
        },
    }

    out_path.write_text(json.dumps(journal_entry, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# =====================================================================
# 4. GERADOR DE NARRAÇÃO DE RESUMO POR VOZ (TTS OFFLINE / OPENAI)
# =====================================================================

def generate_session_narration_tts(
    text: str,
    output_wav: str | Path,
    voice_name: str | None = None,
    openai_client: Any = None,
    model: str = "tts-1",
    engine: str = "system",
) -> Path:
    """Gera arquivo WAV narrando o resumo da sessão.

    Suporta:
    - 'system': Windows Speech API nativa (System.Speech.Synthesis) via PowerShell sem dependências pip.
    - 'openai': OpenAI TTS API (caso cliente OpenAI esteja configurado).
    """
    out = Path(output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)

    text_clean = text.strip()
    if not text_clean:
        raise ValueError("Texto para narração está vazio.")

    if engine == "openai" and openai_client:
        voice = voice_name or "onyx"
        response = openai_client.audio.speech.create(
            model=model,
            voice=voice,
            input=text_clean[:4096],
        )
        response.stream_to_file(str(out))
        return out

    # Engine nativa Windows via PowerShell System.Speech
    ps_script = f"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
"""
    if voice_name:
        ps_script += f"""
try {{ $synth.SelectVoice('{voice_name}') }} catch {{}}
"""
    temp_txt = out.with_suffix(".temp_tts.txt")
    temp_txt.write_text(text_clean, encoding="utf-8")

    ps_script += f"""
$content = Get-Content -Path '{temp_txt}' -Raw -Encoding UTF8
$synth.SetOutputToWaveFile('{out}')
$synth.Speak($content)
$synth.Dispose()
"""
    ps_file = out.with_suffix(".temp_tts.ps1")
    ps_file.write_text(ps_script, encoding="utf-8")

    try:
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_file)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"Falha na síntese de voz nativa: {res.stderr}")
    finally:
        ps_file.unlink(missing_ok=True)
        temp_txt.unlink(missing_ok=True)

    return out


# =====================================================================
# 5. PAINEL COMPANION WEB LOCAL WI-FI (HTTP.SERVER ZERO-DEPENDENCY)
# =====================================================================

def ensure_companion_ssl_cert(cert_dir: Path | str | None = None) -> tuple[str, str]:
    """Gera um certificado SSL autoassinado local para liberar Web Audio / getUserMedia em celulares na LAN."""
    if cert_dir is None:
        cert_dir = Path.home() / ".rpg_chronicler" / "ssl"
    else:
        cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = cert_dir / "companion_cert.pem"
    key_file = cert_dir / "companion_key.pem"
    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "RPG Chronicler Companion"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RPG Chronicler"),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=730))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(socket.inet_aton("127.0.0.1")),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_file.write_bytes(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        return str(cert_file), str(key_file)
    except Exception as exc:
        raise RuntimeError(f"Não foi possível gerar certificado SSL local: {exc}")


class RPGCompanionHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Handler HTTP para o painel companion local via Wi-Fi."""

    server_ref: Any = None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/session":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = self.server_ref.get_session_data() if self.server_ref else {}
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        elif parsed.path == "/api/participants":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            participants = self.server_ref.get_participants() if self.server_ref else []
            self.wfile.write(json.dumps({"participants": participants}, ensure_ascii=False).encode("utf-8"))
        elif parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_page = self.server_ref.render_dashboard_html() if self.server_ref else "<h1>RPG Chronicler</h1>"
            self.wfile.write(html_page.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8", errors="replace") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if parsed.path == "/api/highlight":
            if self.server_ref:
                hl = self.server_ref.on_remote_highlight(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "highlight": hl}, ensure_ascii=False).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()
        elif parsed.path == "/api/voice/train":
            if self.server_ref:
                res = self.server_ref.on_remote_voice_train(payload)
                status_code = 200 if res.get("status") == "ok" else 400
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()
        elif parsed.path == "/api/satellite/chunk":
            if self.server_ref:
                res = self.server_ref.on_remote_satellite_chunk(payload)
                status_code = 200 if res.get("status") == "ok" else 400
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Silencia logs no terminal para não poluir o console do app


class RPGCompanionWebServer:
    """Servidor web local leve (sem frameworks externos) para jogadores acompanharem a sessão no celular."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, use_https: bool = False):
        self.host = host
        self.port = port
        self.use_https = use_https
        self.server: http.server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.highlight_callback: Any = None
        self.voice_train_callback: Any = None
        self.satellite_chunk_callback: Any = None
        self.get_participants_callback: Any = None
        self.session_data: dict = {
            "title": "Sessão Ativa",
            "status": "Gravando...",
            "duration": "00:00:00",
            "recent_dialogues": [],
            "highlights": [],
            "summary": "Sessão em andamento.",
        }

    def start(self, use_https: bool | None = None) -> bool:
        if self.server is not None:
            return True
        if use_https is not None:
            self.use_https = use_https
        try:
            handler_class = RPGCompanionHTTPHandler
            handler_class.server_ref = self
            self.server = http.server.ThreadingHTTPServer((self.host, self.port), handler_class)
            if self.use_https:
                cert_file, key_file = ensure_companion_ssl_cert()
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
                self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            return True
        except Exception:
            self.server = None
            return False

    def stop(self):
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
            self.server = None
            self.thread = None

    def is_running(self) -> bool:
        return self.server is not None

    def update_session_data(self, data: dict):
        with self.lock:
            self.session_data.update(data)

    def get_session_data(self) -> dict:
        with self.lock:
            return dict(self.session_data)

    def get_participants(self) -> list[dict]:
        if callable(self.get_participants_callback):
            try:
                return self.get_participants_callback() or []
            except Exception:
                return []
        return []

    def on_remote_highlight(self, payload: dict) -> dict:
        category = payload.get("category", "epic")
        description = payload.get("description", "Marcado via Celular")
        if callable(self.highlight_callback):
            return self.highlight_callback(category, description)
        return {"category": category, "description": description}

    def on_remote_voice_train(self, payload: dict) -> dict:
        player_label = (payload.get("player_label") or payload.get("label") or "").strip()
        audio_b64 = payload.get("audio_b64") or payload.get("audio", "")
        if not player_label:
            return {"status": "error", "message": "Nenhum perfil de personagem selecionado."}
        if not audio_b64:
            return {"status": "error", "message": "Nenhum dado de áudio foi recebido."}
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as exc:
            return {"status": "error", "message": f"Falha ao decodificar áudio: {exc}"}

        filename = payload.get("filename") or "training.webm"
        suffix = Path(filename).suffix or ".webm"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_file.close()

            if callable(self.voice_train_callback):
                res = self.voice_train_callback(player_label, temp_file.name)
                return res if isinstance(res, dict) else {"status": "ok", "message": f"Perfil '{player_label}' treinado com sucesso!"}
            return {"status": "ok", "message": "Áudio de treino recebido com sucesso."}
        finally:
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception:
                pass

    def on_remote_satellite_chunk(self, payload: dict) -> dict:
        player_label = (payload.get("player_label") or "").strip()
        audio_b64 = payload.get("audio_b64") or ""
        chunk_index = int(payload.get("chunk_index", 0))
        timestamp = float(payload.get("timestamp", 0.0))
        client_id = str(payload.get("client_id", "unknown"))

        if not audio_b64:
            return {"status": "error", "message": "Chunk vazio"}
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as exc:
            return {"status": "error", "message": f"Erro de base64: {exc}"}

        if callable(self.satellite_chunk_callback):
            res = self.satellite_chunk_callback(player_label, client_id, chunk_index, timestamp, audio_bytes)
            return res if isinstance(res, dict) else {"status": "ok"}
        return {"status": "ok"}

    @staticmethod
    def get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def render_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>RPG Chronicler - Companion de Mesa</title>
    <style>
        :root {
            --bg: #0b0f17;
            --card: #151c28;
            --card-hover: #1b2434;
            --border: #233044;
            --text: #e1e7f0;
            --text-muted: #8899ac;
            --gold: #f59e0b;
            --gold-glow: #f59e0b33;
            --accent: #38bdf8;
            --accent-glow: #38bdf833;
            --green: #10b981;
            --green-glow: #10b98133;
            --red: #ef4444;
            --purple: #a855f7;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 14px;
            padding-bottom: 40px;
        }
        .header {
            text-align: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }
        .title { color: var(--gold); font-size: 1.35rem; font-weight: 800; margin: 0; letter-spacing: 0.5px; }
        .sub-header { display: flex; justify-content: center; gap: 8px; align-items: center; margin-top: 6px; }
        .status-badge {
            background: var(--green);
            color: #fff;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
        }
        .sess-duration { font-size: 0.8rem; color: var(--text-muted); }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .card h3 { margin: 0 0 10px 0; font-size: 1.05rem; color: var(--accent); display: flex; align-items: center; gap: 6px; }
        .form-group { margin-bottom: 8px; }
        label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 4px; font-weight: 600; }
        select {
            width: 100%;
            background: #0d131d;
            color: var(--text);
            border: 1px solid var(--border);
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
        }
        select:focus { border-color: var(--accent); }
        .btn-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 4px;
        }
        .btn {
            background: var(--card-hover);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 12px;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: bold;
            cursor: pointer;
            text-align: center;
            transition: all 0.15s ease;
        }
        .btn:active { transform: scale(0.97); }
        .btn-epic { border-color: var(--gold); color: var(--gold); background: var(--gold-glow); }
        .btn-crit { border-color: var(--red); color: var(--red); background: #ef444422; }
        .btn-primary { background: #0284c7; color: #fff; border: none; width: 100%; }
        .btn-danger { background: var(--red); color: #fff; border: none; width: 100%; }
        .btn-satellite-on { background: #059669; color: #fff; border: none; animation: pulse 2s infinite; }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .phrase-box {
            background: #0d131d;
            border-left: 3px solid var(--gold);
            padding: 10px 12px;
            font-size: 0.88rem;
            line-height: 1.4;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
            color: #d1d5db;
        }
        .vu-meter {
            height: 8px;
            width: 100%;
            background: #0d131d;
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }
        .vu-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--green) 60%, var(--gold) 85%, var(--red) 100%);
            transition: width 0.08s ease;
        }
        .alert-box {
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-top: 8px;
            display: none;
        }
        .alert-success { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
        .alert-warn { background: #78350f; color: #fde68a; border: 1px solid #d97706; }
        .dialogue-item { padding: 8px 0; border-bottom: 1px solid #1f293d; font-size: 0.9rem; line-height: 1.35; }
        .speaker { font-weight: bold; color: var(--gold); }
        .highlight-badge { background: #38bdf822; color: var(--accent); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
        .pill-tab {
            display: inline-block;
            background: #0d131d;
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            margin-right: 4px;
            color: var(--text-muted);
        }
        .pill-tab.active { background: var(--accent); color: #000; font-weight: bold; border-color: var(--accent); }
    </style>
</head>
<body>
    <div class="header">
        <div class="title" id="sess-title">⚔️ RPG Chronicler Companion</div>
        <div class="sub-header">
            <span class="status-badge" id="sess-status">Conectado</span>
            <span class="sess-duration" id="sess-duration">00:00:00</span>
        </div>
    </div>

    <!-- SELEÇÃO DE PERSONAGEM -->
    <div class="card">
        <h3>🎭 Meu Personagem na Mesa</h3>
        <div class="form-group">
            <label for="player-select">Selecione quem você está interpretando:</label>
            <select id="player-select" onchange="onPlayerChange()">
                <option value="">-- Carregando participantes... --</option>
            </select>
        </div>
        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;">
            Essa escolha orienta o treino de voz e o microfone satélite no seu aparelho.
        </div>
    </div>

    <!-- TREINO DE VOZ GUIADO -->
    <div class="card">
        <h3>🎙️ Treinar Voz com Leitura Rápida</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted); margin: 0 0 6px 0;">
            Grave uma das frases abaixo (15s) para o sistema aprender seu timbre e identificá-lo automaticamente:
        </p>

        <div style="margin-bottom: 6px;">
            <span class="pill-tab active" id="tab-p1" onclick="switchPhrase(1)">Opção 1: Ação & Furtividade</span>
            <span class="pill-tab" id="tab-p2" onclick="switchPhrase(2)">Opção 2: Combate & D20</span>
        </div>

        <div class="phrase-box" id="phrase-text">
            "Eu saco a minha arma e avanço com cautela pelas sombras do calabouço. Vejo uma porta de ferro enferrujada entreaberta e aviso o grupo em voz baixa: preparem suas tochas e fiquem atentos, sinto cheiro de enxofre e passos pesados logo à frente."
        </div>

        <div class="vu-meter"><div class="vu-fill" id="train-vu"></div></div>

        <div style="display: flex; gap: 8px; margin-top: 8px;">
            <button class="btn btn-primary" id="btn-train-rec" onclick="toggleTrainRecording()">🔴 Iniciar Gravação (15s)</button>
        </div>
        <div id="train-status" class="alert-box"></div>
    </div>

    <!-- MICROFONE SATÉLITE DA SESSÃO -->
    <div class="card">
        <h3>📡 Microfone Satélite da Sessão</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted); margin: 0 0 8px 0;">
            Deixe o celular na mesa perto de você durante o jogo. Ele capta sua voz com nitidez para ajudar a separar as falas da mesa.
        </p>
        <div class="vu-meter"><div class="vu-fill" id="sat-vu"></div></div>
        <button class="btn" id="btn-satellite" style="width: 100%;" onclick="toggleSatellite()">🎙️ Ativar Microfone Satélite</button>
        <div id="sat-status" class="alert-box" style="margin-top: 8px;"></div>
    </div>

    <!-- MARCADORES RÁPIDOS -->
    <div class="btn-grid">
        <button class="btn btn-epic" onclick="sendHighlight('epic', 'Momento Épico via Mobile')">⭐ Momento Épico</button>
        <button class="btn btn-crit" onclick="sendHighlight('critical', 'Crítico / Reviravolta')">⚔️ Crítico!</button>
    </div>

    <!-- ÚLTIMOS DIÁLOGOS -->
    <div class="card" style="margin-top: 14px;">
        <h3>📜 Transcrição Recente</h3>
        <div id="dialogues" style="max-height: 220px; overflow-y: auto;">Aguardando falas...</div>
    </div>

    <!-- DESTAQUES -->
    <div class="card">
        <h3>⭐ Destaques da Sessão</h3>
        <div id="highlights">Nenhum destaque ainda.</div>
    </div>

    <script>
        let currentPhrase = 1;
        const PHRASES = {
            1: "Eu saco a minha arma e avanço com cautela pelas sombras do calabouço. Vejo uma porta de ferro enferrujada entreaberta e aviso o grupo em voz baixa: preparem suas tochas e fiquem atentos, sinto cheiro de enxofre e passos pesados logo à frente.",
            2: "Mestre, tirei 18 no d20 de iniciativa! No meu turno eu conjuro minha magia no goblin da esquerda e rolo quatro dados de dano perfurante. Alguém do grupo precisa de cura ou posso gastar minha ação bônus para desengajar?"
        };

        function switchPhrase(num) {
            currentPhrase = num;
            document.getElementById('phrase-text').innerText = PHRASES[num];
            document.getElementById('tab-p1').classList.toggle('active', num === 1);
            document.getElementById('tab-p2').classList.toggle('active', num === 2);
        }

        function onPlayerChange() {
            const val = document.getElementById('player-select').value;
            if (val) localStorage.setItem('rpg_selected_player', val);
        }

        async function loadParticipants() {
            try {
                const res = await fetch('/api/participants');
                const data = await res.json();
                const sel = document.getElementById('player-select');
                sel.innerHTML = '';
                const parts = data.participants || [];
                if (parts.length === 0) {
                    sel.innerHTML = '<option value="">Nenhum participante configurado no PC</option>';
                    return;
                }
                const saved = localStorage.getItem('rpg_selected_player');
                parts.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p.label || p.nome || p;
                    opt.innerText = p.label || p.nome || p;
                    if (opt.value === saved) opt.selected = true;
                    sel.appendChild(opt);
                });
            } catch (e) {
                console.warn('Erro ao carregar participantes', e);
            }
        }

        // ==================== GRAVAÇÃO DE TREINO DE VOZ ====================
        let trainRecorder = null;
        let trainStream = null;
        let trainChunks = [];
        let trainTimer = null;
        let trainAudioContext = null;
        let trainAnalyser = null;
        let trainVuRaf = null;

        async function toggleTrainRecording() {
            const btn = document.getElementById('btn-train-rec');
            const statusBox = document.getElementById('train-status');
            const player = document.getElementById('player-select').value;

            if (!player) {
                alert('Selecione primeiro o seu personagem no topo!');
                return;
            }

            if (trainRecorder && trainRecorder.state === 'recording') {
                stopTrainRecording();
                return;
            }

            try {
                trainStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            } catch (err) {
                statusBox.style.display = 'block';
                statusBox.className = 'alert-box alert-warn';
                statusBox.innerText = '⚠️ O navegador bloqueou o microfone. No celular, o acesso ao microfone exige HTTPS ou permissão nas configurações do site.';
                return;
            }

            // Configura VU meter
            trainAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = trainAudioContext.createMediaStreamSource(trainStream);
            trainAnalyser = trainAudioContext.createAnalyser();
            trainAnalyser.fftSize = 256;
            source.connect(trainAnalyser);
            const dataArray = new Uint8Array(trainAnalyser.frequencyBinCount);
            function updateVu() {
                if (!trainAnalyser) return;
                trainAnalyser.getByteFrequencyData(dataArray);
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
                let avg = sum / dataArray.length;
                let pct = Math.min(100, Math.round((avg / 128) * 100));
                document.getElementById('train-vu').style.width = pct + '%';
                trainVuRaf = requestAnimationFrame(updateVu);
            }
            updateVu();

            trainChunks = [];
            trainRecorder = new MediaRecorder(trainStream);
            trainRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) trainChunks.push(e.data); };
            trainRecorder.onstop = uploadTrainAudio;
            trainRecorder.start(1000);

            let remaining = 15;
            btn.innerText = `⏹️ Gravando... (${remaining}s) Toque para Concluir`;
            btn.className = 'btn btn-danger';
            statusBox.style.display = 'block';
            statusBox.className = 'alert-box alert-success';
            statusBox.innerText = '🎙️ Leia a frase acima em tom natural...';

            trainTimer = setInterval(() => {
                remaining--;
                if (remaining <= 0) {
                    stopTrainRecording();
                } else {
                    btn.innerText = `⏹️ Gravando... (${remaining}s) Toque para Concluir`;
                }
            }, 1000);
        }

        function stopTrainRecording() {
            if (trainTimer) clearInterval(trainTimer);
            if (trainRecorder && trainRecorder.state === 'recording') trainRecorder.stop();
            if (trainStream) trainStream.getTracks().forEach(t => t.stop());
            if (trainVuRaf) cancelAnimationFrame(trainVuRaf);
            if (trainAudioContext) trainAudioContext.close();
            trainAnalyser = null;
            document.getElementById('train-vu').style.width = '0%';
            document.getElementById('btn-train-rec').innerText = '⏳ Processando e Enviando Treino...';
            document.getElementById('btn-train-rec').className = 'btn btn-primary';
        }

        async function uploadTrainAudio() {
            const statusBox = document.getElementById('train-status');
            const player = document.getElementById('player-select').value;
            const blob = new Blob(trainChunks, { type: 'audio/webm' });

            statusBox.style.display = 'block';
            statusBox.className = 'alert-box alert-success';
            statusBox.innerText = '📤 Enviando amostra vocal para o RPG Chronicler...';

            const reader = new FileReader();
            reader.onloadend = async () => {
                const b64 = reader.result;
                try {
                    const res = await fetch('/api/voice/train', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            player_label: player,
                            audio_b64: b64,
                            filename: 'mobile_train.webm'
                        })
                    });
                    const data = await res.json();
                    if (res.ok && data.status === 'ok') {
                        statusBox.className = 'alert-box alert-success';
                        statusBox.innerText = `✅ Sucesso! ${data.message || 'Perfil de voz reforçado.'}`;
                    } else {
                        statusBox.className = 'alert-box alert-warn';
                        statusBox.innerText = `⚠️ Falha: ${data.message || 'Erro ao treinar.'}`;
                    }
                } catch (err) {
                    statusBox.className = 'alert-box alert-warn';
                    statusBox.innerText = '⚠️ Erro ao comunicar com o servidor do PC: ' + err.message;
                } finally {
                    document.getElementById('btn-train-rec').innerText = '🔴 Iniciar Gravação (15s)';
                }
            };
            reader.readAsDataURL(blob);
        }

        // ==================== MICROFONE SATÉLITE DA SESSÃO ====================
        let satRecorder = null;
        let satStream = null;
        let satWakeLock = null;
        let satChunkIndex = 0;
        let satClientId = 'sat_' + Math.random().toString(36).substr(2, 9);
        let satAudioContext = null;
        let satAnalyser = null;
        let satVuRaf = null;

        async function toggleSatellite() {
            const btn = document.getElementById('btn-satellite');
            const statusBox = document.getElementById('sat-status');
            const player = document.getElementById('player-select').value;

            if (!player) {
                alert('Selecione seu personagem antes de ativar o microfone satélite!');
                return;
            }

            if (satRecorder && satRecorder.state === 'recording') {
                stopSatellite();
                btn.innerText = '🎙️ Ativar Microfone Satélite';
                btn.className = 'btn';
                statusBox.style.display = 'none';
                return;
            }

            try {
                satStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                if ('wakeLock' in navigator) {
                    try { satWakeLock = await navigator.wakeLock.request('screen'); } catch(e){}
                }
            } catch (err) {
                statusBox.style.display = 'block';
                statusBox.className = 'alert-box alert-warn';
                statusBox.innerText = '⚠️ Não foi possível acessar o microfone. Verifique permissões ou HTTPS.';
                return;
            }

            // VU meter satélite
            satAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = satAudioContext.createMediaStreamSource(satStream);
            satAnalyser = satAudioContext.createAnalyser();
            satAnalyser.fftSize = 256;
            source.connect(satAnalyser);
            const dataArray = new Uint8Array(satAnalyser.frequencyBinCount);
            function updateSatVu() {
                if (!satAnalyser) return;
                satAnalyser.getByteFrequencyData(dataArray);
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
                let avg = sum / dataArray.length;
                let pct = Math.min(100, Math.round((avg / 128) * 100));
                document.getElementById('sat-vu').style.width = pct + '%';
                satVuRaf = requestAnimationFrame(updateSatVu);
            }
            updateSatVu();

            satChunkIndex = 0;
            satRecorder = new MediaRecorder(satStream);
            satRecorder.ondataavailable = async e => {
                if (e.data && e.data.size > 0) {
                    sendSatelliteChunk(e.data, satChunkIndex++);
                }
            };
            satRecorder.start(6000); // chunks de 6 segundos

            btn.innerText = '🛑 Desativar Microfone Satélite';
            btn.className = 'btn btn-satellite-on';
            statusBox.style.display = 'block';
            statusBox.className = 'alert-box alert-success';
            statusBox.innerText = '🟢 Satélite Ativo! Transmitindo proximidade para a mesa...';
        }

        function stopSatellite() {
            if (satRecorder && satRecorder.state === 'recording') satRecorder.stop();
            if (satStream) satStream.getTracks().forEach(t => t.stop());
            if (satWakeLock) { satWakeLock.release(); satWakeLock = null; }
            if (satVuRaf) cancelAnimationFrame(satVuRaf);
            if (satAudioContext) satAudioContext.close();
            satAnalyser = null;
            document.getElementById('sat-vu').style.width = '0%';
        }

        async function sendSatelliteChunk(blob, index) {
            const player = document.getElementById('player-select').value;
            const reader = new FileReader();
            reader.onloadend = async () => {
                try {
                    await fetch('/api/satellite/chunk', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            player_label: player,
                            client_id: satClientId,
                            chunk_index: index,
                            timestamp: Date.now() / 1000,
                            audio_b64: reader.result
                        })
                    });
                } catch(e) {}
            };
            reader.readAsDataURL(blob);
        }

        // ==================== MARCADORES & ESTADO ====================
        async function fetchState() {
            try {
                const res = await fetch('/api/session');
                const d = await res.json();
                if (d.title) document.getElementById('sess-title').innerText = d.title;
                if (d.status) document.getElementById('sess-status').innerText = d.status;
                if (d.duration) document.getElementById('sess-duration').innerText = d.duration;
                if (d.recent_dialogues && d.recent_dialogues.length > 0) {
                    document.getElementById('dialogues').innerHTML = d.recent_dialogues.slice(-8).map(s =>
                        `<div class="dialogue-item"><span class="speaker">${s.speaker || 'Voz'}:</span> ${s.text}</div>`
                    ).join('');
                }
                if (d.highlights && d.highlights.length > 0) {
                    document.getElementById('highlights').innerHTML = d.highlights.slice(-5).map(h =>
                        `<div class="dialogue-item"><span class="highlight-badge">[${(h.category || 'DESTAQUE').toUpperCase()}]</span> ${h.description || 'Destaque'} (${h.formatted_time || ''})</div>`
                    ).join('');
                }
            } catch (e) {}
        }

        async function sendHighlight(cat, desc) {
            try {
                await fetch('/api/highlight', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ category: cat, description: desc })
                });
                alert('Marcador enviado para o Mestre!');
                fetchState();
            } catch(e) { alert('Erro ao enviar marcador.'); }
        }

        setInterval(fetchState, 3000);
        loadParticipants();
        fetchState();
    </script>
</body>
</html>
"""

