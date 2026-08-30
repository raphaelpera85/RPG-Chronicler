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
import shutil
import subprocess
import tempfile
from typing import Any, Iterable
from urllib.parse import urlparse

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
        self.mark_stage(
            "transcription",
            "completed",
            files=["transcript.json", "transcript.md", "transcript.srt", "transcript.vtt"],
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
) -> np.ndarray:
    """
    Pipeline mestre de áudio com suporte a Microfone de Conferência & Sala Grande:
    1. Passa-altas (Rumble cut < 80Hz)
    2. Supressão de Transientes (Rolagem de dados na mesa / impactos)
    3. Desreverberação de Sala (Eliminação de eco oco) [se conference_mode]
    4. Supressão espectral de ruído e chiado
    5. Realce de formantes de fala (Presença vocal)
    6. AGC e Normalização de ganho de longa distância (+18dB para participantes distantes)
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


