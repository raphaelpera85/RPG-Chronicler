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
