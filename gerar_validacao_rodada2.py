"""
Valida a precisão após o ajuste fino e extrai uma nova rodada de teste com 2 áudios para cada jogador.
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import rpg_chronicler as app
from rpg_chronicler import (
    load_voice_profiles,
    extract_acoustic_features,
    format_timestamp,
)

BASE_DIR = Path(__file__).parent.resolve()
AUDIO_PATH = BASE_DIR / "gravacoes_sessoes" / "sessao_rpg_2026-08-26_20-27-15.wav"
TRANSCRIPT_PATH = BASE_DIR / "runs" / "20260828-213111-173667" / "transcript.json"
OUT_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_validacao_jogadores_rodada2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYERS = [
    "[Mestre Christian - Narração de Cenários/NPCs]",
    "[Lorenzo (Raphael)]",
    "[Ranger (João)]",
    "[Artificer (Victor)]",
    "[Bruxo (Wilson)]",
    "[Druida (Jorge)]",
    "[Pistoleiro (Fernando)]",
]

def main():
    profiles = load_voice_profiles()
    profile_names = [p for p in PLAYERS if p in profiles]
    profile_embeddings = np.array([profiles[p]["embedding"] for p in profile_names], dtype=np.float32)

    sample_rate, audio_data = wavfile.read(AUDIO_PATH)
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = np.mean(audio_data, axis=1)

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    segments = transcript_data.get("segments", [])

    candidates = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        dur = end - start
        if 4.0 <= dur <= 16.0 and len(text) >= 15:
            candidates.append((idx, seg, start, end, text, dur))

    step = max(1, len(candidates) // 1800)
    sampled = candidates[::step]

    results_by_player = {p: [] for p in profile_names}

    for idx, seg, start, end, text, dur in sampled:
        s_samp = int(start * sample_rate)
        e_samp = int(end * sample_rate)
        chunk = audio_data[s_samp:e_samp]
        feat = extract_acoustic_features(chunk, sample_rate)
        if not app.is_valid_voice_embedding(feat):
            continue

        sims = cosine_similarity(feat.reshape(1, -1), profile_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_player = profile_names[best_idx]
        best_score = float(sims[best_idx])

        results_by_player[best_player].append({
            "start": start,
            "end": end,
            "dur": dur,
            "text": text,
            "score": best_score,
            "chunk": chunk,
        })

    midpoint = (len(audio_data) / sample_rate) / 2.0

    print("=== RESULTADO DA NOVA RODADA DE VALIDAÇÃO (RODADA 2) ===")
    for p_name in profile_names:
        player_clean = p_name.replace("[", "").replace("]", "").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        clips = results_by_player[p_name]
        if not clips:
            print(f"[-] Sem clipes para {p_name}")
            continue

        first_half = [c for c in clips if c["start"] < midpoint]
        second_half = [c for c in clips if c["start"] >= midpoint]

        chosen_1 = max(first_half, key=lambda c: c["score"]) if first_half else clips[0]
        chosen_2 = max(second_half, key=lambda c: c["score"]) if second_half else clips[-1]

        print(f"\n👤 {p_name} ({len(clips)} falas identificadas na sessão):")
        for num, item in enumerate([chosen_1, chosen_2], 1):
            st, en = item["start"], item["end"]
            dur = en - st
            s_samp = max(0, int((st - 0.2) * sample_rate))
            e_samp = min(len(audio_data), int((en + 0.3) * sample_rate))
            clip_audio = audio_data[s_samp:e_samp]
            max_v = np.max(np.abs(clip_audio))
            if max_v > 0:
                clip_audio = (clip_audio / max_v * 32767).astype(np.int16)

            filename = f"rodada2_{player_clean}_pt{num}_{int(st//60)}m_{int(dur)}s.wav"
            out_p = OUT_DIR / filename
            wavfile.write(out_p, sample_rate, clip_audio)

            t1 = format_timestamp(st, decimal=".")
            t2 = format_timestamp(en, decimal=".")
            print(f"  [Áudio {num}] {filename} [{t1} -> {t2}]")
            print(f"    Texto: \"{item['text']}\"")

if __name__ == "__main__":
    main()
