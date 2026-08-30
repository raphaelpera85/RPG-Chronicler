"""
Script para:
1. Processar a sessão de RPG usando o modelo de reconhecimento com os 7 perfis calibrados.
2. Extrair 2 áudios de validação (8 a 15 segundos) de pontos bem diferentes da sessão para CADA UM dos 7 jogadores.
3. Gerar um relatório claro com texto transcrito, minutagem e link para o arquivo WAV para o usuário validar/corrigir.
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
OUT_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_validacao_jogadores"
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
    print("[*] Carregando perfis de voz...")
    profiles = load_voice_profiles()
    profile_names = [p for p in PLAYERS if p in profiles and app.is_valid_voice_embedding(profiles[p].get("embedding"))]
    profile_embeddings = np.array([profiles[p]["embedding"] for p in profile_names], dtype=np.float32)

    print(f"[*] Total de perfis ativos para classificação: {len(profile_names)}")

    print("[*] Carregando áudio da sessão...")
    sample_rate, audio_data = wavfile.read(AUDIO_PATH)
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = np.mean(audio_data, axis=1)

    print("[*] Carregando transcrição da sessão...")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    segments = transcript_data.get("segments", [])

    print(f"[*] Total de segmentos no arquivo: {len(segments)}")

    # Candidatos a falas de validação (duração entre 6 e 18 segundos, com texto razoável)
    candidates = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        dur = end - start
        if 4.0 <= dur <= 18.0 and len(text) >= 15:
            candidates.append((idx, seg, start, end, text, dur))

    print(f"[*] Candidatos com duração adequada: {len(candidates)}")

    # Extrai features de uma amostragem uniforme de 1500 falas
    step = max(1, len(candidates) // 1500)
    sampled = candidates[::step]

    results_by_player = {p: [] for p in profile_names}

    print("[*] Classificando falas candidatas contra o banco biométrico...")
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

        # Segundo melhor para medir margem de confiança
        sorted_indices = np.argsort(sims)[::-1]
        runner_up_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else best_idx
        margin = best_score - float(sims[runner_up_idx])

        results_by_player[best_player].append({
            "start": start,
            "end": end,
            "dur": dur,
            "text": text,
            "score": best_score,
            "margin": margin,
            "chunk": chunk,
        })

    # Selecionar 2 áudios de pontos BEM DISTANTES (um da 1ª metade e outro da 2ª metade da sessão)
    total_audio_duration = len(audio_data) / sample_rate
    midpoint = total_audio_duration / 2.0

    report_lines = [
        "# Relatório de Validação de Vozes por Jogador",
        f"**Áudio analisado**: `sessao_rpg_2026-08-26_20-27-15.wav` ({total_audio_duration/3600:.2f} horas)",
        f"**Amostras por jogador**: 2 áudios de pontos temporais diferentes da sessão\n",
    ]

    print("\n" + "=" * 70)
    print("=== SELEÇÃO DE 2 ÁUDIOS DE PONTOS DIFERENTES PARA CADA JOGADOR ===")
    print("=" * 70)

    for p_name in profile_names:
        player_clean = p_name.replace("[", "").replace("]", "").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        clips = results_by_player[p_name]
        if not clips:
            print(f"[-] Sem clipes encontrados para {p_name}")
            continue

        # Divide entre primeira metade e segunda metade da sessão para garantir momentos diferentes
        first_half = [c for c in clips if c["start"] < midpoint]
        second_half = [c for c in clips if c["start"] >= midpoint]

        # Prioriza maior score e margem
        if first_half:
            first_half.sort(key=lambda c: (c["score"], c["margin"]), reverse=True)
            chosen_1 = first_half[0]
        else:
            clips.sort(key=lambda c: c["start"])
            chosen_1 = clips[0]

        if second_half:
            second_half.sort(key=lambda c: (c["score"], c["margin"]), reverse=True)
            chosen_2 = second_half[0]
        else:
            clips.sort(key=lambda c: c["start"], reverse=True)
            chosen_2 = clips[0]

        selected_pair = [chosen_1, chosen_2]

        report_lines.append(f"\n## 👤 {p_name}")
        print(f"\n👤 {p_name}:")

        for num, item in enumerate(selected_pair, 1):
            st = item["start"]
            en = item["end"]
            dur = en - st

            # Extrai o clipe com leve padding
            s_samp = max(0, int((st - 0.3) * sample_rate))
            e_samp = min(len(audio_data), int((en + 0.5) * sample_rate))
            clip_audio = audio_data[s_samp:e_samp]
            max_v = np.max(np.abs(clip_audio))
            if max_v > 0:
                clip_audio = (clip_audio / max_v * 32767).astype(np.int16)
            else:
                clip_audio = clip_audio.astype(np.int16)

            filename = f"validacao_{player_clean}_pt{num}_{int(st//60)}m_{int(dur)}s.wav"
            filepath = OUT_DIR / filename
            wavfile.write(filepath, sample_rate, clip_audio)

            t_start = format_timestamp(st, decimal=".")
            t_end = format_timestamp(en, decimal=".")
            dur_real = (e_samp - s_samp) / sample_rate

            print(f"  [Áudio {num}] {filename} ({dur_real:.1f}s) [{t_start} -> {t_end}] (Confiança: {item['score']:.1%})")
            print(f"    Fala: \"{item['text']}\"")

            report_lines.append(
                f"- **Áudio {num} ({dur_real:.1f}s)**: [`{filename}`]({filepath.as_uri()}) `[{t_start} -> {t_end}]` (Confiança: **{item['score']:.1%}**)\n"
                f"  > *\"{item['text']}\"*\n"
            )

    report_path = OUT_DIR / "relatorio_validacao.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[+] Relatório de validação salvo em: {report_path}")
    print(f"[+] Áudios salvos em: {OUT_DIR}")

if __name__ == "__main__":
    main()
