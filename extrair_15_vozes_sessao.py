"""
Script para extrair 15 clusters refinados de vozes da gravação da sessão de RPG.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import scipy.io.wavfile as wavfile
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

import rpg_chronicler as app
from rpg_chronicler import (
    extract_acoustic_features,
    load_voice_profiles,
    match_voice_clusters_to_profiles,
    format_timestamp,
)

BASE_DIR = Path(__file__).parent.resolve()
AUDIO_PATH = BASE_DIR / "gravacoes_sessoes" / "sessao_rpg_2026-08-26_20-27-15.wav"
TRANSCRIPT_PATH = BASE_DIR / "runs" / "20260828-213111-173667" / "transcript.json"
OUTPUT_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_separadas_15_vozes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"[*] Carregando transcrição de {TRANSCRIPT_PATH}...")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    segments = transcript_data.get("segments", [])
    print(f"[+] Segmentos carregados: {len(segments)}")

    print(f"[*] Lendo áudio da sessão: {AUDIO_PATH}...")
    sample_rate, audio_data = wavfile.read(AUDIO_PATH)
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Filtra segmentos de fala com boa duração e clareza
    valid_candidates = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        dur = end - start
        if 1.2 <= dur <= 14.0 and len(text) >= 12:
            valid_candidates.append((idx, seg))

    max_samples = min(2400, len(valid_candidates))
    step = len(valid_candidates) // max_samples if max_samples > 0 else 1
    sampled = valid_candidates[::max(1, step)][:max_samples]
    print(f"[*] Amostrando {len(sampled)} segmentos de áudio limpos...")

    features_list = []
    sampled_segs = []
    for pos, (idx, seg) in enumerate(sampled):
        start_samp = int(seg["start"] * sample_rate)
        end_samp = int(seg["end"] * sample_rate)
        chunk = audio_data[start_samp:end_samp]
        feat = extract_acoustic_features(chunk, sample_rate)
        if app.is_valid_voice_embedding(feat):
            features_list.append(feat)
            sampled_segs.append(seg)
        if (pos + 1) % 600 == 0 or pos + 1 == len(sampled):
            print(f"    Extração: {pos + 1}/{len(sampled)}")

    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = 15
    print(f"[*] Agrupando em {n_clusters} timbres/modos de vozes...")
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = clustering.fit_predict(X_scaled)

    cluster_features = {}
    cluster_segments = {}
    for seg, label, feat in zip(sampled_segs, labels, features_list):
        tag = f"Voz #{label + 1:02d}"
        cluster_features.setdefault(tag, []).append(feat)
        cluster_segments.setdefault(tag, []).append(seg)

    cluster_centroids = {}
    for tag, feats in cluster_features.items():
        mean_feat = np.mean(feats, axis=0)
        cluster_centroids[tag] = {
            "mean_feature": mean_feat.tolist(),
            "count": len(feats),
        }

    profiles = load_voice_profiles()
    predictions = match_voice_clusters_to_profiles(cluster_centroids, profiles, threshold=0.68)

    print("\n" + "=" * 65)
    print("🎙️ RESULTADOS DA EXTRAÇÃO DAS 15 VOZES")
    print("=" * 65)

    report_lines = [
        "# Relatório de 15 Vozes Extraídas da Sessão",
        f"**Arquivo original**: `sessao_rpg_2026-08-26_20-27-15.wav`",
        f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Lista de 15 Vozes e Amostras de Áudio para Treinamento",
        "",
    ]

    for label_idx in range(n_clusters):
        tag = f"Voz #{label_idx + 1:02d}"
        segs = cluster_segments.get(tag, [])
        pred = predictions.get(tag, {})
        pred_player = pred.get("player", "Não identificada")
        conf = pred.get("confidence", 0)

        print(f"\n--- {tag} ({len(segs)} falas agrupadas) | Sugestão: {pred_player} ({conf}%) ---")

        # Seleciona as 3 amostras mais nítidas e representativas
        sorted_segs = sorted(
            segs,
            key=lambda s: (s["end"] - s["start"] >= 3.0, len(s.get("text", ""))),
            reverse=True
        )
        top_samples = sorted_segs[:3]

        report_lines.append(f"### {tag} ({len(segs)} falas) -> Sugestão: **{pred_player}** ({conf}%)")

        for s_idx, sample_seg in enumerate(top_samples, start=1):
            s_start = sample_seg["start"]
            s_end = sample_seg["end"]
            s_text = sample_seg.get("text", "").strip()
            clip = audio_data[int(s_start * sample_rate):int(s_end * sample_rate)]

            max_v = np.max(np.abs(clip))
            if max_v > 0:
                clip_norm = (clip / max_v * 32767).astype(np.int16)
            else:
                clip_norm = clip.astype(np.int16)

            filename = f"amostra_voz_{label_idx + 1:02d}_ex{s_idx}.wav"
            file_path = OUTPUT_DIR / filename
            wavfile.write(file_path, sample_rate, clip_norm)

            time_str = f"{format_timestamp(s_start, decimal='.')} -> {format_timestamp(s_end, decimal='.')}"
            print(f"  [{s_idx}] {time_str} | Arquivo: {filename}")
            print(f"      \"{s_text}\"")

            report_lines.append(f"- **`{filename}`** `[{time_str}]`: *\"{s_text}\"*")

        report_lines.append("")

    report_file = OUTPUT_DIR / "relatorio_15_vozes.md"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[+] Relatório de 15 vozes salvo em: {report_file}")
    print(f"[+] Amostras WAV salvas em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
