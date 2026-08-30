"""
Script para escutar, separar e extrair as amostras de áudio de todas as vozes
encontradas na sessão de RPG (sessao_rpg_2026-08-26_20-27-15.wav).
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
    save_voice_profiles,
    match_voice_clusters_to_profiles,
    format_timestamp,
)

BASE_DIR = Path(__file__).parent.resolve()
AUDIO_PATH = BASE_DIR / "gravacoes_sessoes" / "sessao_rpg_2026-08-26_20-27-15.wav"
TRANSCRIPT_PATH = BASE_DIR / "runs" / "20260828-213111-173667" / "transcript.json"
OUTPUT_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_separadas_sessao_2026-08-26"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"[*] Carregando transcrição de {TRANSCRIPT_PATH}...")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    segments = transcript_data.get("segments", [])
    print(f"[+] Total de segmentos na transcrição: {len(segments)}")

    print(f"[*] Lendo áudio da sessão: {AUDIO_PATH}...")
    sample_rate, audio_data = wavfile.read(AUDIO_PATH)
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = np.mean(audio_data, axis=1)
    duration_hrs = len(audio_data) / sample_rate / 3600
    print(f"[+] Áudio carregado: {len(audio_data)} amostras ({duration_hrs:.2f} horas), {sample_rate} Hz")

    # Amostragem representativa de segmentos com fala limpa
    valid_candidates = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        dur = end - start
        if 1.0 <= dur <= 15.0 and len(text) >= 10:
            valid_candidates.append((idx, seg))

    print(f"[+] Candidatos válidos de fala limpa: {len(valid_candidates)}")
    max_samples = min(2000, len(valid_candidates))
    step = len(valid_candidates) // max_samples if max_samples > 0 else 1
    sampled = valid_candidates[::max(1, step)][:max_samples]

    print(f"[*] Extraindo características acústicas de {len(sampled)} segmentos distribuídos...")
    features_list = []
    sampled_segs = []
    for position, (idx, seg) in enumerate(sampled):
        start_samp = int(seg["start"] * sample_rate)
        end_samp = int(seg["end"] * sample_rate)
        chunk = audio_data[start_samp:end_samp]
        feat = extract_acoustic_features(chunk, sample_rate)
        if app.is_valid_voice_embedding(feat):
            features_list.append(feat)
            sampled_segs.append(seg)
        if (position + 1) % 500 == 0 or position + 1 == len(sampled):
            print(f"    Progresso: {position + 1}/{len(sampled)}")

    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = 7
    print(f"[*] Agrupando em {n_clusters} timbres de vozes físicas com Agglomerative Clustering...")
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = clustering.fit_predict(X_scaled)

    # Coleta clusters e seus segmentos
    cluster_features = {}
    cluster_segments = {}
    for seg, label, feat in zip(sampled_segs, labels, features_list):
        tag = f"Voz Física #{label + 1}"
        cluster_features.setdefault(tag, []).append(feat)
        cluster_segments.setdefault(tag, []).append(seg)

    cluster_centroids = {}
    for tag, feats in cluster_features.items():
        mean_feat = np.mean(feats, axis=0)
        cluster_centroids[tag] = {
            "mean_feature": mean_feat.tolist(),
            "count": len(feats),
        }

    # Carrega perfis cadastrados para predizer participantes
    profiles = load_voice_profiles()
    predictions = match_voice_clusters_to_profiles(cluster_centroids, profiles, threshold=0.65)

    print("\n" + "=" * 60)
    print("🎙️ RESULTADOS DA SEPARAÇÃO ACÚSTICA DAS 7 VOZES")
    print("=" * 60)

    report_lines = [
        "# Relatório de Separação de Vozes da Sessão",
        f"**Arquivo**: `sessao_rpg_2026-08-26_20-27-15.wav`",
        f"**Duração**: {duration_hrs:.2f} horas ({len(segments)} falas)",
        f"**Data do processamento**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Vozes Identificadas e Amostras de Áudio Extraídas",
        "",
    ]

    for label_idx in range(n_clusters):
        tag = f"Voz Física #{label_idx + 1}"
        segs_in_cluster = cluster_segments.get(tag, [])
        pred = predictions.get(tag, {})
        pred_player = pred.get("player", "Desconhecido / Nova Voz")
        conf = pred.get("confidence", 0)
        is_conf = pred.get("is_confident", False)

        print(f"\n--- {tag} ({len(segs_in_cluster)} segmentos amostrados) ---")
        print(f"    Associação provável: {pred_player} (Confiança: {conf}%, Confiante: {is_conf})")

        # Seleciona as 3 melhores amostras (áudios de 3 a 7 segundos com texto claro)
        segs_sorted = sorted(segs_in_cluster, key=lambda s: (s["end"] - s["start"] >= 3.0, len(s.get("text", ""))), reverse=True)
        top_samples = segs_sorted[:3]

        report_lines.append(f"### {tag} -> **{pred_player}** (Confiança: {conf}%)")
        report_lines.append(f"- **Total de falas agrupadas**: {len(segs_in_cluster)}")
        report_lines.append("- **Amostras de áudio salvas para escutar:**")

        for s_idx, sample_seg in enumerate(top_samples, start=1):
            s_start = sample_seg["start"]
            s_end = sample_seg["end"]
            s_text = sample_seg.get("text", "").strip()
            clip = audio_data[int(s_start * sample_rate):int(s_end * sample_rate)]
            
            # Normalização de áudio para salvar WAV
            max_v = np.max(np.abs(clip))
            if max_v > 0:
                clip_norm = (clip / max_v * 32767).astype(np.int16)
            else:
                clip_norm = clip.astype(np.int16)

            filename = f"amostra_voz_{label_idx + 1}_ex{s_idx}.wav"
            file_path = OUTPUT_DIR / filename
            wavfile.write(file_path, sample_rate, clip_norm)

            time_str = f"{format_timestamp(s_start, decimal='.')} -> {format_timestamp(s_end, decimal='.')}"
            print(f"    [Exemplo {s_idx}] {time_str} | Arquivo: {filename}")
            print(f"      Fala: \"{s_text}\"")

            report_lines.append(f"  {s_idx}. `[{time_str}]` **`{filename}`**: *\"{s_text}\"*")

        report_lines.append("")

    report_path = BASE_DIR / "gravacoes_sessoes" / "amostras_separadas_sessao_2026-08-26" / "relatorio_vozes.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[+] Relatório detalhado salvo em: {report_path}")
    print(f"[+] Todos os arquivos WAV de áudio separados estão em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
