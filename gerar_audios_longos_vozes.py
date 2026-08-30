"""
Script para:
1. Treinar os perfis confirmados (Victor, Christian, Jorge).
2. Extrair áudios longos (10 a 20 segundos) das Vozes #02 e #08 para identificação final.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.io.wavfile as wavfile
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import rpg_chronicler as app
from rpg_chronicler import (
    load_voice_profiles,
    save_voice_profiles,
    extract_voice_training_embedding,
    extract_acoustic_features,
    format_timestamp,
)

BASE_DIR = Path(__file__).parent.resolve()
AUDIO_PATH = BASE_DIR / "gravacoes_sessoes" / "sessao_rpg_2026-08-26_20-27-15.wav"
TRANSCRIPT_PATH = BASE_DIR / "runs" / "20260828-213111-173667" / "transcript.json"
SAMPLES_15_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_separadas_15_vozes"
LONG_SAMPLES_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_longas_identificacao"
LONG_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def treinar(nome_perfil, arquivos):
    print(f"[*] Treinando {nome_perfil}...")
    features = []
    total_count = 0
    for f in arquivos:
        p = SAMPLES_15_DIR / f
        if p.is_file():
            info = extract_voice_training_embedding(p, window_seconds=2.0, min_window_seconds=0.5)
            feat = np.array(info["mean_feature"], dtype=np.float32)
            if app.is_valid_voice_embedding(feat):
                features.append(feat)
                total_count += info["count"]

    if not features:
        return

    mean_emb = np.mean(features, axis=0).astype(np.float32)
    profiles = load_voice_profiles()
    if nome_perfil in profiles and app.is_valid_voice_embedding(profiles[nome_perfil].get("embedding")):
        old = profiles[nome_perfil]
        old_emb = np.array(old["embedding"], dtype=np.float32)
        old_count = min(30, int(old.get("sample_count", 1)))
        new_count = min(25, total_count)
        fused = (old_emb * old_count + mean_emb * new_count) / max(1, old_count + new_count)
        profiles[nome_perfil]["embedding"] = fused.tolist()
        profiles[nome_perfil]["sample_count"] = min(50, old_count + new_count)
        profiles[nome_perfil]["last_updated"] = datetime.now().isoformat()
    else:
        profiles[nome_perfil] = {
            "embedding": mean_emb.tolist(),
            "sample_count": min(35, max(10, total_count)),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
    save_voice_profiles(profiles)
    print(f"[+] {nome_perfil} atualizado com sucesso!")

def main():
    print("=== ETAPA 1: Treinar Perfis Confirmados ===")
    # Victor -> Vozes 09, 10, 12
    treinar("[Artificer (Victor)]", [
        "amostra_voz_09_ex1.wav", "amostra_voz_09_ex2.wav", "amostra_voz_09_ex3.wav",
        "amostra_voz_10_ex1.wav", "amostra_voz_10_ex2.wav", "amostra_voz_10_ex3.wav",
        "amostra_voz_12_ex1.wav", "amostra_voz_12_ex2.wav", "amostra_voz_12_ex3.wav",
    ])

    # Christian -> Vozes 11, 14, 15
    treinar("[Mestre Christian - Narração de Cenários/NPCs]", [
        "amostra_voz_11_ex1.wav", "amostra_voz_11_ex2.wav", "amostra_voz_11_ex3.wav",
        "amostra_voz_14_ex1.wav", "amostra_voz_14_ex2.wav", "amostra_voz_14_ex3.wav",
        "amostra_voz_15_ex1.wav", "amostra_voz_15_ex2.wav", "amostra_voz_15_ex3.wav",
    ])

    # Jorge -> Voz 13
    treinar("[Druida (Jorge)]", [
        "amostra_voz_13_ex1.wav", "amostra_voz_13_ex2.wav", "amostra_voz_13_ex3.wav",
    ])

    print("\n=== ETAPA 2: Extrair Áudios Longos (10 a 20s) de Voz #02 e Voz #08 ===")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    segments = transcript_data.get("segments", [])

    sample_rate, audio_data = wavfile.read(AUDIO_PATH)
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Agrupamento das falas
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

    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = AgglomerativeClustering(n_clusters=15, metric="euclidean", linkage="ward")
    labels = clustering.fit_predict(X_scaled)

    # Identificar índices das Vozes #02 (label 1) e #08 (label 7)
    target_labels = {1: "Voz #02", 7: "Voz #08"}

    for target_label, tag_name in target_labels.items():
        matching_segs = [seg for seg, lbl in zip(sampled_segs, labels) if lbl == target_label]
        print(f"\n[{tag_name}] Total de segmentos no cluster: {len(matching_segs)}")

        # Encontra sequências ou trechos longos contínuos (10 a 20 segundos)
        # 1. Primeiro tenta trechos individuais longos
        long_candidates = sorted(matching_segs, key=lambda s: s["end"] - s["start"], reverse=True)

        extracted_count = 0
        for idx, seg in enumerate(long_candidates[:5]):
            start_sec = max(0.0, seg["start"] - 0.5)
            # Expande para capturar até 12-16 segundos de contexto contínuo
            end_sec = min(len(audio_data) / sample_rate, max(seg["end"] + 1.0, start_sec + 12.0))
            dur = end_sec - start_sec

            clip = audio_data[int(start_sec * sample_rate):int(end_sec * sample_rate)]
            max_v = np.max(np.abs(clip))
            if max_v > 0:
                clip_norm = (clip / max_v * 32767).astype(np.int16)
            else:
                clip_norm = clip.astype(np.int16)

            tag_clean = tag_name.lower().replace(" ", "_").replace("#", "")
            out_filename = f"amostra_longa_{tag_clean}_ex{idx + 1}_{int(dur)}s.wav"
            out_path = LONG_SAMPLES_DIR / out_filename
            wavfile.write(out_path, sample_rate, clip_norm)

            time_str = f"{format_timestamp(start_sec, decimal='.')} -> {format_timestamp(end_sec, decimal='.')}"
            print(f"  - Áudio Longo {idx + 1} ({dur:.1f}s): {out_filename} [{time_str}]")
            print(f"    Texto: \"{seg.get('text', '')}\"")
            extracted_count += 1
            if extracted_count >= 3:
                break

    print(f"\n[+] Áudios longos gerados em: {LONG_SAMPLES_DIR}")

if __name__ == "__main__":
    main()
