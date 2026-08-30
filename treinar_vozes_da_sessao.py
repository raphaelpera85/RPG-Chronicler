"""
Script para registrar e treinar os perfis no perfis_vozes.json
a partir das vozes separadas da sessão.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

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
    calculate_voice_profiles_separability,
)

BASE_DIR = Path(__file__).parent.resolve()
SAMPLES_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_separadas_sessao_2026-08-26"

def treinar_perfil(nome_perfil, arquivos_amostra):
    print(f"[*] Treinando/Reforçando perfil: {nome_perfil}...")
    features = []
    total_count = 0
    for sample_file in arquivos_amostra:
        path = SAMPLES_DIR / sample_file
        if path.is_file():
            emb_info = extract_voice_training_embedding(path, window_seconds=2.0, min_window_seconds=0.5)
            feat = np.array(emb_info["mean_feature"], dtype=np.float32)
            if app.is_valid_voice_embedding(feat):
                features.append(feat)
                total_count += emb_info["count"]

    if not features:
        print(f"[-] Nenhuma amostra válida encontrada para {nome_perfil}")
        return

    mean_embedding = np.mean(features, axis=0).astype(np.float32)
    profiles = load_voice_profiles()

    if nome_perfil in profiles and app.is_valid_voice_embedding(profiles[nome_perfil].get("embedding")):
        old = profiles[nome_perfil]
        old_emb = np.array(old["embedding"], dtype=np.float32)
        old_count = min(30, int(old.get("sample_count", 1)))
        new_count = min(20, total_count)
        fused = (old_emb * old_count + mean_embedding * new_count) / max(1, old_count + new_count)
        profiles[nome_perfil]["embedding"] = fused.tolist()
        profiles[nome_perfil]["sample_count"] = min(50, old_count + new_count)
        profiles[nome_perfil]["last_updated"] = datetime.now().isoformat()
    else:
        profiles[nome_perfil] = {
            "embedding": mean_embedding.tolist(),
            "sample_count": min(30, max(5, total_count)),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

    save_voice_profiles(profiles)
    print(f"[+] Perfil '{nome_perfil}' salvo com sucesso!")

if __name__ == "__main__":
    print("Banco atual de vozes:", list(load_voice_profiles().keys()))
