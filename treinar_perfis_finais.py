"""
Treina perfis de vozes com as amostras longas identificadas e exibe diagnóstico de separabilidade.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

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
LONG_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_longas_identificacao"

def treinar(nome_perfil, arquivos, reset=False):
    print(f"[*] Treinando {nome_perfil} com {len(arquivos)} arquivos...")
    features = []
    total_count = 0
    for f in arquivos:
        p = LONG_DIR / f
        if p.is_file():
            info = extract_voice_training_embedding(p, window_seconds=2.0, min_window_seconds=0.5)
            feat = np.array(info["mean_feature"], dtype=np.float32)
            if app.is_valid_voice_embedding(feat):
                features.append(feat)
                total_count += info["count"]

    if not features:
        print(f"[-] Sem features para {nome_perfil}")
        return

    mean_emb = np.mean(features, axis=0).astype(np.float32)
    profiles = load_voice_profiles()

    if not reset and nome_perfil in profiles and app.is_valid_voice_embedding(profiles[nome_perfil].get("embedding")):
        old = profiles[nome_perfil]
        old_emb = np.array(old["embedding"], dtype=np.float32)
        old_count = min(30, int(old.get("sample_count", 1)))
        new_count = min(30, total_count)
        fused = (old_emb * old_count + mean_emb * new_count) / max(1, old_count + new_count)
        profiles[nome_perfil]["embedding"] = fused.tolist()
        profiles[nome_perfil]["sample_count"] = min(60, old_count + new_count)
        profiles[nome_perfil]["last_updated"] = datetime.now().isoformat()
    else:
        profiles[nome_perfil] = {
            "embedding": mean_emb.tolist(),
            "sample_count": min(40, max(15, total_count)),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
    save_voice_profiles(profiles)
    print(f"[+] {nome_perfil} gravado com sucesso!")

def main():
    # 1. Fernando (Pistoleiro) - Reset com a voz real autêntica
    treinar("[Pistoleiro (Fernando)]", [
        "amostra_longa_voz_08_ex1_13s.wav"
    ], reset=True)

    # 2. Wilson (Bruxo)
    treinar("[Bruxo (Wilson)]", [
        "amostra_longa_voz_08_ex2_12s.wav"
    ])

    # 3. João (Ranger)
    treinar("[Ranger (João)]", [
        "amostra_longa_voz_02_ex1_12s.wav",
        "amostra_longa_voz_02_ex2_12s.wav",
        "amostra_longa_voz_02_ex3_12s.wav",
        "amostra_longa_voz_08_ex3_12s.wav"
    ])

    print("\n" + "=" * 60)
    print("=== DIAGNÓSTICO DE SEPARABILIDADE ENTRE TODOS OS 7 PERFIS ===")
    print("=" * 60)
    diag = calculate_voice_profiles_separability()
    print(f"Perfis válidos no banco: {diag['valid_count']}")
    print(f"Similaridade média entre falantes: {diag.get('mean_similarity', 0):.2%}")
    print("\nMatriz de Similaridade Cruzada (quanto menor, mais fácil distinguir):")
    for p1, p2, sim in diag.get("top_similarities", []):
        alerta = "⚠️ ALTA" if sim > 0.88 else "✅ BOA SEPARAÇÃO"
        print(f"  - {p1:<28} <-> {p2:<28}: {sim:.1%} [{alerta}]")

if __name__ == "__main__":
    main()
