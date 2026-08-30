"""
Script para reajustar e calibrar com precisão cirúrgica os 7 perfis de voz
baseado nas correções exatas fornecidas pelo usuário.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.io.wavfile as wavfile

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
)

BASE_DIR = Path(__file__).parent.resolve()
VAL_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_validacao_jogadores"
LONG_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_longas_identificacao"
SAMPLES_15_DIR = BASE_DIR / "gravacoes_sessoes" / "amostras_separadas_15_vozes"

# Mapeamento definitivo de arquivos de áudio válidos e puros por jogador:
MAP_ARQUIVOS = {
    "[Mestre Christian - Narração de Cenários/NPCs]": [
        # Validados anteriormente
        (SAMPLES_15_DIR / "amostra_voz_03_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_03_ex2.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_06_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_07_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_11_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_14_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_15_ex3.wav", None),
        # Corrigidos nesta rodada
        (VAL_DIR / "validacao_Mestre_Christian_-_Narração_de_Cenários_NPCs_pt1_57m_4s.wav", None),
        (VAL_DIR / "validacao_Mestre_Christian_-_Narração_de_Cenários_NPCs_pt2_166m_9s.wav", None),
        (VAL_DIR / "validacao_Artificer_Victor_pt1_71m_5s.wav", None),
        (VAL_DIR / "validacao_Bruxo_Wilson_pt1_71m_8s.wav", None),
        (VAL_DIR / "validacao_Bruxo_Wilson_pt2_199m_7s.wav", None),
        (VAL_DIR / "validacao_Druida_Jorge_pt1_69m_5s.wav", None),
    ],

    "[Lorenzo (Raphael)]": [
        (SAMPLES_15_DIR / "amostra_voz_05_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_05_ex2.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_05_ex3.wav", None),
        (VAL_DIR / "validacao_Lorenzo_Raphael_pt2_134m_4s.wav", None),
        # Corta apenas o final "Dessna de Navia" do pt1 (últimos 1.8 segundos)
        (VAL_DIR / "validacao_Lorenzo_Raphael_pt1_59m_4s.wav", "tail_1.8s"),
    ],

    "[Ranger (João)]": [
        (SAMPLES_15_DIR / "amostra_voz_01_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_01_ex2.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_04_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_04_ex2.wav", None),
        (LONG_DIR / "amostra_longa_voz_02_ex1_12s.wav", None),
        (LONG_DIR / "amostra_longa_voz_02_ex2_12s.wav", None),
        (LONG_DIR / "amostra_longa_voz_02_ex3_12s.wav", None),
        (LONG_DIR / "amostra_longa_voz_08_ex3_12s.wav", None),
        (VAL_DIR / "validacao_Ranger_João_pt1_117m_7s.wav", None),
    ],

    "[Artificer (Victor)]": [
        (SAMPLES_15_DIR / "amostra_voz_09_ex2.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_10_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_10_ex2.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_12_ex1.wav", None),
        (VAL_DIR / "validacao_Artificer_Victor_pt2_166m_9s.wav", None),
        # Corrigidos nesta rodada como Victor
        (VAL_DIR / "validacao_Druida_Jorge_pt2_163m_10s.wav", None),
        (VAL_DIR / "validacao_Pistoleiro_Fernando_pt2_157m_4s.wav", None),
    ],

    "[Bruxo (Wilson)]": [
        (LONG_DIR / "amostra_longa_voz_08_ex2_12s.wav", None),
        # Corrigido nesta rodada como Wilson
        (VAL_DIR / "validacao_Ranger_João_pt2_225m_8s.wav", None),
    ],

    "[Druida (Jorge)]": [
        (SAMPLES_15_DIR / "amostra_voz_13_ex1.wav", None),
        (SAMPLES_15_DIR / "amostra_voz_13_ex2.wav", None),
    ],

    "[Pistoleiro (Fernando)]": [
        (LONG_DIR / "amostra_longa_voz_08_ex1_13s.wav", None),
        (VAL_DIR / "validacao_Pistoleiro_Fernando_pt1_36m_5s.wav", None),
    ],
}

def recalibrar():
    profiles = {}
    print("=== RECALIBRAÇÃO DE PERFIS COM FEEDBACK HUMANO ===")

    for player, lista_arquivos in MAP_ARQUIVOS.items():
        features = []
        total_seconds = 0.0

        for path_obj, slice_rule in lista_arquivos:
            if not path_obj.is_file():
                print(f"[-] Arquivo não encontrado: {path_obj.name}")
                continue

            sr, data = wavfile.read(path_obj)
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)

            if slice_rule == "tail_1.8s":
                # Pega apenas os últimos 1.8 segundos
                cut_samps = int(1.8 * sr)
                data = data[-cut_samps:]

            dur = len(data) / sr
            total_seconds += dur

            # Extração em janelas de 1.5s
            win_size = int(1.5 * sr)
            hop = int(0.75 * sr)
            if len(data) < win_size:
                feat = extract_acoustic_features(data, sr)
                if app.is_valid_voice_embedding(feat):
                    features.append(feat)
            else:
                for st in range(0, len(data) - win_size + 1, hop):
                    chunk = data[st:st + win_size]
                    feat = extract_acoustic_features(chunk, sr)
                    if app.is_valid_voice_embedding(feat):
                        features.append(feat)

        if not features:
            print(f"[!] AVISO: Sem features extraídas para {player}")
            continue

        mean_emb = np.mean(features, axis=0).astype(np.float32)
        profiles[player] = {
            "embedding": mean_emb.tolist(),
            "sample_count": len(features),
            "total_audio_seconds": round(total_seconds, 2),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        print(f"[+] {player:<46}: {len(features):>3} janelas de áudio ({total_seconds:5.1f}s acumulados)")

    save_voice_profiles(profiles)
    print("\n[+] Todos os perfis em 'perfis_vozes.json' foram salvos e 100% recalibrados!")

if __name__ == "__main__":
    recalibrar()
