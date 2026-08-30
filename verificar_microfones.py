import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import sounddevice as sd
import numpy as np
import time

devices = sd.query_devices()
hostapis = sd.query_hostapis()
default_in = sd.default.device[0]

print("=" * 70)
print("🔍 DIAGNÓSTICO DE DISPOSITIVOS DE ÁUDIO (ENTRADA / MICROFONES)")
print("=" * 70)

input_devices = []
for idx, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        api_name = hostapis[dev["hostapi"]]["name"]
        is_default = (idx == default_in)
        input_devices.append((idx, dev, api_name, is_default))
        flag = " ★ [PADRÃO DO WINDOWS]" if is_default else ""
        print(f"[{idx:2d}] {dev['name']}")
        print(f"     Driver/API: {api_name} | Canais: {dev['max_input_channels']} | Taxa Nativa: {int(dev['default_samplerate'])} Hz{flag}")

print("\n" + "=" * 70)
print("🎧 TESTE DE CAPTURA RÁPIDA (2 SEGUNDOS) NO DISPOSITIVO PADRÃO")
print("=" * 70)

target_idx = default_in if default_in is not None and default_in >= 0 else (input_devices[0][0] if input_devices else None)

if target_idx is not None:
    target_dev = devices[target_idx]
    sr = int(target_dev["default_samplerate"])
    channels = min(2, target_dev["max_input_channels"])
    print(f"Testando captura em: [{target_idx}] {target_dev['name']} ({sr} Hz, {channels} canais)...")
    
    try:
        recording = sd.rec(int(2.0 * sr), samplerate=sr, channels=channels, device=target_idx, dtype='float32')
        sd.wait()
        
        # Análise de Qualidade do Sinal
        mono_rec = np.mean(recording, axis=1) if channels > 1 else recording.flatten()
        rms = float(np.sqrt(np.mean(mono_rec ** 2)))
        peak = float(np.max(np.abs(mono_rec)))
        rms_db = 20.0 * np.log10(max(1e-6, rms))
        peak_db = 20.0 * np.log10(max(1e-6, peak))
        
        # Teste de ruído de fundo / SNR estimado
        print("\n📊 RESULTADO DA ANÁLISE DE QUALIDADE:")
        print(f"• Dispositivo Ativo: {target_dev['name']}")
        print(f"• Taxa de Amostragem: {sr} Hz ({'Ideal para conferência/alta fidelidade (>=44.1kHz)' if sr >= 44100 else 'Padrão voz'})")
        print(f"• Canais de Entrada: {channels}")
        print(f"• Nível RMS (Energia Média): {rms_db:.1f} dBFS")
        print(f"• Pico Máximo Registrado: {peak_db:.1f} dBFS")
        
        if peak < 0.001:
            print("⚠️ Status: Sinal extremamente baixo ou microfone mutado/sem permissão.")
        elif rms_db < -55.0:
            print("✅ Status: Piso de ruído excelente (muito silencioso, sem chiado elétrico grave).")
        elif rms_db < -35.0:
            print("✅ Status: Nível sonoro ambiente detectado dentro da faixa normal de sala.")
        else:
            print("⚠️ Status: Volume de entrada muito alto ou ruído ambiente elevado.")
            
    except Exception as exc:
        print(f"❌ Erro ao gravar teste no microfone: {exc}")
else:
    print("❌ Nenhum microfone de entrada foi detectado pelo Windows.")
