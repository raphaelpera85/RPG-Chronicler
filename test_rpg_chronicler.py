"""Isolated standard-library test suite for RPG Chronicler."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np
import scipy.io.wavfile as wavfile

import rpg_chronicler as app_module
from rpg_chronicler import (
    AudioRecorder,
    build_voice_ai_context,
    calculate_voice_profiles_separability,
    clean_voice_profiles,
    chunk_text_by_token_budget,
    delete_voice_profile,
    estimate_text_tokens,
    extract_acoustic_features,
    extract_cluster_audio_samples,
    fit_text_to_token_budget,
    identify_voice_sample,
    is_valid_voice_embedding,
    match_voice_clusters_to_profiles,
    perform_acoustic_diarization,
    train_voice_profile_from_audio,
)
from rpg_chronicler_core import (
    SessionRun,
    atomic_write_json,
    format_timestamp,
    normalize_openai_base_url,
    prepare_audio,
    probe_audio,
    segments_to_srt,
    segments_to_vtt,
    cloud_destination_fingerprint,
    describe_model_access,
    format_model_catalog_choice,
    model_catalog_sort_key,
    apply_highpass_filter,
    apply_spectral_noise_suppression,
    apply_speech_vocal_enhancer,
    apply_dynamic_gain_control,
    apply_transient_suppression,
    apply_dereverberation,
    enhance_audio_pipeline,
    clean_and_enhance_audio_file,
)


class RPGChroniclerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rpg-chronicler-test-")
        self.temp_path = Path(self.temporary.name)
        sample_rate = 16000
        time_axis = np.arange(sample_rate) / sample_rate
        audio = (0.25 * np.sin(2 * np.pi * 220 * time_axis) * 32767).astype(np.int16)
        self.mono_wav = self.temp_path / "voice.wav"
        wavfile.write(self.mono_wav, sample_rate, audio)

    def tearDown(self):
        self.temporary.cleanup()

    def test_feature_vector_is_finite_and_bounded_in_size(self):
        sample_rate, audio = wavfile.read(self.mono_wav)
        features = extract_acoustic_features(audio, sample_rate)
        self.assertEqual(features.shape, (38,))
        self.assertTrue(np.isfinite(features).all())

    def test_single_segment_diarization(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Uma fala curta"}]
        tags, centroids = perform_acoustic_diarization(
            self.mono_wav, segments, estimated_speakers=7, return_centroids=True
        )
        self.assertEqual(tags, ["Voz Física #1"])
        self.assertEqual(centroids["Voz Física #1"]["count"], 1)

    def test_used_player_penalty_is_compared_consistently(self):
        similarity = lambda _feat, profile: {1: 1.0, 2: 0.95}[int(profile[0])]
        clusters = {
            "Voz Física #1": {"mean_feature": [0.0], "count": 20},
            "Voz Física #2": {"mean_feature": [0.0], "count": 10},
        }
        profiles = {
            "Jogador A": {"embedding": [1.0]},
            "Jogador B": {"embedding": [2.0]},
        }
        with mock.patch.object(app_module, "normalized_voice_similarity", similarity):
            result = match_voice_clusters_to_profiles(clusters, profiles)
        self.assertEqual(result["Voz Física #1"]["player"], "Jogador A")
        self.assertEqual(result["Voz Física #2"]["player"], "Jogador B")

    def test_snippets_are_isolated(self):
        output = self.temp_path / "snippets"
        samples = extract_cluster_audio_samples(
            self.mono_wav,
            [{"start": 0.0, "end": 1.0, "text": "fala representativa"}],
            ["Voz Física #1"],
            output_dir=output,
        )
        self.assertEqual(len(samples), 1)
        sample_path = Path(samples[0]["sample_path"])
        self.assertTrue(sample_path.is_file())
        self.assertTrue(sample_path.is_relative_to(output))

    def test_manual_voice_training_updates_profile_bank(self):
        profiles_file = self.temp_path / "profiles.json"
        with mock.patch.object(app_module, "VOICE_PROFILES_FILE", profiles_file):
            profile = train_voice_profile_from_audio("[Heroi (Jogador)]", self.mono_wav)
            saved = json.loads(profiles_file.read_text(encoding="utf-8"))

        self.assertIn("[Heroi (Jogador)]", saved)
        self.assertEqual(len(profile["embedding"]), 38)
        self.assertGreater(profile["sample_count"], 0)

    def test_voice_identification_ranks_trained_profile_first(self):
        profiles_file = self.temp_path / "profiles.json"
        with mock.patch.object(app_module, "VOICE_PROFILES_FILE", profiles_file):
            train_voice_profile_from_audio("[Heroi (Jogador)]", self.mono_wav)
            matches = identify_voice_sample(self.mono_wav)

        self.assertEqual(matches[0]["label"], "[Heroi (Jogador)]")
        self.assertGreaterEqual(matches[0]["confidence"], 99)

    def test_voice_ai_context_includes_predictions_confirmations_and_examples(self):
        context = build_voice_ai_context(
            [
                {"start": 1.0, "end": 2.0, "text": "Eu abro a porta com cuidado."},
                {"start": 3.0, "end": 4.0, "text": "A sala está escura e fria."},
            ],
            ["Voz Física #1", "Voz Física #2"],
            predictions={
                "Voz Física #1": {"player": "[Heroi (Jogador)]", "confidence": 88, "is_confident": True},
                "Voz Física #2": {"player": "[Mestre Mesa - Narração de Cenários/NPCs]", "confidence": 73, "is_confident": False},
            },
            user_mapping={"Voz Física #1": "[Heroi (Jogador)]"},
            samples_info=[{"tag": "Voz Física #2", "sample_text": "A sala está escura e fria."}],
        )

        self.assertIn("Confirmado pelo usuário: [Heroi (Jogador)]", context)
        self.assertIn("Predição acústica do banco: [Mestre Mesa - Narração de Cenários/NPCs] (73%", context)
        self.assertIn("Amostra representativa: A sala está escura e fria.", context)
        self.assertIn("Eu abro a porta com cuidado.", context)

    def test_text_chunking_respects_token_budget(self):
        text = "\n".join(f"[00:{idx:02d}] fala longa de teste para o modelo local" for idx in range(80))
        chunks = chunk_text_by_token_budget(text, 256)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("\n".join(chunks), text)
        self.assertTrue(all(estimate_text_tokens(chunk) <= 256 for chunk in chunks))

    def test_fit_text_to_token_budget_preserves_head_and_tail(self):
        text = "A" * 3000 + "\nMEIO\n" + "Z" * 3000
        fitted = fit_text_to_token_budget(text, 600)

        self.assertLessEqual(estimate_text_tokens(fitted), 650)
        self.assertTrue(fitted.startswith("A"))
        self.assertTrue(fitted.endswith("Z"))
        self.assertIn("omitido", fitted)

    def test_audio_recorder_flushes_writer_before_return(self):
        class FakeStream:
            def __init__(self, **_kwargs):
                pass
            def start(self):
                pass
            def stop(self):
                pass
            def close(self):
                pass

        with mock.patch.object(app_module.sd, "InputStream", FakeStream), mock.patch.object(
            app_module, "AUDIO_DIR", self.temp_path
        ):
            recorder = AudioRecorder(sample_rate=16000, channels=1)
            recorder.start()
            for _ in range(20):
                recorder.callback(np.full((800, 1), 0.25, dtype=np.float32), 800, None, None)
            destination = recorder.stop()
        self.assertIsNotNone(destination)
        with wave.open(str(destination), "rb") as handle:
            self.assertEqual(handle.getnframes(), 20 * 800)

    def test_gemini_route_is_not_corrupted(self):
        route = "https://generativelanguage.googleapis.com/v1beta/openai"
        self.assertEqual(normalize_openai_base_url(route), route)
        self.assertEqual(normalize_openai_base_url("localhost:1234"), "http://localhost:1234/v1")

    def test_cloud_consent_identity_includes_destination_and_model(self):
        a = cloud_destination_fingerprint("https://example.test/v1", "model-a")
        self.assertNotEqual(a, cloud_destination_fingerprint("https://other.test/v1", "model-a"))
        self.assertNotEqual(a, cloud_destination_fingerprint("https://example.test/v1", "model-b"))

    def test_hermes_model_access_distinguishes_maker_provider_and_credits(self):
        paid = describe_model_access(
            "http://127.0.0.1:8645/v1", "openai/gpt-5.4",
            {"pricing": {"prompt": "0.000002", "completion": "0.000012"}},
        )
        self.assertEqual(paid["maker"], "OpenAI")
        self.assertEqual(paid["provider"], "Hermes / Nous Portal")
        self.assertEqual(paid["billing"], "Créditos / pagamento por uso")

    def test_free_and_local_models_have_explicit_billing_labels(self):
        free = describe_model_access(
            "http://127.0.0.1:8645/v1", "stepfun/model:free",
            {"pricing": {"prompt": "0", "completion": "0"}},
        )
        local = describe_model_access("http://localhost:1234/v1", "qwen-local")
        self.assertEqual(free["billing"], "Grátis (pode ter limites)")
        self.assertEqual(local["provider"], "LM Studio local")

    def test_hermes_launcher_defaults_to_an_explicitly_free_model(self):
        launcher = (Path(app_module.__file__).parent / "hermes_session.ps1").read_text(encoding="utf-8")
        self.assertIn(":free'", launcher)
        self.assertNotIn("RPG_CHRONICLER_LM_MODEL = 'openai/", launcher)

    def test_catalog_is_grouped_by_maker_and_labels_access_mode(self):
        base_url = "http://127.0.0.1:8645/v1"
        entries = [
            {"id": "stepfun/free:free", "pricing": {"prompt": "0"}},
            {"id": "openai/paid", "pricing": {"prompt": "0.01"}},
            {"id": "openai/subscription", "access_mode": "subscription"},
        ]
        ordered = sorted(entries, key=lambda item: model_catalog_sort_key(base_url, item))
        labels = [format_model_catalog_choice(base_url, item) for item in ordered]
        self.assertEqual(
            [item["id"] for item in ordered],
            ["openai/subscription", "openai/paid", "stepfun/free:free"],
        )
        self.assertIn("OpenAI | ASSINATURA |", labels[0])
        self.assertIn("OpenAI | CRÉDITOS NOUS |", labels[1])
        self.assertIn("StepFun | GRÁTIS |", labels[2])

    def test_diarization_honors_cancellation(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 0.0, "end": 1.0, "text": "b"}]
        event = __import__("threading").Event()
        event.set()
        with self.assertRaises(InterruptedError):
            perform_acoustic_diarization(self.mono_wav, segments, cancel_event=event)

    def test_trackable_source_has_no_campaign_identifiers(self):
        source = Path(app_module.__file__).read_text(encoding="utf-8")
        for value in ("PrivateCharacterName", "PrivateCityName", "PrivateFactionName", "PrivatePlayerName"):
            self.assertNotIn(value, source)

    def test_caption_exports_support_more_than_one_hour(self):
        segments = [{"start": 3661.25, "end": 3663.5, "text": "Teste", "speaker": "Mestre"}]
        self.assertEqual(format_timestamp(3661.25), "01:01:01,250")
        self.assertIn("01:01:01,250 --> 01:01:03,500", segments_to_srt(segments))
        self.assertTrue(segments_to_vtt(segments).startswith("WEBVTT\n"))

    def test_atomic_json_replaces_complete_document(self):
        destination = self.temp_path / "state.json"
        atomic_write_json(destination, {"version": 1})
        atomic_write_json(destination, {"version": 2, "ok": True})
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"version": 2, "ok": True})
        self.assertFalse(list(self.temp_path.glob("*.tmp")))

    def test_session_run_writes_all_transcript_formats(self):
        metadata = probe_audio(self.mono_wav)
        run = SessionRun.create(self.temp_path / "runs", metadata, {"api": "local"})
        run.write_transcript([{"start": 0.0, "end": 1.0, "text": "Olá", "speaker": "Jogador"}])
        run.finish()
        self.assertEqual(json.loads(run.manifest_path.read_text(encoding="utf-8"))["status"], "completed")
        for name in ("transcript.json", "transcript.md", "transcript.srt", "transcript.vtt"):
            self.assertTrue((run.directory / name).is_file())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg não instalado")
    def test_compressed_formats_are_converted_to_supported_pcm_wav(self):
        for extension in ("mp3", "m4a", "ogg", "flac", "aac"):
            with self.subTest(extension=extension):
                compressed = self.temp_path / f"voice.{extension}"
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(self.mono_wav), str(compressed)],
                    check=True,
                )
                prepared = prepare_audio(compressed)
                temporary = prepared.temporary_directory
                try:
                    self.assertEqual(prepared.processing_path.suffix, ".wav")
                    self.assertEqual(prepared.metadata.codec, "pcm_s16le")
                    self.assertEqual(prepared.metadata.sample_rate, 16000)
                    self.assertEqual(prepared.metadata.channels, 1)
                finally:
                    prepared.cleanup()
                self.assertIsNotNone(temporary)
                self.assertFalse(temporary.exists())

    def test_is_valid_voice_embedding_and_clean_profiles(self):
        valid = np.linspace(-10, 10, 38).tolist()
        zero_emb = [0.0] * 38
        self.assertTrue(is_valid_voice_embedding(valid))
        self.assertFalse(is_valid_voice_embedding(zero_emb))
        self.assertFalse(is_valid_voice_embedding([]))
        self.assertFalse(is_valid_voice_embedding([np.nan] * 38))

        dirty_profiles = {
            "Valid User": {"embedding": valid, "sample_count": 10},
            "Corrupt User": {"embedding": zero_emb, "sample_count": 5},
        }
        cleaned = clean_voice_profiles(dirty_profiles)
        self.assertIn("Valid User", cleaned)
        self.assertNotIn("Corrupt User", cleaned)

    def test_voice_profiles_separability_and_conflicts(self):
        v1 = np.linspace(1, 38, 38).tolist()
        v2 = (np.linspace(1, 38, 38) + 0.05).tolist()  # Muito similar a v1
        v3 = np.linspace(38, 1, 38).tolist()          # Bem distinto

        profiles = {
            "Jogador 1": {"embedding": v1},
            "Jogador 2 (Clone)": {"embedding": v2},
            "Jogador 3 (Distinto)": {"embedding": v3},
        }
        report = calculate_voice_profiles_separability(profiles)
        self.assertEqual(report["total_profiles"], 3)
        self.assertGreater(len(report["conflicts"]), 0)
        self.assertEqual(report["conflicts"][0]["profile_a"], "Jogador 1")
        self.assertEqual(report["conflicts"][0]["profile_b"], "Jogador 2 (Clone)")

    def test_delete_voice_profile(self):
        profiles_file = self.temp_path / "profiles.json"
        with mock.patch.object(app_module, "VOICE_PROFILES_FILE", profiles_file):
            train_voice_profile_from_audio("[Heroi (Jogador)]", self.mono_wav)
            self.assertTrue(delete_voice_profile("[Heroi (Jogador)]"))
            saved = json.loads(profiles_file.read_text(encoding="utf-8"))
            self.assertNotIn("[Heroi (Jogador)]", saved)

    def test_hungarian_optimal_assignment_multi_speaker(self):
        feat_a = np.linspace(1, 10, 38, dtype=np.float32)
        feat_b = np.linspace(10, 1, 38, dtype=np.float32)
        clusters = {
            "Voz Física #1": {"mean_feature": feat_a.tolist(), "count": 25},
            "Voz Física #2": {"mean_feature": feat_b.tolist(), "count": 18},
        }
        profiles = {
            "Personagem Alfa": {"embedding": feat_a.tolist()},
            "Personagem Beta": {"embedding": feat_b.tolist()},
        }
        preds = match_voice_clusters_to_profiles(clusters, profiles)
        self.assertEqual(preds["Voz Física #1"]["player"], "Personagem Alfa")
        self.assertEqual(preds["Voz Física #2"]["player"], "Personagem Beta")
        self.assertTrue(preds["Voz Física #1"]["is_confident"])

    def test_highpass_filter_attenuates_low_frequency_rumble(self):
        sr = 16000
        t = np.arange(sr) / sr
        # 30Hz rumble (sub-bass) + 440Hz vocal tone
        rumble = np.sin(2 * np.pi * 30 * t)
        vocal = np.sin(2 * np.pi * 440 * t)
        combined = (rumble + vocal).astype(np.float32)

        filtered = apply_highpass_filter(combined, sample_rate=sr, cutoff=80.0)
        self.assertEqual(len(filtered), len(combined))
        # Rumble (<80Hz) should have significantly lower energy after filter
        rumble_energy_before = float(np.mean(rumble ** 2))
        filtered_sub30 = float(np.mean(filtered ** 2))
        self.assertLess(filtered_sub30, float(np.mean(combined ** 2)))

    def test_spectral_noise_suppression_reduces_noise_floor(self):
        sr = 16000
        t = np.arange(sr) / sr
        speech = np.sin(2 * np.pi * 300 * t) * (t > 0.3) * (t < 0.7)
        noise = np.random.normal(0, 0.05, sr)
        noisy_audio = (speech + noise).astype(np.float32)

        denoised = apply_spectral_noise_suppression(noisy_audio, sample_rate=sr, noise_reduction_ratio=0.85)
        self.assertEqual(len(denoised), len(noisy_audio))
        self.assertFalse(np.isnan(denoised).any())
        self.assertFalse(np.isinf(denoised).any())

    def test_speech_vocal_enhancer_maintains_signal_integrity(self):
        sr = 16000
        t = np.arange(sr) / sr
        audio = (0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
        enhanced = apply_speech_vocal_enhancer(audio, sample_rate=sr)
        self.assertEqual(len(enhanced), len(audio))
        self.assertFalse(np.isnan(enhanced).any())

    def test_dynamic_gain_control_levels_audio(self):
        sr = 16000
        t = np.arange(sr) / sr
        # Quiet voice (very low amplitude)
        quiet_voice = (0.01 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        leveled = apply_dynamic_gain_control(quiet_voice, sample_rate=sr, target_rms_db=-20.0)
        
        rms_before = np.sqrt(np.mean(quiet_voice ** 2))
        rms_after = np.sqrt(np.mean(leveled ** 2))
        self.assertGreater(rms_after, rms_before)
        self.assertLessEqual(np.max(np.abs(leveled)), 1.0)

    def test_enhance_audio_pipeline_and_file_cleaner(self):
        sr = 16000
        t = np.arange(sr * 2) / sr
        audio = (0.2 * np.sin(2 * np.pi * 350 * t) + np.random.normal(0, 0.02, len(t))).astype(np.float32)
        
        # Test pipeline array
        enhanced = enhance_audio_pipeline(audio, sample_rate=sr, enable_denoise=True, enable_enhance=True, enable_agc=True)
        self.assertEqual(len(enhanced), len(audio))
        self.assertFalse(np.isnan(enhanced).any())

        # Test file cleaner
        src_path = self.temp_path / "noisy.wav"
        wavfile.write(src_path, sr, (audio * 32767).astype(np.int16))
        out_path = clean_and_enhance_audio_file(src_path)
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)

    def test_transient_suppression_removes_impulsive_spikes(self):
        sr = 16000
        t = np.arange(sr) / sr
        speech = 0.2 * np.sin(2 * np.pi * 300 * t)
        # Injeta um pico impulsivo seco (ex: dado rolando na mesa ou batida)
        speech[4000] = 0.99
        speech[4001] = -0.95
        
        cleaned = apply_transient_suppression(speech, sample_rate=sr)
        self.assertEqual(len(cleaned), len(speech))
        self.assertLess(abs(cleaned[4000]), 0.5)
        self.assertFalse(np.isnan(cleaned).any())

    def test_dereverberation_room_echo_decay_attenuation(self):
        sr = 16000
        t = np.arange(sr * 2) / sr
        # Sinal simulado com cauda de reverberação
        impulse = np.zeros(len(t), dtype=np.float32)
        impulse[1000] = 1.0
        # Cria reflexões tardias
        reverb_tail = np.convolve(impulse, np.exp(-np.linspace(0, 5, 2000)), mode='same')
        audio_with_reverb = reverb_tail + 0.1 * np.sin(2 * np.pi * 440 * t)
        
        dereverbed = apply_dereverberation(audio_with_reverb, sample_rate=sr)
        self.assertEqual(len(dereverbed), len(audio_with_reverb))
        self.assertFalse(np.isnan(dereverbed).any())

    def test_conference_room_pipeline_end_to_end(self):
        sr = 16000
        t = np.arange(sr * 3) / sr
        # Simulando uma sala de 7 pessoas:
        # - Pessoa perto (alta amplitude)
        # - Pessoa longe (baixa amplitude)
        # - Batida de dado na mesa
        near_speaker = 0.6 * np.sin(2 * np.pi * 300 * t) * (t < 1.0)
        far_speaker = 0.03 * np.sin(2 * np.pi * 450 * t) * (t > 1.5) * (t < 2.5)
        dice_roll = np.zeros(len(t))
        dice_roll[int(sr * 1.2)] = 0.95
        room_audio = (near_speaker + far_speaker + dice_roll).astype(np.float32)

        processed = enhance_audio_pipeline(
            room_audio,
            sample_rate=sr,
            enable_denoise=True,
            enable_enhance=True,
            enable_agc=True,
            enable_conference_mode=True,
        )
        self.assertEqual(len(processed), len(room_audio))
        self.assertFalse(np.isnan(processed).any())

        # Verifica se o participante distante (far_speaker) teve seu volume amplificado significativamente
        far_slice_raw = room_audio[int(sr * 1.6):int(sr * 2.4)]
        far_slice_proc = processed[int(sr * 1.6):int(sr * 2.4)]
        self.assertGreater(float(np.mean(far_slice_proc ** 2)), float(np.mean(far_slice_raw ** 2)))


if __name__ == "__main__":
    unittest.main()
