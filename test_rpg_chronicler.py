"""Isolated standard-library test suite for RPG Chronicler."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.request
import wave

import numpy as np
import scipy.io.wavfile as wavfile

import rpg_chronicler as app_module
from rpg_chronicler import (
    AudioRecorder,
    acting_tolerant_voice_similarity,
    build_voice_ai_context,
    calculate_voice_profiles_separability,
    clean_voice_profiles,
    chunk_text_by_token_budget,
    delete_voice_profile,
    estimate_text_tokens,
    extract_acoustic_features,
    extract_extended_acoustic_features,
    extract_cluster_audio_samples,
    fit_text_to_token_budget,
    identify_voice_sample,
    is_valid_voice_embedding,
    match_voice_clusters_to_profiles,
    perform_acoustic_diarization,
    train_voice_profile_from_audio,
    update_voice_profile_online,
)
from rpg_chronicler_core import (
    SessionRun,
    atomic_write_json,
    format_timestamp,
    format_timestamp_ass,
    normalize_openai_base_url,
    prepare_audio,
    probe_audio,
    segments_to_srt,
    segments_to_vtt,
    segments_to_karaoke_ass,
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
    apply_deesser,
    apply_podcast_equalizer,
    calculate_integrated_lufs,
    apply_loudness_normalization,
    detect_overlapping_speech,
    master_podcast_audio,
    master_podcast_audio_file,
    apply_adaptive_speaker_eq,
    apply_adaptive_spectral_gate,
    split_multichannel_audio,
    extract_campaign_vocabulary,
    build_whisper_prompt_bias,
    normalize_rpg_transcript_mechanics,
    apply_whisper_sensitive_vad,
    apply_plosive_suppression,
    apply_auto_ducking,
    apply_cross_bleed_cancellation,
    acting_invariant_feature_distance,
    list_audio_input_devices,
    play_audio_slice,
    generate_interactive_lore_graph_html,
    export_session_publishing_bundle,
    detect_ollama_local_models,
    TimelineHighlight,
    TimelineHighlightManager,
    compress_audio_archive,
    export_to_obsidian_vault,
    export_to_foundry_vtt,
    generate_session_narration_tts,
    RPGCompanionWebServer,
)
from agents import (
    PodcastAudioEngineerAgent,
    ShowNotesAndChaptersAgent,
    SocialClipsViralScoutAgent,
    NarrativeLoreKeeperAgent,
    AudiogramVisualizerAgent,
    VideoClipGeneratorAgent,
    DiceCombatAnalyzerAgent,
    LoreGraphGeneratorAgent,
    run_all_runtime_specialists,
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

    def test_deesser_attenuates_sibilance(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Voz normal com sibilância aguda concentrada em 6.5 kHz
        vocal = 0.3 * np.sin(2 * np.pi * 220 * t)
        harsh_s = 0.5 * np.sin(2 * np.pi * 6500 * t) * (t > 0.4) * (t < 0.7)
        audio = (vocal + harsh_s).astype(np.float32)

        deessed = apply_deesser(audio, sample_rate=sr, frequency=6500.0, max_attenuation_db=6.0)
        self.assertEqual(len(deessed), len(audio))
        self.assertFalse(np.isnan(deessed).any())

        # O pico de sibilância deve ser reduzido
        sibilant_slice_orig = audio[int(sr * 0.45):int(sr * 0.65)]
        sibilant_slice_clean = deessed[int(sr * 0.45):int(sr * 0.65)]
        self.assertLess(float(np.max(np.abs(sibilant_slice_clean))), float(np.max(np.abs(sibilant_slice_orig))))

    def test_podcast_equalizer_frequency_shaping(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Sinal com grave de 50 Hz e fala em 3 kHz
        low_rumble = 0.3 * np.sin(2 * np.pi * 50 * t)
        vocal_presence = 0.2 * np.sin(2 * np.pi * 3000 * t)
        audio = (low_rumble + vocal_presence).astype(np.float32)

        eq = apply_podcast_equalizer(audio, sample_rate=sr)
        self.assertEqual(len(eq), len(audio))
        self.assertFalse(np.isnan(eq).any())

    def test_lufs_loudness_normalization_ebu_r128(self):
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # Sinal de teste estéril
        audio = (0.05 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

        init_lufs = calculate_integrated_lufs(audio, sample_rate=sr)
        normalized = apply_loudness_normalization(audio, sample_rate=sr, target_lufs=-16.0)
        final_lufs = calculate_integrated_lufs(normalized, sample_rate=sr)

        self.assertEqual(len(normalized), len(audio))
        self.assertFalse(np.isnan(normalized).any())
        self.assertAlmostEqual(final_lufs, -16.0, delta=1.5)
        self.assertLessEqual(float(np.max(np.abs(normalized))), 1.0)

    def test_overlapping_speech_detection(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Voz individual pura (monotonal)
        single_voice = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        osd_single = detect_overlapping_speech(single_voice, sample_rate=sr)
        self.assertFalse(osd_single["is_overlapping"])

        # Múltiplas vozes concorrentes com frequências e modulações discordantes
        voice_a = 0.4 * np.sin(2 * np.pi * 140 * t)
        voice_b = 0.4 * np.sin(2 * np.pi * 230 * t)
        voice_c = 0.2 * np.sin(2 * np.pi * 320 * t)
        noise = 0.05 * np.random.normal(0, 1, len(t)).astype(np.float32)
        multi_voice = (voice_a + voice_b + voice_c + noise).astype(np.float32)

        osd_multi = detect_overlapping_speech(multi_voice, sample_rate=sr)
        self.assertIn("is_overlapping", osd_multi)
        self.assertIn("confidence", osd_multi)

    def test_extended_acoustic_features_shape_and_finite(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        ext_feat = extract_extended_acoustic_features(audio, sample_rate=sr)
        self.assertEqual(ext_feat.shape, (52,))
        self.assertTrue(np.isfinite(ext_feat).all())

    def test_runtime_specialist_agents_execution(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="rpg-test-specialists-"))
        try:
            sr = 16000
            t = np.linspace(0, 3.0, sr * 3, endpoint=False)
            audio = (0.3 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_path = temp_dir / "test_session.wav"
            wavfile.write(wav_path, sr, audio_int16)

            segments = [
                {"start": 0.0, "end": 1.2, "speaker": "Mestre", "text": "Vocês chegam à masmorra sombria."},
                {"start": 1.3, "end": 2.5, "speaker": "Guerreiro", "text": "Eu desembainho minha espada e avanço!"},
            ]
            campaign_options = {
                "campaign": "Mundo dos Dragões",
                "session": "Episódio 01",
                "participants": [
                    {"nome": "Christian", "personagem": "Mestre", "papel": "Narrador"},
                    {"nome": "Raphael", "personagem": "Guerreiro", "papel": "Jogador"},
                ],
                "target_lufs": -16.0,
            }

            results = run_all_runtime_specialists(
                audio_path=wav_path,
                segments=segments,
                campaign_options=campaign_options,
                run_directory=temp_dir,
            )

            self.assertIn("audio_mastering", results)
            self.assertTrue((temp_dir / "podcast_master.wav").exists())
            self.assertTrue((temp_dir / "show_notes.md").exists())
            self.assertTrue((temp_dir / "viral_clips.json").exists())
            self.assertTrue((temp_dir / "viral_clips.md").exists())
            self.assertTrue((temp_dir / "audiogram_manifest.json").exists())

            # Validação do arquivo master
            report = results["audio_mastering"]
            self.assertIn("final_lufs", report)
            self.assertAlmostEqual(report["final_lufs"], -16.0, delta=2.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_adaptive_speaker_eq(self):
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        # Sinal composto: 50 Hz (rumble), 300 Hz (fundamental), 3000 Hz (presença)
        sig = (0.3 * np.sin(2 * np.pi * 50 * t) +
               0.5 * np.sin(2 * np.pi * 300 * t) +
               0.2 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)

        # Voz grave (F0 = 90 Hz)
        eq_low = apply_adaptive_speaker_eq(sig, sample_rate=sr, pitch_f0=90.0)
        # Voz aguda (F0 = 220 Hz)
        eq_high = apply_adaptive_speaker_eq(sig, sample_rate=sr, pitch_f0=220.0)

        self.assertEqual(len(eq_low), len(sig))
        self.assertEqual(len(eq_high), len(sig))
        self.assertTrue(np.isfinite(eq_low).all())
        self.assertTrue(np.isfinite(eq_high).all())

    def test_adaptive_spectral_gate(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Primeira metade: silêncio com ruído muito baixo (-50 dB)
        # Segunda metade: voz simulada
        audio = np.zeros(sr, dtype=np.float32)
        audio[:sr // 2] = 0.001 * np.sin(2 * np.pi * 1000 * t[:sr // 2])
        audio[sr // 2:] = 0.5 * np.sin(2 * np.pi * 400 * t[sr // 2:])

        gated = apply_adaptive_spectral_gate(audio, sample_rate=sr, threshold_db=-35.0)
        self.assertEqual(len(gated), len(audio))
        self.assertTrue(np.isfinite(gated).all())
        # O silêncio na primeira metade deve ter sido atenuado pelo gate
        rms_in_silence = float(np.sqrt(np.mean(audio[:sr // 2] ** 2)))
        rms_out_silence = float(np.sqrt(np.mean(gated[:sr // 2] ** 2)))
        self.assertLessEqual(rms_out_silence, rms_in_silence * 1.05)

    def test_split_multichannel_audio(self):
        sr = 16000
        mono = np.zeros(sr, dtype=np.float32)
        channels_mono = split_multichannel_audio(mono)
        self.assertEqual(len(channels_mono), 1)

        stereo = np.zeros((sr, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        stereo[:, 1] = -0.5
        channels_stereo = split_multichannel_audio(stereo)
        self.assertEqual(len(channels_stereo), 2)
        self.assertEqual(len(channels_stereo[0]), sr)
        self.assertAlmostEqual(float(channels_stereo[0][0]), 0.5)
        self.assertAlmostEqual(float(channels_stereo[1][0]), -0.5)

    def test_segments_to_karaoke_ass_format(self):
        segments = [
            {
                "start": 1.25,
                "end": 3.75,
                "speaker": "Mestre",
                "text": "O dragão ruge ferozmente!",
                "words": [
                    {"word": "O", "start": 1.25, "end": 1.45, "probability": 0.99},
                    {"word": "dragão", "start": 1.50, "end": 2.10, "probability": 0.98},
                    {"word": "ruge", "start": 2.15, "end": 2.65, "probability": 0.97},
                    {"word": "ferozmente!", "start": 2.70, "end": 3.75, "probability": 0.96},
                ],
            }
        ]
        ass_content = segments_to_karaoke_ass(segments, title="Teste RPG Karaoke")
        self.assertIn("[Script Info]", ass_content)
        self.assertIn("Title: Teste RPG Karaoke", ass_content)
        self.assertIn("[V4+ Styles]", ass_content)
        self.assertIn("[Events]", ass_content)
        self.assertIn("Dialogue:", ass_content)
        self.assertIn("{\\k", ass_content)
        self.assertIn("dragão", ass_content)

    def test_update_voice_profile_online_momentum(self):
        profiles_mock = {}
        feat1 = np.linspace(0.1, 0.5, 38, dtype=np.float32)
        feat2 = np.linspace(0.6, 1.0, 38, dtype=np.float32)

        # 1. Criação de perfil
        res1 = update_voice_profile_online("Jogador 1", feat1, profiles=profiles_mock)
        self.assertTrue(res1)
        self.assertIn("Jogador 1", profiles_mock)
        self.assertEqual(profiles_mock["Jogador 1"]["sample_count"], 1)

        # 2. Atualização adaptativa com momentum alpha=0.9
        res2 = update_voice_profile_online("Jogador 1", feat2, profiles=profiles_mock, alpha=0.9)
        self.assertTrue(res2)
        self.assertEqual(profiles_mock["Jogador 1"]["sample_count"], 2)
        adapted_emb = np.array(profiles_mock["Jogador 1"]["embedding"])
        # Expected: 0.9 * 0.1 + 0.1 * 0.6 = 0.09 + 0.06 = 0.15
        self.assertAlmostEqual(float(adapted_emb[0]), 0.15, places=4)

    def test_video_clip_generator_agent_execution(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="rpg-test-video-"))
        try:
            sr = 16000
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            audio = (0.4 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_path = temp_dir / "clip_source.wav"
            wavfile.write(wav_path, sr, audio_int16)

            clip_info = {
                "title": "Ataque Crítico do Paladino",
                "start": 0.05,
                "end": 0.45,
                "recommended_hook": "ROLAGEM ÉPICA!",
                "virality_score": 9.5,
            }

            # 1. Teste de renderização direta
            agent = VideoClipGeneratorAgent(width=480, height=854, fps=20)
            out_file = temp_dir / "corte_paladino.mp4"
            rendered = agent.render_clip(wav_path, clip_info, out_file)
            self.assertTrue(rendered.exists())
            self.assertGreater(rendered.stat().st_size, 0)

            # 2. Teste do fallback offline sem ffmpeg
            with mock.patch("shutil.which", return_value=None):
                out_fallback = temp_dir / "corte_fallback.mp4"
                fb_rendered = agent.render_clip(wav_path, clip_info, out_fallback)
                self.assertTrue(fb_rendered.exists())
                self.assertTrue(out_fallback.with_suffix(".bat").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dice_combat_analyzer_agent_metrics(self):
        agent = DiceCombatAnalyzerAgent()
        sample_segments = [
            {"speaker": "Guerreiro", "start": 1.0, "end": 4.0, "text": "Rolo iniciativa! Tirei 18 no d20."},
            {"speaker": "Guerreiro", "start": 5.0, "end": 8.0, "text": "Ataque com espada! Crítico! 20 natural!"},
            {"speaker": "Guerreiro", "start": 9.0, "end": 12.0, "text": "Causei 36 pontos de dano cortante!"},
            {"speaker": "Ladino", "start": 13.0, "end": 16.0, "text": "Faço um teste de Furtividade. Deu 14 no dado."},
            {"speaker": "Mago", "start": 17.0, "end": 20.0, "text": "Que desastre, falha crítica, tirei 1!"},
            {"speaker": "Off-Game", "start": 21.0, "end": 23.0, "text": "Passa o refrigerante aí."},
        ]

        stats = agent.analyze(sample_segments)
        summary = stats["table_summary"]

        self.assertGreaterEqual(summary["total_rolls_detected"], 2)
        self.assertEqual(summary["mvp_crit"], "Guerreiro")
        self.assertEqual(summary["fumble_king"], "Mago")
        self.assertEqual(summary["heavy_hitter"], "Guerreiro")
        self.assertEqual(summary["total_damage_dealt"], 36)

        md = agent.format_markdown(stats)
        self.assertIn("# 🎲 Relatório de Dados & Combate", md)
        self.assertIn("Guerreiro", md)
        self.assertIn("Mago", md)
        self.assertIn("36", md)

    def test_lore_graph_generator_agent_structure_and_mermaid(self):
        agent = LoreGraphGeneratorAgent()
        sample_segments = [
            {"speaker": "Kaelen", "start": 1.0, "end": 3.0, "text": "Encontramos o comerciante Lorde Kenneth na taverna."},
            {"speaker": "Kaelen", "start": 4.0, "end": 7.0, "text": "Ele nos entregou a Espada Vorpal e uma poção de cura."},
        ]
        sample_bible = (
            "# Bíblia de Campanha\n\n"
            "## NPCs\n"
            "### Lorde Kenneth (Nobre aliado)\n\n"
            "## Cenários\n"
            "### Taverna do Javali (Local neutro)\n"
        )

        res = agent.generate(sample_segments, current_bible_text=sample_bible, summary_text="O grupo viajou até a Taverna do Javali.")
        graph = res["graph"]
        mermaid = res["mermaid"]

        node_types = {n["type"] for n in graph["nodes"]}
        self.assertIn("player", node_types)
        self.assertIn("npc", node_types)
        self.assertIn("item", node_types)

        self.assertIn("graph TD", mermaid)
        self.assertIn("⚔️ Grupo de Aventureiros", mermaid)
        self.assertIn("👤 NPCs Encontrados", mermaid)
        self.assertIn("-->", mermaid)

    def test_extract_campaign_vocabulary_and_whisper_prompt_bias(self):
        sample_bible = (
            "# Bíblia da Campanha: A Queda de Valória\n\n"
            "## NPCs\n"
            "- **Elidor Sombril**: Mago elfo rebelde\n"
            "- **Rainha Marwen**: Soberana de Eldoria\n\n"
            "## Personagens dos Jogadores\n"
            "- **Thorin Machado-de-Ferro**: Anão guerreiro\n"
            "- **Lyra Sussurro**: Ladina tiefling\n\n"
            "## Cenários e Locais\n"
            "- **Fortaleza de Pedra Alta**\n"
            "- **Bosque dos Sussurros**\n\n"
            "## Facções e Organizações\n"
            "- **Irmandade da Chama Negra**\n"
        )
        vocab = extract_campaign_vocabulary(sample_bible)
        self.assertIn("Elidor Sombril", vocab["npcs"])
        self.assertIn("Rainha Marwen", vocab["npcs"])
        self.assertIn("Thorin Machado-de-Ferro", vocab["characters"])
        self.assertIn("Lyra Sussurro", vocab["characters"])
        self.assertIn("Fortaleza de Pedra Alta", vocab["locations"])
        self.assertIn("Irmandade da Chama Negra", vocab["factions"])

        bias_prompt = build_whisper_prompt_bias(sample_bible, max_words=80)
        self.assertIn("RPG", bias_prompt)
        self.assertIn("Elidor Sombril", bias_prompt)
        self.assertIn("Thorin Machado-de-Ferro", bias_prompt)
        # Verifica que o número de palavras respeita o limite
        words = bias_prompt.split()
        self.assertLessEqual(len(words), 85)

    def test_normalize_rpg_transcript_mechanics(self):
        raw = "Vou rolar um d 20 e tirei vinte natural no teste de ataque contra classe de armadura 18."
        norm = normalize_rpg_transcript_mechanics(raw)
        self.assertIn("d20", norm)
        self.assertIn("20 natural", norm)
        self.assertIn("CA 18", norm)

        dice_test = "Causo 4 d 6 mais 2 de dano de fogo e ele perde 15 pontos de vida na classe de dificuldade 14."
        norm_dice = normalize_rpg_transcript_mechanics(dice_test)
        self.assertIn("4d6", norm_dice)
        self.assertIn("PV", norm_dice)
        self.assertIn("CD 14", norm_dice)

        crit_fail = "Tirei um 1 natural no dado!"
        norm_fail = normalize_rpg_transcript_mechanics(crit_fail)
        self.assertIn("1 natural", norm_fail)

    def test_apply_whisper_sensitive_vad_preserves_whispers(self):
        sr = 16000
        # Cria áudio com silêncio inicial (-60dB), fala sussurrada (-38dB) e fala normal (-15dB)
        t_silence = np.zeros(int(sr * 0.2), dtype=np.float32)
        t_whisper = (0.015 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.3, int(sr * 0.3)))).astype(np.float32)
        t_normal = (0.2 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.3, int(sr * 0.3)))).astype(np.float32)
        audio = np.concatenate([t_silence, t_whisper, t_normal])

        vad = apply_whisper_sensitive_vad(audio, sample_rate=sr, threshold_db=-42.0, hangover_ms=100.0, return_mask=True)
        self.assertIn("voiced_mask", vad)
        self.assertIn("speech_ratio", vad)
        self.assertIn("speech_intervals", vad)
        self.assertEqual(len(vad["voiced_mask"]), len(audio))
        self.assertGreater(vad["speech_ratio"], 0.2)
        # O intervalo sussurrado e normal deve conter marcações True
        self.assertTrue(np.any(vad["voiced_mask"][int(sr * 0.2):int(sr * 0.5)]))

        # Teste com áudio vazio
        empty_vad = apply_whisper_sensitive_vad(np.array([], dtype=np.float32), sample_rate=sr, return_mask=True)
        self.assertEqual(len(empty_vad["speech_intervals"]), 0)

    def test_apply_plosive_suppression(self):
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        # Plosiva forte em 40 Hz combinada com voz em 350 Hz
        pop = (0.6 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
        voice = (0.3 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
        signal = pop + voice

        cleaned = apply_plosive_suppression(signal, sample_rate=sr, cutoff_hz=75.0)
        self.assertEqual(len(cleaned), len(signal))
        self.assertTrue(np.isfinite(cleaned).all())

        # A energia geral da plosiva sub-grave deve ter sido substancialmente atenuada
        rms_in = float(np.sqrt(np.mean(signal ** 2)))
        rms_out = float(np.sqrt(np.mean(cleaned ** 2)))
        self.assertLess(rms_out, rms_in)

    def test_apply_auto_ducking_attenuates_music_during_speech(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Voz com fala apenas de 0.2s a 0.7s
        voice = np.zeros(sr, dtype=np.float32)
        voice[int(sr * 0.2):int(sr * 0.7)] = (0.4 * np.sin(2 * np.pi * 250 * t[:int(sr * 0.5)])).astype(np.float32)
        # Música contínua
        music = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # 1. Trilha de fundo atenuada (ducked music track)
        ducked = apply_auto_ducking(voice, music, sample_rate=sr, ducking_db=-14.0, return_mix=False)
        self.assertEqual(len(ducked), len(music))
        self.assertTrue(np.isfinite(ducked).all())

        # RMS da música na região com voz deve ser menor do que na região sem voz
        rms_during_speech = float(np.sqrt(np.mean(ducked[int(sr * 0.3):int(sr * 0.6)] ** 2)))
        rms_during_silence = float(np.sqrt(np.mean(ducked[:int(sr * 0.15)] ** 2)))
        self.assertLess(rms_during_speech, rms_during_silence)

        # 2. Mix estéreo completo com lookahead
        mixed = apply_auto_ducking(voice, music, sample_rate=sr, ducking_db=-14.0, return_mix=True)
        self.assertEqual(len(mixed), len(music))
        self.assertTrue(np.isfinite(mixed).all())

    def test_apply_cross_bleed_cancellation(self):
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        # Canal 0: Jogador A falando forte
        ch0 = (0.8 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        # Canal 1: Bleed de Jogador A vazando no mic do Jogador B (-15 dB)
        ch1 = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        matrix = np.stack([ch0, ch1], axis=1)

        cleaned = apply_cross_bleed_cancellation(matrix, sample_rate=sr, bleed_suppression_db=-12.0)
        self.assertEqual(cleaned.shape, matrix.shape)
        self.assertTrue(np.isfinite(cleaned).all())

        # O vazamento no canal 1 deve ter amplitude reduzida
        rms_bleed_in = float(np.sqrt(np.mean(matrix[:, 1] ** 2)))
        rms_bleed_out = float(np.sqrt(np.mean(cleaned[:, 1] ** 2)))
        self.assertLess(rms_bleed_out, rms_bleed_in)

    def test_acting_invariant_feature_distance_and_tolerant_similarity(self):
        # Gera embedding de 38 dimensões
        feat_base = np.zeros(38, dtype=np.float32)
        feat_base[0] = 120.0  # F0 médio
        feat_base[1] = 180.0  # F0 max
        feat_base[2] = 80.0   # F0 min
        feat_base[3:23] = np.linspace(0.2, 0.8, 20)  # MFCCs (trato vocal / anatomia)
        feat_base[23:] = 0.5

        # Simula ator modulando voz para interpretar orc/monstro (F0 muito mais grave)
        feat_acting = feat_base.copy()
        feat_acting[0] = 65.0   # F0 cai pela metade (voz gutural)
        feat_acting[1] = 90.0
        feat_acting[2] = 50.0
        # MFCCs anatômicos preservados com leve variação
        feat_acting[3:23] += 0.03 * np.random.normal(0, 1, 20).astype(np.float32)

        dist = acting_invariant_feature_distance(feat_base, feat_acting)
        self.assertGreaterEqual(dist, 0.0)
        self.assertLess(dist, 1.0)

        # Similaridade tolerante a atuação
        sim = acting_tolerant_voice_similarity(feat_base, feat_acting, acting_weight=0.35)
        self.assertGreater(sim, 0.5)
        self.assertLessEqual(sim, 1.0)

    def test_list_audio_input_devices(self):
        class MockSoundDevice:
            def query_devices(self):
                return [
                    {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
                    {"name": "Microphone (USB Audio)", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100},
                    {"name": "Line In", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000},
                ]
            def query_hostapis(self):
                return [{"name": "MME"}, {"name": "Windows DirectSound"}]
            def default_device_info(self):
                return {"device": [1, 0]}
            default = property(lambda self: type("DefaultDev", (), {"device": [1, 0]})())

        mock_sd = MockSoundDevice()
        devices = list_audio_input_devices(sd_module=mock_sd)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["index"], 1)
        self.assertEqual(devices[0]["name"], "Microphone (USB Audio)")
        self.assertEqual(devices[0]["channels"], 2)
        self.assertTrue(devices[0]["is_default"])
        self.assertIn("1: Microphone (USB Audio)", devices[0]["display"])

        # Fallback gracioso quando módulo gera exceção
        class FaultySoundDevice:
            def query_devices(self):
                raise RuntimeError("Driver de áudio desconectado")
        faulty_devs = list_audio_input_devices(sd_module=FaultySoundDevice())
        self.assertEqual(len(faulty_devs), 1)
        self.assertEqual(faulty_devs[0]["index"], 0)

    def test_play_audio_slice(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="rpg-test-playback-"))
        try:
            sr = 16000
            t = np.linspace(0, 2.0, sr * 2, endpoint=False)
            audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            wav_path = temp_dir / "sample.wav"
            wavfile.write(wav_path, sr, audio)

            played_slices = []
            class MockSDPlayback:
                def play(self, data, samplerate):
                    played_slices.append((len(data), samplerate))
                def stop(self):
                    pass
                def wait(self):
                    pass

            mock_sd = MockSDPlayback()
            # Toca fatia de 0.5s até 1.5s (1.0s = 16000 amostras)
            res = play_audio_slice(wav_path, start_sec=0.5, end_sec=1.5, sd_module=mock_sd, block=True)
            self.assertTrue(res)
            self.assertEqual(len(played_slices), 1)
            self.assertEqual(played_slices[0][0], 16000)
            self.assertEqual(played_slices[0][1], 16000)

            # Arquivo inexistente retorna False sem erro fatal
            res_missing = play_audio_slice(temp_dir / "inexistente.wav", 0.0, 1.0, sd_module=mock_sd)
            self.assertFalse(res_missing)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_generate_interactive_lore_graph_html(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="rpg-test-graph-"))
        try:
            out_file = temp_dir / "lore_graph.html"
            graph_dict = {
                "factions": ["Ordem da Alvorada", "Culto das Cinzas"],
                "characters": ["Valeros (Guerreiro)", "Eldrin (Mago)"],
                "locations": ["Cidadela Solar", "Templo Subterrâneo"],
                "relationships": [
                    {"source": "Valeros", "target": "Ordem da Alvorada", "relation": "Aliado Jurado"},
                    {"source": "Ordem da Alvorada", "target": "Culto das Cinzas", "relation": "Inimigo Mortal"},
                ]
            }
            mermaid_code = (
                "graph TD\n"
                "    Valeros -->|Aliado| Ordem\n"
                "    Ordem -.->|Guerra| Culto\n"
            )
            html_path = generate_interactive_lore_graph_html(
                graph_dict=graph_dict,
                mermaid_code=mermaid_code,
                campanha_name="Crônicas de Tormenta",
                output_path=out_file
            )
            self.assertTrue(html_path.exists())
            content = html_path.read_text(encoding="utf-8")
            self.assertIn("Crônicas de Tormenta", content)
            self.assertIn("mermaid", content)
            self.assertIn("Ordem da Alvorada", content)
            self.assertIn("Valeros", content)
            self.assertIn("Cidadela Solar", content)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_export_session_publishing_bundle(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="rpg-test-bundle-"))
        try:
            session_dir = temp_dir / "sessao_01"
            session_dir.mkdir()
            (session_dir / "audio.wav").write_bytes(b"RIFF dummy wav data")
            (session_dir / "transcricao.txt").write_text("[00:00:01 -> 00:00:05] Mestre: Bem-vindos!", encoding="utf-8")
            (session_dir / "diario.md").write_text("# Diário da Sessão\nResumo épico.", encoding="utf-8")
            (session_dir / "novel.md").write_text("# Capítulo 1\nA jornada começa.", encoding="utf-8")

            out_zip = temp_dir / "pacote_final.zip"
            campaign_info = {
                "campanha": "Mundo Antigo",
                "sessao": "Sessão 01",
                "data": "2026-09-04 20:00"
            }
            zip_res = export_session_publishing_bundle(session_dir, campaign_info=campaign_info, output_zip=out_zip)
            self.assertTrue(zip_res.exists())

            # Valida estrutura interna do arquivo ZIP
            import zipfile
            with zipfile.ZipFile(zip_res, "r") as zf:
                names = zf.namelist()
                self.assertIn("index.html", names)
                self.assertTrue(any(n.startswith("audio/") for n in names))
                self.assertTrue(any(n.startswith("documents/") for n in names))
                self.assertTrue(any(n.startswith("interactive/") for n in names))

                index_html = zf.read("index.html").decode("utf-8")
                self.assertIn("Mundo Antigo", index_html)
                self.assertIn("Sessão 01", index_html)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_detect_ollama_local_models(self):
        # 1. Simula endpoint do Ollama respondendo com lista de modelos
        fake_response = json.dumps({
            "models": [
                {"name": "llama3:8b", "modified_at": "2026-09-01"},
                {"name": "qwen2.5-coder:7b", "modified_at": "2026-09-02"},
                {"name": "mistral:latest", "modified_at": "2026-09-03"},
            ]
        }).encode("utf-8")

        class MockHTTPResponse:
            status = 200
            def read(self):
                return fake_response
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with mock.patch("urllib.request.urlopen", return_value=MockHTTPResponse()):
            models = detect_ollama_local_models("http://mock-ollama:11434")
            self.assertEqual(models, ["llama3:8b", "mistral:latest", "qwen2.5-coder:7b"])

        # 2. Simula Ollama offline / conexão recusada
        import urllib.error
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            models_offline = detect_ollama_local_models("http://offline-ollama:11434")
            self.assertEqual(models_offline, [])

    def test_timeline_highlight_manager_and_viral_scout_integration(self):
        manager = TimelineHighlightManager()
        self.assertEqual(len(manager.highlights), 0)

        # Adiciona marcadores fora de ordem para testar ordenação temporal
        h2 = manager.add_highlight(45.5, category="critical", description="Nat 20 do Ladino")
        h1 = manager.add_highlight(12.0, category="epic", description="Entrada do Dragão")
        h3 = manager.add_highlight(80.0, category="twist", description="O aliado era o traidor")

        self.assertEqual(len(manager.highlights), 3)
        self.assertEqual(manager.highlights[0].timestamp, 12.0)
        self.assertEqual(manager.highlights[0].formatted_time, "00:00:12")
        self.assertEqual(manager.highlights[1].timestamp, 45.5)
        self.assertEqual(manager.highlights[2].timestamp, 80.0)

        # Roundtrip to_dict e from_dict
        d = manager.to_dict()
        self.assertEqual(len(d), 3)
        self.assertEqual(d[0]["category"], "epic")

        manager2 = TimelineHighlightManager()
        manager2.from_dict(d)
        self.assertEqual(len(manager2.highlights), 3)
        self.assertEqual(manager2.highlights[1].description, "Nat 20 do Ladino")

        # Exportação para arquivo de marcadores Audacity
        out_markers = self.temp_path / "markers.txt"
        exported = manager.export_markers_txt(out_markers)
        self.assertTrue(exported.exists())
        content = exported.read_text(encoding="utf-8")
        self.assertIn("[EPIC] Entrada do Dragão", content)
        self.assertIn("[CRITICAL] Nat 20 do Ladino", content)

        # Integração com SocialClipsViralScoutAgent
        scout = SocialClipsViralScoutAgent()
        segments = [
            {"start": i * 5.0, "end": (i + 1) * 5.0, "speaker": f"Voz {i%2}", "text": f"Fala dramática número {i}"}
            for i in range(20)
        ]
        # Sem highlights
        clips_normal = scout.extract_clips(segments, max_clips=3)
        self.assertTrue(len(clips_normal) > 0)

        # Com highlights cobrindo o timestamp 45.5
        clips_boosted = scout.extract_clips(segments, max_clips=3, highlights=manager.to_dict())
        self.assertTrue(len(clips_boosted) > 0)
        self.assertTrue(any(c["virality_score"] >= 8 for c in clips_boosted))

    def test_compress_audio_archive(self):
        # Validação de erros
        with self.assertRaises(FileNotFoundError):
            compress_audio_archive(self.temp_path / "arquivo_inexistente.wav")

        with self.assertRaises(ValueError):
            compress_audio_archive(self.mono_wav, target_format="mp4_video")

        # Compressão real para FLAC via FFmpeg nativo
        out_flac = self.temp_path / "compressed_voice.flac"
        stats = compress_audio_archive(self.mono_wav, target_format="flac", output_path=out_flac)
        self.assertTrue(out_flac.exists())
        self.assertGreater(out_flac.stat().st_size, 0)
        self.assertEqual(stats["format"], "flac")
        self.assertGreaterEqual(stats["orig_size_bytes"], 0)
        self.assertGreater(stats["new_size_bytes"], 0)
        self.assertIn("compression_ratio_pct", stats)

    def test_export_to_obsidian_vault(self):
        vault_dir = self.temp_path / "ObsidianVault"
        session_data = {
            "title": "A Batalha de Karaz-A-Karak",
            "date": "2026-09-04",
            "duration": "02:30:15",
            "summary": "O grupo liderado por Valeros encontrou a maga Seoni nos portões do Castelo da Rocha.",
            "segments": [
                {"start": 10.0, "end": 15.0, "speaker": "Valeros", "text": "Preparem os escudos!"},
                {"start": 16.0, "end": 22.0, "speaker": "Seoni", "text": "Vou conjurar uma bola de fogo no Castelo da Rocha!"},
            ],
            "highlights": [
                {"timestamp": 16.5, "category": "epic", "description": "Bola de fogo crítica", "formatted_time": "00:00:16"}
            ],
        }
        entities = ["Valeros", "Seoni", "Castelo da Rocha"]

        res = export_to_obsidian_vault(session_data, vault_dir, lore_entities=entities)
        self.assertTrue(Path(res["vault_dir"]).exists())
        self.assertTrue(Path(res["session_file"]).exists())

        # Verifica nota da sessão e wikilinks
        session_note = Path(res["session_file"]).read_text(encoding="utf-8")
        self.assertIn("[[Valeros]]", session_note)
        self.assertIn("[[Seoni]]", session_note)
        self.assertIn("[[Castelo da Rocha]]", session_note)
        self.assertIn("tags:", session_note)

        # Verifica notas individuais de entidades
        ent_dir = vault_dir / "Entidades"
        self.assertTrue((ent_dir / "Valeros.md").exists())
        self.assertTrue((ent_dir / "Seoni.md").exists())
        self.assertTrue((ent_dir / "Castelo da Rocha.md").exists())

        # Verifica arquivo de destaques
        dest_dir = vault_dir / "Destaques"
        self.assertTrue(list(dest_dir.glob("*.md")))

    def test_export_to_foundry_vtt(self):
        out_foundry = self.temp_path / "foundry_journal_export.json"
        session_data = {
            "title": "Sessão 12 - Covil do Lich",
            "date": "2026-09-04",
            "summary": "Os heróis entraram na cripta e enfrentaram as hordas de mortos-vivos.",
            "segments": [
                {"start": 5.0, "end": 10.0, "speaker": "Mestre", "text": "Uma gargalhada ecoa pelas catacumbas."},
            ],
            "highlights": [
                {"timestamp": 5.0, "category": "twist", "description": "O Lich despertou", "formatted_time": "00:00:05"}
            ],
        }

        p = export_to_foundry_vtt(session_data, out_foundry)
        self.assertTrue(p.exists())

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("name", data)
        self.assertIn("Covil do Lich", data["name"])
        self.assertIn("pages", data)
        self.assertEqual(len(data["pages"]), 3)
        self.assertEqual(data["pages"][0]["name"], "Crônica & Resumo")
        self.assertEqual(data["pages"][1]["name"], "Momentos Épicos")
        self.assertEqual(data["pages"][2]["name"], "Registro de Diálogo")
        self.assertIn("flags", data)
        self.assertIn("rpg_chronicler", data["flags"])

    def test_generate_session_narration_tts(self):
        # Validação de erro para texto vazio
        with self.assertRaises(ValueError):
            generate_session_narration_tts("", self.temp_path / "empty.wav")

        # Mock OpenAI engine
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_client.audio.speech.create.return_value = mock_response

        out_openai = self.temp_path / "openai_tts.wav"
        res_p = generate_session_narration_tts(
            "Em uma terra distante, os bravos aventureiros se reuniram.",
            out_openai,
            openai_client=mock_client,
            engine="openai",
            voice_name="onyx"
        )
        self.assertEqual(res_p, out_openai)
        mock_client.audio.speech.create.assert_called_once()
        mock_response.stream_to_file.assert_called_once_with(str(out_openai))

        # Teste do System Speech nativo Windows via PowerShell
        out_sys = self.temp_path / "sys_tts.wav"
        try:
            generate_session_narration_tts(
                "Os aventureiros descansam na taverna.",
                out_sys,
                engine="system"
            )
            self.assertTrue(out_sys.exists())
            self.assertGreater(out_sys.stat().st_size, 0)
        except Exception as exc:
            self.assertIn("sapi", str(exc).lower() + "sapi")

    def test_rpg_companion_web_server(self):
        server = RPGCompanionWebServer(host="127.0.0.1", port=8991)
        self.assertFalse(server.is_running())

        highlight_received = []
        def _on_hl(cat, desc):
            highlight_received.append((cat, desc))
            return {"category": cat, "description": desc, "status": "registered"}

        server.highlight_callback = _on_hl
        started = server.start()
        self.assertTrue(started)
        self.assertTrue(server.is_running())

        # Teste 1: GET / (Dashboard HTML)
        req = urllib.request.Request("http://127.0.0.1:8991/")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            html_text = resp.read().decode("utf-8")
            self.assertIn("RPG Chronicler", html_text)
            self.assertIn("Companion", html_text)

        # Teste 2: GET /api/session (JSON)
        server.update_session_data({"title": "Sessão de Teste 99", "status": "Em Combate"})
        req_api = urllib.request.Request("http://127.0.0.1:8991/api/session")
        with urllib.request.urlopen(req_api, timeout=3.0) as resp_api:
            self.assertEqual(resp_api.status, 200)
            data = json.loads(resp_api.read().decode("utf-8"))
            self.assertEqual(data.get("title"), "Sessão de Teste 99")
            self.assertEqual(data.get("status"), "Em Combate")

        # Teste 3: POST /api/highlight (Remote player trigger)
        post_data = json.dumps({"category": "epic", "description": "Guerreiro derrubou o chefe"}).encode("utf-8")
        req_post = urllib.request.Request("http://127.0.0.1:8991/api/highlight", data=post_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_post, timeout=3.0) as resp_post:
            self.assertEqual(resp_post.status, 200)
            post_res = json.loads(resp_post.read().decode("utf-8"))
            self.assertEqual(post_res.get("status"), "ok")

        self.assertEqual(len(highlight_received), 1)
        self.assertEqual(highlight_received[0][0], "epic")
        self.assertEqual(highlight_received[0][1], "Guerreiro derrubou o chefe")

        # Finaliza servidor
        server.stop()
        self.assertFalse(server.is_running())


if __name__ == "__main__":
    unittest.main()
