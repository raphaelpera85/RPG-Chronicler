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
    chunk_text_by_token_budget,
    estimate_text_tokens,
    extract_acoustic_features,
    extract_cluster_audio_samples,
    fit_text_to_token_budget,
    identify_voice_sample,
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


if __name__ == "__main__":
    unittest.main()
