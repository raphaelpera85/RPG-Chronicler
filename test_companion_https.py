"""Real TLS and browser permission regression checks, without using personal data."""
import ipaddress
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request

from cryptography import x509
from rpg_chronicler_core import RPGCompanionWebServer, ensure_companion_ssl_cert


class CompanionHTTPS(unittest.TestCase):
    def test_incomplete_tls_client_does_not_block_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            server = RPGCompanionWebServer("127.0.0.1", 0, True, directory, 0)
            self.assertTrue(server.start(), server.last_error)
            with socket.create_connection(("127.0.0.1", server.port), timeout=3) as idle_client:
                # A valid second client proves the accept loop remains responsive.
                context = ssl.create_default_context(cafile=str(Path(directory) / "companion_ca.pem"))
                try:
                    with urllib.request.urlopen(server.get_url(), context=context, timeout=2) as response:
                        self.assertEqual(response.status, 200)
                    stopped = threading.Event()
                    def stop():
                        server.stop()
                        stopped.set()
                    worker = threading.Thread(target=stop, daemon=True)
                    worker.start()
                    self.assertTrue(stopped.wait(2), "Idle TLS client blocked shutdown")
                finally:
                    idle_client.close()
                    server.stop()

    def test_verified_tls_and_setup_only_exposes_public_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            server = RPGCompanionWebServer("127.0.0.1", 0, True, directory, 0)
            self.assertTrue(server.start(), server.last_error)
            self.addCleanup(server.stop)
            context = ssl.create_default_context(cafile=str(Path(directory) / "companion_ca.pem"))
            with urllib.request.urlopen(server.get_url(), context=context, timeout=3) as response:
                self.assertIn("requestMicrophone", response.read().decode())
            with urllib.request.urlopen(server.get_setup_url(), timeout=3) as response:
                page = response.read().decode()
                self.assertIn(server.get_url(), page)
                self.assertIn(server.certificate_fingerprint(), page)
            with urllib.request.urlopen(server.get_setup_url() + "/companion_ca.cer", timeout=3) as response:
                self.assertEqual(response.read(), (Path(directory) / "companion_ca.cer").read_bytes())
            for path in ("/companion_ca_key.pem", "/companion_key.pem", "/api/session", "/../companion_ca_key.pem"):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(server.get_setup_url() + path, timeout=3)
                self.assertEqual(error.exception.code, 404)
                error.exception.close()
            server.stop()
            self.assertFalse(server.is_running())
            self.assertIsNone(server.setup_server)

    def test_lan_ip_change_renews_leaf_but_preserves_trusted_ca(self):
        with tempfile.TemporaryDirectory() as directory:
            cert_path, _ = ensure_companion_ssl_cert(directory, "192.168.1.23")
            original = Path(cert_path).read_bytes()
            ca = (Path(directory) / "companion_ca.cer").read_bytes()
            ensure_companion_ssl_cert(directory, "192.168.1.23")
            self.assertEqual(original, Path(cert_path).read_bytes())
            ensure_companion_ssl_cert(directory, "192.168.1.24")
            self.assertNotEqual(original, Path(cert_path).read_bytes())
            self.assertEqual(ca, (Path(directory) / "companion_ca.cer").read_bytes())
            cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
            ips = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.IPAddress)
            self.assertIn(ipaddress.ip_address("192.168.1.24"), ips)

    def test_failed_tls_start_closes_socket_and_reports_reason(self):
        server = RPGCompanionWebServer("127.0.0.1", 0, True)
        with mock.patch("rpg_chronicler_core.ensure_companion_ssl_cert", side_effect=RuntimeError("certificate failure")):
            self.assertFalse(server.start())
        self.assertIn("certificate failure", server.last_error)
        self.assertFalse(server.is_running())
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", server.port))

    def test_microphone_guards_and_errors_in_javascript(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required for browser logic checks")
        page = RPGCompanionWebServer().render_dashboard_html()
        script = page.split("<script>", 1)[1].split("</script>", 1)[0]
        helper = script.split("let currentPhrase", 1)[0]
        checks = r'''
const assert = require('node:assert/strict');
global.window = {isSecureContext:false};
Object.defineProperty(global, 'navigator', {value:{}, configurable:true});
global.MediaRecorder = function() {};
(async () => {
  await assert.rejects(requestMicrophone(), /antes de pedir permissão/);
  window.isSecureContext = true;
  await assert.rejects(requestMicrophone(), /Chrome ou Safari/);
  let calls = 0;
  const stream = {};
  navigator.mediaDevices = {getUserMedia: async (constraints) => {
    calls++; assert.deepEqual(constraints, {audio:true}); return stream;
  }};
  assert.equal(await requestMicrophone(), stream);
  assert.equal(calls, 1);
  for (const [name, text] of [['NotAllowedError', /configurações/], ['NotFoundError', /Nenhum/], ['NotReadableError', /outros aplicativos/]]) {
    navigator.mediaDevices.getUserMedia = async () => {throw {name};};
    await assert.rejects(requestMicrophone(), text);
  }
})().catch(err => { console.error(err); process.exitCode = 1; });
'''
        with tempfile.TemporaryDirectory() as directory:
            full_script = Path(directory) / "dashboard.js"
            full_script.write_text(script, encoding="utf-8")
            subprocess.run([node, "--check", str(full_script)], check=True, capture_output=True)
            result = subprocess.run([node, "-e", helper + checks], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(page.count("await requestMicrophone()"), 2)

    def test_audio_payload_validation_rejects_invalid_and_oversized_data(self):
        server = RPGCompanionWebServer()
        invalid = server.on_remote_voice_train({"player_label": "[A]", "audio_b64": "%%%"})
        self.assertEqual(invalid["status"], "error")
        oversized = server.on_remote_satellite_chunk({
            "player_label": "[A]", "audio_b64": "A" * ((8 * 1024 * 1024 * 4 // 3) + 100)
        })
        self.assertEqual(oversized["status"], "error")


if __name__ == "__main__":
    unittest.main()
