"""Logging must never cost a reply: malformed requests and a dead stdout."""

from __future__ import annotations

import io
import socket
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.request import urlopen

from remote_control import server as server_module
from remote_control.server import RemoteServer


class BrokenStdout(io.TextIOBase):
    """Stdout whose reader has gone away, as when `server | tee log` loses its tee."""

    def write(self, text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        pass                      # write() already fails; raising here too trips interpreter cleanup


class RequestLoggingTests(unittest.TestCase):
    def serve(self):
        runtime = SimpleNamespace(status=lambda: {"ok": True, "capabilities": {}})
        server = RemoteServer(("127.0.0.1", 0), runtime)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        return server.server_address[1]

    def test_pages_are_still_served_when_stdout_is_broken(self):
        port = self.serve()
        with mock.patch.object(sys, "stdout", BrokenStdout()):
            with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"voiceButton", response.read())
            with urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as response:
                self.assertEqual(response.status, 200)

    def test_a_malformed_request_gets_an_error_reply_not_a_dropped_connection(self):
        port = self.serve()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as conn, \
                mock.patch.object(sys, "stdout", io.StringIO()):
            conn.sendall(b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03 garbage\r\n\r\n")   # TLS bytes to http
            reply = conn.recv(1024)
        # No parsable HTTP version, so http.server answers HTTP/0.9-style: the error page
        # without a status line. What matters is that it answers instead of dropping.
        self.assertTrue(reply, "connection dropped without a reply")
        self.assertIn(b"400", reply)
        self.assertIn(b"Bad request", reply)

    def test_safe_print_swallows_a_closed_stream(self):
        with mock.patch.object(sys, "stdout", BrokenStdout()):
            server_module._safe_print("still fine")


if __name__ == "__main__":
    unittest.main()
