"""Tests pour le CLI (run.py).

Lancez : pytest tests/test_cli.py -v
"""

import subprocess


class TestCLI:

    def test_help_without_args(self):
        """run.py sans argument affiche l'aide."""
        r = subprocess.run(
            ["python", "run.py"],
            capture_output=True, text=True,
        )
        combined = r.stdout + r.stderr
        assert "index" in combined
        assert "query" in combined

    def test_index_help(self):
        """run.py index --help fonctionne."""
        r = subprocess.run(
            ["python", "run.py", "index", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "pdf-dir" in r.stdout or "pdf_dir" in r.stdout

    def test_query_help(self):
        """run.py query --help fonctionne."""
        r = subprocess.run(
            ["python", "run.py", "query", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "question" in r.stdout

    def test_history_help(self):
        """run.py history --help fonctionne."""
        r = subprocess.run(
            ["python", "run.py", "history", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
