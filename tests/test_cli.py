"""Command line formatting and argument handling."""

import pytest

from raganything.cli import format_sources, main


def test_format_sources_lists_numbered_cited_passages():
    text = format_sources(
        [
            {
                "n": 1,
                "kind": "text",
                "file": "a.pdf",
                "page": 12,
                "excerpt": "câble U1000 R2V",
            },
            {
                "n": 2,
                "kind": "plan",
                "file": "b.pdf",
                "page": None,
                "excerpt": "tableau TGBT",
            },
            {
                "n": 3,
                "kind": "email",
                "file": "m.msg",
                "page": None,
                "excerpt": "x" * 400,
            },
        ]
    )
    lines = text.splitlines()
    assert lines[0].startswith("[1] 📄 a.pdf p.12 — « câble")
    assert lines[1].startswith("[2] 📐 b.pdf — «")
    assert lines[2].startswith("[3] ✉️ m.msg") and len(lines[2]) < 220


def test_inspect_prints_manifest(tmp_path, capsys):
    import json
    import zipfile

    bundle = tmp_path / "kb.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr(
            "manifest.json", json.dumps({"format": "raganything-bundle", "version": 1})
        )
    assert main(["inspect", str(bundle)]) == 0
    assert "raganything-bundle" in capsys.readouterr().out


def test_unknown_command_is_an_error():
    with pytest.raises(SystemExit):
        main(["bogus"])
