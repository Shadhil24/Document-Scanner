import json
import sys

import main


def test_cli_single_file_smoke(tmp_path, monkeypatch, capsys):
    in_file = tmp_path / "input.jpg"
    in_file.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    def fake_process_image_path(path, output_dir, use_ocr=False, debug=False, cfg_overrides=None):
        assert path == str(in_file)
        assert output_dir == str(out_dir)
        return str(out_dir / "input__angle_+0.0.jpg"), {
            "accepted": True,
            "score": 0.99,
            "angle": 0.0,
            "route": "mock",
            "elapsed_ms": 5.0,
            "scale": 1.0,
        }

    monkeypatch.setattr(main, "process_image_path", fake_process_image_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scan", "--input", str(in_file), "--output", str(out_dir)],
    )

    main.main()
    out = capsys.readouterr().out
    assert "Processing:" in out
    assert "Done. 1/1 accepted." in out


def test_process_image_path_writes_meta(tmp_path, monkeypatch):
    in_file = tmp_path / "doc.jpg"
    in_file.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    fake_img = object()
    fake_padded = object()
    fake_info = {
        "accepted": True,
        "score": 0.88,
        "angle": 2.5,
        "route": "mock",
        "elapsed_ms": 12.0,
        "scale": 1.0,
    }

    monkeypatch.setattr(main, "load_image", lambda path: fake_img)
    monkeypatch.setattr(main, "build_retry_configs", lambda cfg: [cfg])
    monkeypatch.setattr(main, "process_once", lambda *args, **kwargs: (fake_padded, fake_info))
    monkeypatch.setattr(main, "save_image", lambda *args, **kwargs: None)

    out_path, info = main.process_image_path(str(in_file), str(out_dir))
    assert out_path is not None
    assert info["accepted"] is True

    meta_path = out_dir / "doc__meta.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["route"] == "mock"
