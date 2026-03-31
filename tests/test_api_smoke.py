from fastapi.testclient import TestClient

import api


def test_health_ok():
    client = TestClient(api.app)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "time" in body


def test_scan_angle_with_mocked_pipeline(monkeypatch):
    def fake_process_image_path(path, out_dir, use_ocr=False, debug=False):
        return "fake.jpg", {
            "accepted": True,
            "score": 0.91,
            "angle": 1.5,
            "route": "mock",
            "elapsed_ms": 4.0,
            "scale": 1.0,
        }

    monkeypatch.setattr(api, "process_image_path", fake_process_image_path)
    client = TestClient(api.app)
    res = client.post(
        "/scan/angle",
        files={"file": ("sample.jpg", b"fake-bytes", "image/jpeg")},
        data={"use_ocr": "false", "debug": "false"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["angle"] == 1.5
    assert body["route"] == "mock"
    assert body["score"] == 0.91
