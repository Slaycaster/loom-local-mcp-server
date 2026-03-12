import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from loom_agent.transcriber import Transcriber


def test_transcriber_init():
    transcriber = Transcriber(model_name="base")
    assert transcriber.model_name == "base"
    assert transcriber._model is None  # Lazy-loaded


def test_format_srt_timestamp():
    transcriber = Transcriber()
    assert transcriber.format_srt_timestamp(0.0) == "00:00:00,000"
    assert transcriber.format_srt_timestamp(62.5) == "00:01:02,500"
    assert transcriber.format_srt_timestamp(3661.123) == "01:01:01,123"


def test_segments_to_srt():
    transcriber = Transcriber()
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Hello world"},
        {"start": 2.5, "end": 5.0, "text": " Goodbye"},
    ]
    srt = transcriber.segments_to_srt(segments)
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in srt
    assert "2\n00:00:02,500 --> 00:00:05,000\nGoodbye" in srt


def test_write_srt_file():
    transcriber = Transcriber()
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Hello"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "transcript.srt"
        transcriber.write_srt(segments, output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "Hello" in content


@patch("loom_agent.transcriber.whisper")
def test_transcribe_calls_whisper(mock_whisper):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " Test transcript"}
        ]
    }
    mock_whisper.load_model.return_value = mock_model

    transcriber = Transcriber(model_name="base")
    result = transcriber.transcribe("/tmp/test.mp4")

    mock_whisper.load_model.assert_called_once_with("base")
    mock_model.transcribe.assert_called_once_with("/tmp/test.mp4")
    assert len(result) == 1
    assert result[0]["text"] == " Test transcript"
