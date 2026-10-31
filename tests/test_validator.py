import pytest
import os
import tempfile
from src.utils.validator import validate_media_file

def test_validate_nonexistent_file():
    is_valid, msg, meta = validate_media_file("/path/to/nonexistent_file.mp4")
    assert not is_valid
    assert "does not exist" in msg

def test_validate_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        is_valid, msg, meta = validate_media_file(tmp_path)
        assert not is_valid
        assert "empty" in msg.lower()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_validate_unsupported_extension():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"fake pdf content")
        tmp_path = tmp.name
    try:
        is_valid, msg, meta = validate_media_file(tmp_path)
        assert not is_valid
        assert "unsupported" in msg.lower()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_validate_corrupted_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"not a valid jpeg byte stream")
        tmp_path = tmp.name
    try:
        is_valid, msg, meta = validate_media_file(tmp_path)
        assert not is_valid
        assert "corrupt" in msg.lower() or "error" in msg.lower()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
