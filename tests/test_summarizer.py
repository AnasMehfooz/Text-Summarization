import pytest

from summarizer import ModelNotAvailableError, Summarizer, clean_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hello\r\nWorld", "hello world"),
        ("  lots   of\t\tspace  ", "lots of space"),
        ("Tom: <b>Hi</b> Anna", "tom:  hi  anna"),
        ("<br>", ""),
    ],
)
def test_clean_text_matches_training_preprocessing(raw, expected):
    assert clean_text(raw) == expected


def test_empty_after_cleaning_raises_before_loading_model(tmp_path):
    s = Summarizer(model_path=str(tmp_path))  # no weights: load() would fail
    with pytest.raises(ValueError):
        s.summarize("<p></p>")
    assert not s.is_loaded


def test_missing_weights_raise_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="No model weights"):
        Summarizer(model_path=str(tmp_path)).load()


def test_lfs_pointer_is_detected(tmp_path):
    (tmp_path / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 242041896\n"
    )
    with pytest.raises(ModelNotAvailableError, match="git lfs pull"):
        Summarizer(model_path=str(tmp_path)).load()
