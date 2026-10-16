import numpy as np
import pandas as pd
import pytest

from fusi_tribe.insilico.swaps import edit_transcript

BLOCKS = [{"trial_type": "Speech", "onset": 0.0, "duration": 10.0},
          {"trial_type": "Audio only", "onset": 20.0, "duration": 10.0}]


def words(starts, texts, seq):
    return pd.DataFrame({"text": texts, "start": starts, "duration": 0.2,
                         "sequence_id": seq, "sentence": [" ".join(texts)] * len(texts)})


OWN = words([1.0, 2.0, 21.0, 22.0, 23.0], ["the", "cat", "xa", "yb", "zc"], [0, 0, 1, 1, 1])
TRUE = words([1.0, 2.0, 21.0, 22.0], ["the", "cat", "old", "dog"], [0, 0, 1, 1])


def test_none_removes_only_control_block_words():
    out = edit_transcript(OWN, TRUE, BLOCKS, "none")
    assert list(out.text) == ["the", "cat"]


def test_true_inserts_the_true_control_words_and_keeps_speech_words():
    out = edit_transcript(OWN, TRUE, BLOCKS, "true")
    assert list(out.text) == ["the", "cat", "old", "dog"]
    assert out.sequence_id.iloc[2] > OWN.sequence_id.max()   # no id collision with kept sentences


def test_shuffled_keeps_timestamps_and_word_set():
    out = edit_transcript(OWN, TRUE, BLOCKS, "shuffled", np.random.default_rng(3))
    ctrl = out[out.start >= 20]
    assert list(ctrl.start) == [21.0, 22.0, 23.0]
    assert sorted(ctrl.text) == ["xa", "yb", "zc"]
    assert all(ctrl.sentence == " " + " ".join(ctrl.text))
    assert list(out[out.start < 20].text) == ["the", "cat"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        edit_transcript(OWN, TRUE, BLOCKS, "reversed")
