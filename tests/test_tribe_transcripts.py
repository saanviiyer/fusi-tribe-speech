import pandas as pd

from fusi_tribe.insilico.tribe import place_transcript


def test_each_chunk_keeps_only_its_own_words_on_the_global_clock():
    # A 150 s file split into 60 s chunks, the file starting at 10 s on the timeline.
    transcript = pd.DataFrame({"start": [5.0, 59.9, 60.0, 119.0, 130.0], "text": list("abcde")})
    chunks = [(10.0, 0.0, 60.0), (70.0, 60.0, 60.0), (130.0, 120.0, 30.0)]
    placed = [place_transcript(transcript, *c) for c in chunks]
    assert [list(p.text) for p in placed] == [["a", "b"], ["c", "d"], ["e"]]
    assert list(pd.concat(placed).start) == [15.0, 69.9, 70.0, 129.0, 140.0]


def test_words_are_not_duplicated_across_chunks():
    transcript = pd.DataFrame({"start": [float(t) for t in range(0, 416, 3)], "text": "w"})
    chunks = [(o, o, min(60.0, 416 - o)) for o in range(0, 416, 60)]
    total = sum(len(place_transcript(transcript, *c)) for c in chunks)
    assert total == len(transcript)


def test_truncate_context_keeps_target_and_n_preceding_words():
    from fusi_tribe.insilico.tribe import truncate_context
    assert truncate_context("a b c d e", 2) == "c d e"
    assert truncate_context("a b", 5) == "a b"
    assert truncate_context("a b c", 0) == "c"
