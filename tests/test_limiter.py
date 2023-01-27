import pickle
from functools import lru_cache

import pytest
import numpy as np
from cylimiter import Limiter


@lru_cache(1)
def get_audio():
    # 1s of 44.1kHz audio in fp32, centered on zero, with dynamic range [-5, 5]
    return ((np.random.rand(44100) - 0.5) * 10).astype(np.float32)


def test_limiter():
    # Example of applying limiter that copies the signal
    audio = get_audio()
    limiter = Limiter()
    audio_lim = limiter.apply(audio)
    assert (audio_lim != audio).any()
    np.testing.assert_array_less(audio_lim, 1.0)


def test_limiter_nondefault_args():
    # Example of applying limiter that copies the signal
    audio = get_audio()
    limiter = Limiter(attack=0.5, release=0.9, delay=100, threshold=0.9)
    audio_lim = limiter.apply(audio)
    assert (audio_lim != audio).any()


def test_limiter_nondefault_args_validation():
    with pytest.raises(AssertionError):
        limiter = Limiter(attack=1.1)

    with pytest.raises(AssertionError):
        limiter = Limiter(attack=0)

    with pytest.raises(AssertionError):
        limiter = Limiter(release=1.1)

    with pytest.raises(AssertionError):
        limiter = Limiter(threshold=-0.5)

    # does not raise when above 1 to support other audio effects
    # in effect chains that may drive the signal above 1.0 magnitude
    limiter = Limiter(threshold=1.1)

    with pytest.raises(AssertionError):
        limiter = Limiter(delay=0)


def test_limiter_inplace_fails_with_python_list():
    limiter = Limiter()

    # does not raise
    limiter.apply([1.0] * 22050)

    with pytest.raises(AssertionError):
        limiter.apply_inplace([1.0] * 22050)


def test_limiter_inplace_fails_with_float64():
    limiter = Limiter()

    audio = get_audio().astype(np.float64)

    with pytest.raises(AssertionError):
        limiter.apply_inplace(audio)


def test_limiter_inplace():
    limiter = Limiter()
    chunk_size = 1200  # for streaming processing

    # Example of applying limiter in-place (more efficient)
    audio = get_audio()
    print(len(audio))
    for offset in range(0, len(audio), chunk_size):
        chunk = audio[offset : offset + chunk_size]
        chunk_cpy = np.copy(chunk)
        limiter.apply_inplace(chunk)
        # ... do sth with chunk
        assert (chunk != chunk_cpy).any()
        np.testing.assert_array_less(chunk, 1.0)


def test_limiter_reset():
    # Example of applying limiter that copies the signal
    audio = get_audio()
    limiter = Limiter()
    audio_lim = limiter.apply(audio)
    audio_lim2 = limiter.apply(audio)
    assert (
        audio_lim != audio_lim2
    ).any()  # some state was accumulated in limiter so the results is different

    limiter.reset()
    audio_lim_reset = limiter.apply(audio)
    np.testing.assert_allclose(audio_lim, audio_lim_reset)


def test_limiter_pickle_works():
    limiter_default = Limiter()
    limiter = Limiter(attack=0.555, delay=1000, threshold=0.2, release=0.01)
    data = pickle.dumps(limiter)
    limiter_unpickled = pickle.loads(data)

    audio = get_audio()
    audio_lim = limiter.apply(audio)
    assert (audio != audio_lim).any()
    audio_lim_unpickled = limiter_unpickled.apply(audio)
    assert (audio != audio_lim_unpickled).any()

    audio_lim_default = limiter_default.apply(audio)
    assert (audio_lim != audio_lim_default).any()
    assert (audio_lim_unpickled != audio_lim_default).any()

    np.testing.assert_allclose(audio_lim, audio_lim_unpickled)
