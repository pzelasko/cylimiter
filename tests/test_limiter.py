import random
import pickle
import pytest
from cylimiter import Limiter


def get_audio():
    ret = []
    for i in range(44100):
        ret.append(random.random())
    return ret


def test_limiter():
    # Example of applying limiter that copies the signal
    audio = get_audio()
    limiter = Limiter()
    audio_lim = limiter.limit(audio)


def test_limiter_nondefault_args():
    # Example of applying limiter that copies the signal
    audio = get_audio()
    limiter = Limiter(attack=0.5, release=0.9, delay=100, threshold=0.9)
    audio_lim = limiter.limit(audio)


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


def test_limiter_inplace():
    limiter = Limiter()
    chunk_size = 1200  # for streaming processing

    # Example of applying limiter in-place (more efficient)
    audio = get_audio()
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        limiter.limit_inplace(chunk)
        # ... do sth with chunk


def test_limiter_reset():
    # Example of applying limiter that copies the signal
    audio = get_audio()
    limiter = Limiter()
    audio_lim = limiter.limit(audio)
    audio_lim2 = limiter.limit(audio)
    assert (
        audio_lim != audio_lim2
    )  # some state was accumulated in limiter so the results is different

    limiter.reset()
    audio_lim_reset = limiter.limit(audio)
    assert audio_lim == audio_lim_reset

def test_limiter_pickle_works():
    limiter_default = Limiter()
    limiter = Limiter(attack=0.555, delay=1000, threshold=0.2, release=0.01)
    data = pickle.dumps(limiter)
    limiter_unpickled = pickle.loads(data)

    audio = get_audio()
    audio_lim = limiter.limit(audio)
    assert audio != audio_lim
    audio_lim_unpickled = limiter_unpickled.limit(audio)
    assert audio != audio_lim_unpickled

    audio_lim_default = limiter_default.limit(audio)
    assert audio_lim != audio_lim_default
    assert audio_lim_unpickled != audio_lim_default

    assert audio_lim == audio_lim_unpickled
