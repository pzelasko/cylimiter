import pickle
from functools import lru_cache

import pytest
import numpy as np
from cylimiter import ReverbRIR


@lru_cache(1)
def get_audio():
    # 100ms of 44.1kHz audio in fp32, centered on zero, with dynamic range [-1, 1]
    return ((np.random.rand(4410) - 0.5) * 2).astype(np.float32)


def test_reverb_rir():
    # Example of applying reverb that copies the signal
    audio = get_audio()
    effect = ReverbRIR()
    out = effect.apply(audio)
    assert (out!= audio).any()
    np.testing.assert_array_less(out, 1.0)


def test_reverb_rir_mix():
    # Example of applying reverb that copies the signal
    audio = get_audio()
    effect = ReverbRIR()
    out = effect.apply(audio)

    effect2 = ReverbRIR(mix=0.9)
    out2 = effect.apply(audio)

    assert (out!= out2).any()


def test_reverb_nondefault_args():
    # Example of applying effect that copies the signal
    audio = get_audio()
    effect = ReverbRIR([0.0, 0.0, 0.0001, 0.02, 0.3, 0.7, 0.1, 0.1232, 0.000123, 0.0])
    audio_lim = effect.apply(audio)
    assert (audio_lim != audio).any()


def test_reverb_inplace_fails_with_python_list():
    effect = ReverbRIR()

    # does not raise
    effect.apply([1.0] * 22050)

    with pytest.raises(AssertionError):
        effect.apply_inplace([1.0] * 22050)


def test_reverb_inplace_fails_with_float64():
    effect = ReverbRIR()

    audio = get_audio().astype(np.float64)

    with pytest.raises(AssertionError):
        effect.apply_inplace(audio)


def test_reverb_inplace():
    effect = ReverbRIR()
    chunk_size = 1200  # for streaming processing

    # Example of applying effect in-place (more efficient)
    audio = get_audio()
    print(len(audio))
    for offset in range(0, len(audio), chunk_size):
        chunk = audio[offset : offset + chunk_size]
        chunk_cpy = np.copy(chunk)
        effect.apply_inplace(chunk)
        # ... do sth with chunk
        assert (chunk != chunk_cpy).any()
        np.testing.assert_array_less(chunk, 1.0)


def test_reverb_reset():
    # Example of applying effect that copies the signal
    audio = get_audio()
    effect = ReverbRIR()
    audio_lim = effect.apply(audio)
    audio_lim2 = effect.apply(audio)
    assert (
        audio_lim != audio_lim2
    ).any()  # some state was accumulated in effect so the results is different

    effect.reset()
    audio_lim_reset = effect.apply(audio)
    np.testing.assert_allclose(audio_lim, audio_lim_reset)


def test_reverb_pickle_works():
    effect_default = ReverbRIR()
    effect = ReverbRIR([0.0, 0.0, 0.001, 0.3, 0.7, 0.3, 0.001, 0.0, 0.0])
    data = pickle.dumps(effect)
    effect_unpickled = pickle.loads(data)

    audio = get_audio()
    audio_lim = effect.apply(audio)
    assert (audio != audio_lim).any()
    audio_lim_unpickled = effect_unpickled.apply(audio)
    assert (audio != audio_lim_unpickled).any()

    audio_lim_default = effect_default.apply(audio)
    assert (audio_lim != audio_lim_default).any()
    assert (audio_lim_unpickled != audio_lim_default).any()

    np.testing.assert_allclose(audio_lim, audio_lim_unpickled)
