import numpy as np
from cylimiter import ReverbRIR


def run(audio):
    effect = ReverbRIR()
    return effect.apply(audio)


def test_reverb_speed(benchmark):
    # 10s of audio in -1, 1 range as fp32
    audio = ((np.random.rand(441000) - 0.5) * 2).astype(np.float32)

    out = benchmark(run, audio)

    # Check basic properties of the output
    assert out.shape == audio.shape
    assert (out != audio).any()
