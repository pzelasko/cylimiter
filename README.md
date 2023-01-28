# cylimiter

A small package with stateful audio limiter implementation in Cython. Since the limiter is stateful it is suitable for streaming audio processing.

We're slowly growing into other effects beyond limiter, starting with streaming reverb in v0.4.

## What's new?

### v0.4: Added reverb with RIR

```python
from cylimiter import ReverbRIR

reverb = ReverbRIR()  # default RIR
# or user-provided RIR:
#  reverb = ReverbRIR([0.001, 0.021, 0.007, ...])

reverb.apply(audio)          # makes a copy
reverb.apply_inplace(audio)  # no copies with np.array input

# preserves context between calls for streaming applications
for chunk in audio_source:
    reverb.apply_inplace(chunk)
```



### Examples

```python
import numpy as np
from cylimiter import Limiter

limiter = Limiter(attack=0.5, release=0.9, delay=100, threshold=0.9)
chunk_size = 1200  # for streaming processing

# Example of applying limiter in-place (more efficient)
audio = np.random.randn(44100) * 10
for i in range(0, 44100, chunk_size):
    chunk = audio[i * chunk_size: (i + 1) * chunk_size]
    limiter.limit_inplace(chunk)
    # ... do sth with chunk

# Example of applying limiter that copies the signal
audio = np.random.randn(1, 44100) * 10
audio_lim = limiter.limit(audio)

# Reset the limiter to re-use it on other signals
limiter.reset()
```

## Installation

From PyPI via pip:
```bash
pip install cylimiter
```

From source:
```bash
git clone https://github.com/pzelasko/cylimiter
cd cylimiter
pip install .
```

Re-generate C++ sources from Cython:
```bash
cd extensions
cython -3 --cplus *.pyx
```

## Motivation

I couldn't easily find a package that implements audio limiter in Python in a suitable way for streaming audio. The closest (and the main inspiration) is [this gist by @bastibe](https://gist.github.com/bastibe/747283c55aad66404046). Since the algorithm is auto-regressive, I figured C++ will be much more efficient than Python.
