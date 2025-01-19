[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/kst/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
</div>

# K-Scale Speech Tokenizer

Welcome to the K-Scale Speech Tokenizer Library! For more information, see the [documentation](https://docs.kscale.dev/machinelearning/kst).


## Download model weights: 
```bash
kscale model download kst_1024
```


## Encode and decode audio:
```python
from kst import KST

kst = KST(model="kst_1024")
codes = kst.encode(audio_input)
wav = kst.decode(codes)
```


## Appreciation
1. [Encodec](https://github.com/facebookresearch/encodec)
2. [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
2. [MagVIT2](https://magvit.cs.cmu.edu/v2/)
