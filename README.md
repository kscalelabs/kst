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

Welcome to the K-Scale Speech Tokenizer Library!


## To encode and decode audio:
```python
from kst.model import KSTConfig, KSTTokenizer

config = KSTConfig(
    n_filters=64,
    strides=[8, 5, 4, 2],
    dimension=1024,
    semantic_dimension=768,
    bidirectional=True,
    dilation_base=2,
    residual_kernel_size=3,
    n_residual_layers=1,
    lstm_layers=2,
    activation="ELU",
    codebook_size=1024,
    n_q=1,
    sample_rate=16000,
)

ckpt_path = "path/to/kst_1024.pt"
tokenizer = KSTTokenizer.load_from_checkpoint(config, ckpt_path)

audio_input = "path/to/audio/file.wav"
codes = tokenizer.encode(audio_input)
wav = tokenizer.decode(codes)
```


## Appreciation
1. [Encodec](https://github.com/facebookresearch/encodec)
2. [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
2. [MagVIT2](https://magvit.cs.cmu.edu/v2/)
