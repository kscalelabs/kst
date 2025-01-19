import torch
import torchaudio

from kst.model import KSTConfig, KSTTokenizer

if __name__ == "__main__":
    # python -m kst.model_test
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

    input_path = "samples/gt.wav"
    wav, sr = torchaudio.load(input_path, format="wav")

    ckpt_path = "kst_1024/kst_1024.pt"
    tokenizer = KSTTokenizer.load_from_checkpoint(config, ckpt_path)

    # monophonic checking
    if wav.shape[0] > 1:
        wav = wav[:1,:]

    if sr != tokenizer.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, tokenizer.sample_rate)

    wav = wav.unsqueeze(0)

    # Extract discrete codes from KST
    with torch.no_grad():
        codes = tokenizer.encode(wav) # codes: (n_q, B, T)
        output_wav = tokenizer.decode(codes)

    torchaudio.save(f'output.wav', output_wav.squeeze(0).detach().cpu(), tokenizer.sample_rate)
