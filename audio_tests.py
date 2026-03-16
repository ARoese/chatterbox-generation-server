import torch
import torchaudio as ta
from torchaudio import functional as F
from torch.nn.functional import pad
import os
from os import path
import math

# test padding prior to vad to reduce cutoffs. This test went well.

SOURCE_DIR = "test_samples/samples"
OUT_DIR = "test_samples/outputs"

for path in os.listdir(SOURCE_DIR):
    source_path = f"{SOURCE_DIR}/{path}"
    out_path = f"{OUT_DIR}/{path}"

    print(f"{source_path} -> {out_path}")

    sound, sr = ta.load(source_path)
    pad_samples_count = int(sr * 0.4)
    sound = pad(sound, (pad_samples_count,0), "constant", value=0)
    trimmed = F.vad(sound, sr) # trim off preceeding silence and/or trailing part of the final prefix word


    ta.save(out_path, trimmed, sr)

