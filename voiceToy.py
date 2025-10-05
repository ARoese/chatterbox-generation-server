import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import re
import playsound
from tempfile import NamedTemporaryFile
import argparse

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Generate voicelines using reference audio")
parser.add_argument("reference_wav")
parser.add_argument("-l", default=None)
args = parser.parse_args()

print("Loading model...")
if(args.l is not None and args.l):
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    def generate_wav(dialogue, exaggeration, cfg_weight, temperature):
        return model.generate(dialogue, language_id=args.l, audio_prompt_path=ref_audio, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    if(args.l not in model.get_supported_languages().keys()):
        print(args.l, "is not a supported language. Supported languages are:")
        print(model.get_supported_languages())
        exit()
    else:
        print("Using language:", model.get_supported_languages()[args.l])
else:
    model = ChatterboxTTS.from_pretrained(device=device)
    def generate_wav(dialogue, exaggeration, cfg_weight, temperature):
        return model.generate(dialogue, audio_prompt_path=ref_audio, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)

print("Model loaded")

ref_audio = args.reference_wav #"ref_files/lefin-general-long.wav"
print("Using reference audio:", ref_audio)

exaggeration=0.5
cfg_weight=0.5
temperature=0.8

print("Using parameters:", exaggeration, cfg_weight, temperature)

while(True):
    dialogue = input("Enter Dialogue: ")
    if(dialogue.startswith("[")):
        matches = re.findall(r"\[\s*([0-9.]*) +([0-9.]*) +([0-9.]*)\s*\]", dialogue)
        if(len(matches) != 0):
            newVals = matches[0]
            exaggeration, cfg_weight, temperature = map(float, newVals)
            print("Using parameters:", exaggeration, cfg_weight, temperature)
            continue
    wav = generate_wav(dialogue, exaggeration, cfg_weight, temperature)
    with NamedTemporaryFile(delete=True, suffix='.wav') as tmpWav:
        ta.save(tmpWav, wav, model.sr, format="wav", bits_per_sample=16)
        tmpWav.flush()
        playsound.playsound(tmpWav.name, block=True)