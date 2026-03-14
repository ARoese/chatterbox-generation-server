import argparse
import sys
import socket
import os.path as path
from os import makedirs, remove
import argparse
import io
import gc
import num2words
import re
from typing import Callable, Optional
import tempfile
from random import randint

from torch import Tensor
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from prefix_trimming import get_prefix_phrase_end, trim_prefix_time, trim_prefix_time_tensor
import whisper_timestamped as wt

# Automatically detect the best available device
if torch.cuda.is_available():
    cb_device = torch.device("cuda")
    whisper_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    cb_device = torch.device("mps")
    whisper_device = torch.device("mps")
else:
    cb_device = torch.device("cpu")
    whisper_device = torch.device("cpu")

# can be 'cuda', 'cpu', or 'mps'
DEVICE_OVERRIDE = None
if DEVICE_OVERRIDE is not None:
    cb_device = DEVICE_OVERRIDE
    whisper_device = DEVICE_OVERRIDE

print(f"Using device {cb_device} for chatterbox")
print(f"Using device {whisper_device} for chatterbox")

SHORT_DIALOGUE_PREFIXES = [
    "Green eggs and ham.",
    "The true king.",
    "Please just work.",
    "The last try."
]

# for some reason, model performance drops rapidly as it is inferenced.
# Resolve this by unloading and then reloading the model after ~40 inferences
MODEL_REFRESH_INTERVAL = 40
if(MODEL_REFRESH_INTERVAL > 0):
    print(f"Model refresh enabled. Model will be reloaded every {MODEL_REFRESH_INTERVAL} generations for performance purposes")

parser = argparse.ArgumentParser(description="Generate voicelines using reference audio")
parser.add_argument("-l", default=None)
parser.add_argument("-p", type=int, default=9005)
parser.add_argument("-m", type=str, default=None, help="mappings file. Each line is in the form jarl||yarl, so all occurances of 'jarl' will be replaced with 'yarl' for better pronunciation")
parser.add_argument("--convert_numbers", action="store_true", help="use num2words to replace number strings with their word equivalents. This will give more stable outputs for spoken numbers.")
args = parser.parse_args()

subMap: dict[str,str] = {}
if args.m is not None:
    def handle_line(l: str) -> tuple[str,str]:
        pair = l.strip().split("||")
        if len(pair) != 2 or not bool(pair[0]) or not bool(pair[0]):
            raise ValueError(f"substitution file line '{l}' is not valid")
        
        return pair[0].strip(), pair[1].strip()
    with open(args.m, 'r') as mf:
        subMap = dict(map(handle_line, mf.readlines()))
    
    print("The following substitutions will be used:")
    for k,v in subMap.items():
        print(f"'{k}' -> '{v}'")

import string

def is_short_dialogue(dialogue: str) -> bool:
    """
    Chatterbox has trouble generating short dialogues. This attempts to predict whether chatterbox will struggle to generate a given line or dialogue.

    A line of dialogue is considered short if it is either:

        - <=10 characters
        - <=4 words
    """
    punctuation_remover = str.maketrans('', '', string.punctuation)
    translated = dialogue.translate(punctuation_remover)
    words = translated.split(" ")

    return len(translated) <= 10 or len(words) <= 4

def try_generate_wav(generate_wav: Callable[[str], Tensor], sample_rate: int, dialogue: str, dialogue_prefix: str) -> Optional[Tensor]:
    if not is_short_dialogue(dialogue):
        return generate_wav(dialogue)
    
    print("Generating using prefix method, since the line is short")
    prefixed_dialogue = f"{dialogue_prefix} {dialogue}"
    prospective_dialogue = generate_wav(prefixed_dialogue)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tempGen:
        tmpfile_name = tempGen.name
        ta.save(tmpfile_name, prospective_dialogue, sample_rate)
        prefix_end_time = get_prefix_phrase_end(tmpfile_name, whisper_model, dialogue_prefix, confidence_threshold=0.0)
        if prefix_end_time is None:
            return None
        
    print("detected prefix end time:", prefix_end_time)
    print("analyzed total duration:", prospective_dialogue.shape[1]/sample_rate)
    generated_dialogue = trim_prefix_time_tensor(prospective_dialogue, sample_rate, prefix_end_time)
    if generated_dialogue is None:
        return None
    
    # use vad to trim off silence and trailing phonemes from the prefix
    # this can sometimes be too aggressive, so we'll gradually decrease the trigger level from the default until we get something acceptable (non-zero)
    trigger_level = 2.0
    step = 0.5
    vad_dialogue = None # prevent potentially unbound warning
    while trigger_level > 1.0:
        vad_dialogue = ta.functional.vad(generated_dialogue[0], sample_rate, trigger_level=trigger_level, trigger_time=0.1)
        if not vad_dialogue.shape[1]/sample_rate < 0.2:
            break
        trigger_level -= step
        print(f"vad failure. Trigger level reduced to {trigger_level}")
    
    # if all else fails, forget vad and just trim off an extra couple hundred milliseconds
    if trigger_level <= 1.0:
        print("No acceptable result even after vad trigger level sweep. Falling back to naive prefix tail removal.")
        vad_dialogue = trim_prefix_time_tensor(prospective_dialogue, sample_rate, prefix_end_time+0.2)
        if vad_dialogue is None:
            return None
        vad_dialogue,_ = vad_dialogue
        if vad_dialogue.shape[1]/sample_rate < 0.2:
            print("Total generation failure. Even naive method failed.")
            return None
        
    if vad_dialogue is not None:
        # apply fade so that the first sample is always 0 and there are no discontinuities to cause popping
        fade_samples = int(0.1 * sample_rate)
        transform = ta.transforms.Fade(
            fade_in_len=fade_samples,
            fade_out_len=0,
            fade_shape="linear"
        )
        vad_dialogue = transform(vad_dialogue)

    return vad_dialogue
        

print("Loading whisper model...")
whisper_model = wt.load_model("medium", whisper_device)
print("Loading chatterbox model...")
def load_model():
    if(args.l is not None and args.l):
        model = ChatterboxMultilingualTTS.from_pretrained(device=cb_device)
        def generate_wav(dialogue, ref_audio, exaggeration, cfg_weight, temperature):
            return model.generate(dialogue, language_id=args.l, audio_prompt_path=ref_audio, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
        if(args.l not in model.get_supported_languages().keys()):
            print(args.l, "is not a supported language. Supported languages are:")
            print(model.get_supported_languages())
            exit()
        else:
            print("Using language:", model.get_supported_languages()[args.l])
    else:
        model = ChatterboxTTS.from_pretrained(device=cb_device)
        def generate_wav(dialogue, ref_audio, exaggeration, cfg_weight, temperature):
            return model.generate(dialogue, audio_prompt_path=ref_audio, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
        
    return model, generate_wav

model, generate_wav = load_model()

print("Models loaded")

HOST = "0.0.0.0"
PORT = args.p

def recv_line(sock):
    """
    Receives data from a socket until a newline character is encountered.
    """
    buffer = b""
    while True:
        data = sock.recv(1)  # Read one byte at a time
        if not data:  # Connection closed
            return ""
        buffer += data
        if b"\n" in buffer:
            line, _, buffer = buffer.partition(b"\n")
            return line.decode('utf-8')  # Decode the line to a string

def recv_all_fixed_size(sock: socket.socket, expected_len: int):
        data = b''
        while len(data) < expected_len:
            packet = sock.recv(expected_len - len(data))
            if not packet:
                raise ConnectionError(f"connection closed while reading {expected_len} bytes")
            data += packet
        return data

numbers_re = re.compile(r"(\d[\d,]*)")
def process_spoken_numbers(dialogue: str) -> str:
    pieces = numbers_re.split(dialogue)
    def map_fn(piece: str) -> str:
        if(not numbers_re.search(piece)):
            return piece
        piece = piece.replace(",", "")
        return num2words.num2words(int(piece))
    pieces = map(map_fn, pieces)
    return "".join(pieces)

def run_substitutions(dialogue: str) -> str:
    for k,v in subMap.items():
        if re.search(k, dialogue, flags=re.IGNORECASE) is not None:
            print(f"replacing '{k}' with '{v}'")
            dialogue = re.sub(k,v,dialogue,flags=re.IGNORECASE)

    return dialogue


def handle_request(conn: socket.socket, generate_wav):
    line = recv_line(conn)
    ref_file, exaggeration, cfg_weight, temperature = line.strip().split("|")
    exaggeration, cfg_weight, temperature = map(float, (exaggeration, cfg_weight, temperature))
    makedirs("ref_files", exist_ok=True)
    ref_file = f"ref_files/{ref_file}"

    print(ref_file, exaggeration, cfg_weight, temperature)

    if(not path.exists(ref_file)):
        print("requesting reference file")
        conn.sendall("SEND_REF\n".encode())
        length = int(recv_line(conn))
        ref_audio = recv_all_fixed_size(conn, length)
        with open(ref_file, 'wb') as f:
            f.write(ref_audio)
        print("got reference file")
    else:
        conn.sendall("NO_SEND_REF\n".encode())
    
    dialogue = recv_line(conn)
    print("generating dialogue:", dialogue)
    if(args.convert_numbers):
        new_dialogue = process_spoken_numbers(dialogue)
        if(new_dialogue != dialogue):
            print("Replacing numbers. New Dialogue:")
            print(new_dialogue)
            dialogue = new_dialogue
    
    dialogue = run_substitutions(dialogue)

    def generate_wav_lambda(dialogue: str) -> Tensor:
        return generate_wav(dialogue, ref_file, exaggeration, cfg_weight, temperature)
    
    generation_attempt: Optional[Tensor] = None
    attempts = 0
    MAX_RETRIES = 4
    # random offset to ensure that all prefixes are used equally
    rand = randint(0, len(SHORT_DIALOGUE_PREFIXES))
    while generation_attempt is None and attempts <= MAX_RETRIES:
        if attempts > 0:
            print(f"Retrying generation. ({attempts}/{MAX_RETRIES})")
        generation_attempt = try_generate_wav(generate_wav_lambda, model.sr, dialogue, SHORT_DIALOGUE_PREFIXES[(attempts+rand) % len(SHORT_DIALOGUE_PREFIXES)])
        attempts += 1
    
    wav = generation_attempt
    if wav == None:
        print("Failed to generate dialogue using prefix method. Sending back whatever comes out of chatterbox")
        wav = generate_wav_lambda(dialogue)

    print("Tensor shape:", wav.shape)
    with io.BytesIO() as tmpWav:
        ta.save(tmpWav, wav, model.sr, format="wav", bits_per_sample=16)
        tmpWav.flush()
        tmpWav.seek(0)
        with open("last_output.wav", "wb") as f:
            byt = tmpWav.read()
            conn.sendall(byt)
            f.write(byt)
        

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening for connections on {HOST}:{PORT}. Use ctrl+c to stop.")
    generations = 0
    while True:
        try:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                if(generations > MODEL_REFRESH_INTERVAL and MODEL_REFRESH_INTERVAL > 0):
                    print("Reloading model to refresh performance")
                    del model #type: ignore
                    del generate_wav #type: ignore
                    gc.collect()
                    model, generate_wav = load_model()
                    generations = 0
                handle_request(conn, generate_wav) #type: ignore
                generations += 1
        except KeyboardInterrupt:
            print("exiting")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"The server has not stopped listening and will accept more requests")
