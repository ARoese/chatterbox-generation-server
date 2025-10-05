import argparse
import sys
import socket
import os.path as path
from os import makedirs, remove
import argparse
import io

import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Generate voicelines using reference audio")
parser.add_argument("-l", default=None)
parser.add_argument("-p", type=int, default=9005)
args = parser.parse_args()

print("Loading model...")
if(args.l is not None and args.l):
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    def generate_wav(dialogue, ref_audio, exaggeration, cfg_weight, temperature):
        return model.generate(dialogue, language_id=args.l, audio_prompt_path=ref_audio, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    if(args.l not in model.get_supported_languages().keys()):
        print(args.l, "is not a supported language. Supported languages are:")
        print(model.get_supported_languages())
        exit()
    else:
        print("Using language:", model.get_supported_languages()[args.l])
else:
    model = ChatterboxTTS.from_pretrained(device=device)
    def generate_wav(dialogue, ref_audio, exaggeration, cfg_weight, temperature):
        return model.generate(dialogue, audio_prompt_path=ref_audio, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)

print("Model loaded")


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
                return None  # Connection closed
            data += packet
        return data

def handle_request(conn: socket.socket):
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
    wav = generate_wav(dialogue, ref_file, exaggeration, cfg_weight, temperature)

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
    while True:
        try:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                handle_request(conn)
        except KeyboardInterrupt:
            print("exiting")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"The server has not stopped listening and will accept more requests")
