# Chatterbox local generation server

This is a collection of tools for running chatterbox-tts locally for the purpose of batch-generating voicelines.

## Components:

- voiceToy.py: REPL for generating voicelines   
  - interactively with different generation params.
  - Input a line of text and press enter to hear that line spoken through your default speakers. 
  - Enter a line in the form `[exaggeration cfg_weight temp]` (e.g. `[0.5 0.5 0.8]`) to set the generation parameters for future generation.
  - Pass `-l LANG` to use the multilingual model with the specified language set. (e.g. `-l ru`) Entering an invalid option will print the valid options.

- chatterboxGenServer.py: Batch voiceline generation server
  - Listens for requests for voiceline generation using a basic (primitive) protocol
    - Retrieves reference audio file name and generation params
    - requests the reference audio file be sent by the client if it doesn't already exist locally
    - replies with the raw wav data
  - supports generation using the multilingual model, but the language is not configurable per-request. Use the `-l LANG` parameter similar to voiceToy.py.
  - use `-p PORT` to configure what port it listens on
  - use `--convert_numbers` to convert numbers like "10" to the word equivalent like "ten". Sometimes, the models struggle to come up with the right pronunciation of the numeric strings, or when using the multilingual model for accents, will speak the numbers in the wrong language. This fixes that.
  - use `-m` to provide a mappings file. This is meant for replacing certain words with phonetic equivalents to aid with pronunciation. For example, you could add a line `gif||jif` that makes "gif" be pronounced explicitly with a j sound. Replacements are case-insensitive

## Environment setup
1. Make a clean python venv. I use pyenv with the virtualenv plugin for this.
  - **IMPORTANT:** Your environment must be set to use **python 3.9.21** or similar. Newer python versions remove depricated features that chatterbox-tts relies on.
2. activate your venv
3. `pip install -r requirements.txt`

## Running with docker
run the docker container using the following commands:  
1. `docker compose build`  
2. `docker compose run --remove-orphans chatterbox-gen-server`

This will drop you into a shell with the correct environment to run chatterboxGenServer.py and voiceToy.py. Port 9005 is forwarded by default, and all system gpus are made available to the container if possible. The models will be downloaded initially, but will be cached in a docker volume so they are availalbe on future runs. To change which port is forwarded, set it in `docker-compose.yml`.

## Other notes

- This server is written specifically to serve my fork of [AbsolutePhoenix's DBVO pack builder](https://github.com/AbsolutePhoenix/DBVO_Pack_Builder) which uses it to generate dialogue. You can download that from [here](https://github.com/ARoese/DBVO_Pack_Builder)

- You should avoid using genericly-named files as references for `chatterboxGenServer.py`. The references are saved and cached server-side using the leaf file name only in `ref_files`. For example, `character1/sample.wav` and `character2/sample.wav` are considered to be the same file. Subsequent generations offering `character2/sample.wav`, will use `character1/sample.wav` if it was sent first. This can be resolved client-side by offering a file named with its hash, but that is not currently implemented. 

## Implementation notes
Chatterbox struggles to generate short dialogue lines. Often, they devolve into groaning, consist of rapid and random phonemes, or are otherwise broken. This problem is rectified simply by never generating a short voiceline.

Here, a "short" voiceline is one that consists of 4 or fewer words, and is 10 or less characters (excluding punctuation)

When a short voiceline is requested for generation, it is handled specially. First, a prefix phrase is selected and added so that the dialogue that actually gets generated is suitably long enough. For example:
Requested: George. (too short. Will just be blabbering)
Generated: Green eggs and ham. George. (long enough to be generated normally)

Having a random phrase prefixing the spoken dialogue is undesirable, so the prefix phrase is detected using whisper_timestamped. This takes the generated audio and produces a transcription of what was said to sub-second precision. First, the presence of the prefix phrase is verified--if it isn't detected, then generation is repeated until it is. Then, the generation is cut starting from the end of the final word of the prefix phrase onwards. (the prefix phrase is cut out of the audio) To prevent breaths and the final portions of phonemes from the prefix phrase from being audible in the resulting dialogue, the audio clip is finally run through torchaudio's functional VAD transform. This removes subtle noises up until the beginning of the next word. (the word that should actually have been spoken) So far, this VAD step is actually the worst part of the process. The VAD step is repeated multiple times with different parameters if the result is bad, but some poor dialogue still squeezes through. 

Most common issues right now:
- The first letter/phoneme of a dialogue is cut off. Probably because the VAD does not have a quick enough attack in those cases.
- A subtle popping sound at the beginning of audio. This is probably trailing noise from the prefix phrase sticking around, or the cut audio not starting on a zero-crossing and causing impulse artifacts. (This is mostly solved by applying a 100ms linear fade-in to the resultant clip. You can probably still hear it if you focus.)

tl;dr: Short dialogue (bad) -> long dialogue with extra filler words (fine) -> cut out the filler words using a separate model -> short dialogue audio (woohoo!)