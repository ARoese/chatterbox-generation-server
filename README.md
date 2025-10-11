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