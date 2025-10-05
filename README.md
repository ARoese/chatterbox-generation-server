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

## Environment setup
1. Make a clean python venv. I use pyenv with the virtualenv plugin for this.
  - **IMPORTANT:** Your environment must be set to use **python 3.9.21** or similar. Newer python versions remove depricated features that chatterbox-tts relies on.
2. activate your venv
3. `pip install -r requirements.txt`

## Other notes

- This server is written specifically to serve my fork of [AbsolutePhoenix's DBVO pack builder](https://github.com/AbsolutePhoenix/DBVO_Pack_Builder) which uses it to generate dialogue. (link to that fork coming soon)