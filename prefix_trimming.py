import whisper_timestamped as wt
import torchaudio as ta
import torchaudio.functional as F
from torch import tensor, Tensor
import json
import string
from typing import Optional, Tuple

def get_prefix_phrase_end(audio_path: str, model, prefix_phrase: str, language="en", confidence_threshold=0.5) -> Optional[float]:
    """
    Detects and returns the end (in seconds) of the prefix_phrase spoken in the given audio. 

    It is recommended that a prefix phrase that is very easy to detect be used. 
    All words in the prefix phrase MUST be in the whisper dictionary, or this function will always return None

    Returns None if:

        - Any part of the prefix phrase is not detected
        - The confidence of the detection of a word in the prefix phrase is below the confidence_threshold
    """
    prefix_phrase = prefix_phrase.lower()
    audio = wt.load_audio(audio_path)
    result = wt.transcribe(model, audio, language=language)
    translator = str.maketrans('', '', string.punctuation)
    expected_words = [w.translate(translator) for w in prefix_phrase.split(" ")]

    try:
        words = result["segments"][0]["words"]
        words_flat: list[str] = [word["text"].translate(translator).lower() for word in words]
        words_confidence: list[float] = [word["confidence"] for word in words]

        any_mismatch = any(map(lambda t: t[0] != t[1],zip(words_flat, expected_words)))
        any_low_confidence = any(map(lambda c: c < confidence_threshold, words_confidence[:len(expected_words)]))
        #print(words)
        
        if len(words_flat) < len(expected_words) or any_mismatch or any_low_confidence:
            return None
        
        return words[len(expected_words)-1]["end"]

    except (KeyError, IndexError):
        return None
    
def trim_prefix_time(audio_path: str, prefix_end_time: float) -> Optional[Tuple[Tensor, int]]:
    audio, sr = ta.load(audio_path)
    return trim_prefix_time_tensor(audio, sr, prefix_end_time)

def trim_prefix_time_tensor(audio: Tensor, sr: int, prefix_end_time: float) -> Optional[Tuple[Tensor, int]]:
    start_sample = int(sr*prefix_end_time)
    if audio.shape[1] <= start_sample:
        return None
    suffix = audio[:, start_sample:]
    return (suffix, sr)
    

if __name__ == "__main__":
    #audio = wt.load_audio("last_output.wav")
    model = wt.load_model("small")

    #result = wt.transcribe(model, audio, language="en")
    #audio_ta, sample_rate = ta.load("last_output.wav")

    AUDIO_PATH = "outputs/voiceToy_last_output.wav"
    PREFIX_PHRASE = "Green eggs and ham."
    end_timestamp = get_prefix_phrase_end(AUDIO_PATH, model, prefix_phrase=PREFIX_PHRASE)
    if end_timestamp is None:
        print("could not locate end timestamp")
        exit()
    trimmed = trim_prefix_time(AUDIO_PATH, end_timestamp)
    if trimmed is None:
        print("could not trim off prefix")
        exit()
    trimmed, sr = trimmed
    trimmed = F.vad(trimmed, sr) # trim off preceeding silence and/or trailing part of the final prefix word
    print("end timestamp:", end_timestamp)
    ta.save("trimmed_audio.wav", trimmed, sample_rate=sr)