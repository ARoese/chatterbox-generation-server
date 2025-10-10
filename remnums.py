# remove wav files with raw spoken numbers

import argparse
import os
import os.path as path
import re

parser = argparse.ArgumentParser()
parser.add_argument("wavs_dir")

args = parser.parse_args()

wavs_dir = args.wavs_dir

files = map(lambda x: path.join(wavs_dir, x), os.listdir(wavs_dir))
files = filter(lambda x: x.endswith(".wav"), files)
files = list(files)

number_re = re.compile(r"(\d[\d,]*)")
parens_re = re.compile(r"\(.*\)")
def has_number(x: str) -> bool:
    x = parens_re.sub("", x)
    if number_re.search(x):
        return True 
    return False

number_files = list(filter(has_number, files))
for f in number_files:
    os.remove(f)