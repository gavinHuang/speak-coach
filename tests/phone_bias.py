import os, sys
sys.path.insert(0, ".")
import app
from app import tts
from app.audio_asr import load16k
from app import phonemes
phonemes.PHONE_MODEL = sys.argv[1]
s = phonemes.PhonemeScorer()
for t in ["I sink dis is de right ting to do.", "I think this is the right thing to do.", "sink", "think",
          "I am very happy to see you.", "I am wery happy to zee you.", "She lives in a big house.", "She leaves in a beeg house."]:
    k, p = tts.synth(t, "work", gender="male")
    lp = s.logprobs(load16k(str(p))); out = []; prev = None
    for i in lp.argmax(1):
        if i != prev and i != s.blank: out.append(s.id2tok[int(i)])
        prev = i
    print(f"{t:40} {' '.join(out)}")
