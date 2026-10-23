import json, sys, time
sys.path.insert(0, ".")
from app import tts, analysis
from app.audio_asr import load16k
target = "I am very happy to see you. She lives in a big house. Can we talk about it tomorrow?"
t=time.time(); k, p = tts.synth(target, "work"); print("tts", time.time()-t)
_, pbad = tts.synth("I am wery happy to zee you. She leaves in a beeg house. Can we talk about it tomorrow?", "work", gender="male")
ref = load16k(str(p))
for name, path in [("good", p), ("bad", pbad)]:
    t=time.time()
    r = analysis.analyze(load16k(str(path)), "read", "work", "us", target, ref_audio=ref)
    print("==", name, round(time.time()-t,1), "s", r["transcript"])
    print(r["scores"])
    for w in r["words"]:
        print(f'  {w["text"]:10} {w["score"]} /{w["ipa"]}/ heard={w["heard"]} asr={w.get("asr","")} {w.get("stress_issue","")}',
              [(p["p"],p["heard"],p["score"]) for p in w["phones"] if p["score"]<60])
    print(json.dumps({k:r[k] for k in ["fluency","accuracy","register_checks","sound_problems"]}, ensure_ascii=False)[:1500])
    print({k:v for k,v in r["intonation"].items()})
    print({k:v for k,v in (r["reference"] or {}).items() if k not in ("contour",)})
