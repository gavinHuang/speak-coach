# Speak Coach: a local spoken-English evaluator

Everything runs on your Mac (Apple Silicon). After the first model download, no audio or text leaves the machine.

```
./run.sh        # then open http://127.0.0.1:8765
```

## What it measures

| Area | How | What you see |
|---|---|---|
| **Sounds (phonetics)** | espeak-ng gives the expected IPA phonemes in context. A wav2vec2 phoneme recogniser (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`) is CTC-force-aligned to them, and each phone is scored by how clearly the model hears it. | A score for every word and phone, plus what it heard instead (e.g. /θ/ → /s/) and an articulation tip |
| **Intelligibility** | Whisper large-v3-turbo (MLX) transcribes you without knowing the target text | Words a listener would likely misunderstand ("bit" heard as "beat"), and an accuracy % |
| **Word stress** | Pitch, loudness and length of each vowel compared with the dictionary's stressed syllable | Words underlined when the stress seems to be on the wrong syllable (experimental) |
| **Intonation / tone** | Praat (parselmouth) pitch track in semitones | Pitch variation (monotone or lively), sentence-final rise/fall vs. expected, and your pitch curve over the reference voice's |
| **Sentence stress & rhythm** | Each word's prominence (pitch, loudness, duration) compared with the reference voice | Which words the reference emphasises that you didn't, emphasis match %, pace ratio |
| **Fluency** | Word timings plus silence detection | Words/min, articulation rate, pauses (and pauses in the middle of a phrase), mean length of run, fillers, discourse markers, repetitions |
| **Coherence, vocabulary, grammar, register fit** | Local LLM (Qwen3-8B, MLX), given the transcript and all the measurements | IELTS-style bands, CEFR estimate, specific issues with fixes, corrected version, "how a native would say it", voice/tone coaching, drills |

## The three situations (registers)

**Work**, **Casual** and **Intimate** each have target ranges (pace, pitch variation, fillers), coaching notes, and a reference voice preset (Kokoro TTS voice and speed). "This line in 3 situations" rewrites any sentence for each register, and you can play each version.

## Real-speech library (shadow real people)

The **Real-speech library** page holds clips of real people talking, to listen to and shadow. Click **Shadow** on any clip and you practise that exact line, and your delivery is compared with the real speaker's (pitch curve, word emphasis, pace) instead of a TTS voice. After every practice, the results also suggest "real people saying similar things".

Sources you can add from the page (or the CLI), with the number of clips you want:

| Source | What it is | Situations |
|---|---|---|
| Friends (MELD) | 13.7k lines from 10 seasons of the sitcom, with emotion labels | casual, intimate |
| AMI meetings | Real design-team meetings | work |
| Earnings calls | Executives and analysts, presentations and Q&A | work |
| European Parliament (VoxPopuli) | Formal speeches | work |
| Expresso | Lines in many tones: whisper, happy, sad, laughing… | intimate, casual |
| EdAcc | Friends chatting, 40+ accents of English | casual |
| **Your own media** | Any movie, show, podcast or interview: a file path (uses a matching or embedded `.srt`, else Whisper) or a URL (yt-dlp) | whatever it contains |

Every candidate line is checked before it is kept:
- **Whisper** re-transcribes it and drops it if the transcript disagrees with the dataset or subtitle.
- **The phoneme aligner** must find clean speech, which rejects laugh tracks, music and people talking over each other.
- **Silence vs speech:** it must clear a minimum noise level.
- **Measurements:** pace, pitch variation and the sentence-final contour are recorded.

The local LLM then labels each clip with a situation (work, casual or intimate), a tone, a purpose (apology, request, joke…) and how useful it is to shadow. You can filter by all of these, plus voice, accent, length and pace, or search for words ("sorry", "deadline", "miss you").

Downloads are cached in `~/.cache/huggingface`. Clips and the index live in `data/library/` (or in `$SPEAKCOACH_LIBRARY`). Unfinished jobs resume when the server restarts.

```
uv run python -m app.library ingest meld --max 3000      # when the web app is not running
uv run python -m app.library import "~/Movies/Before Sunrise (1995).mkv"
uv run python -m app.library stats
```

Only import media you have the right to use. The datasets keep their own licences (see the Library page); several are non-commercial or research-only, which covers personal study.

## Modes
- **Read aloud**: practise a given line. Your delivery is compared to the reference voice for that register.
- **Free talk**: answer a speaking task (or your own topic) for 30–90 s. Content, grammar and coherence are judged too.

History (with a trend chart) is saved in `data/sessions/`.

## Configuration (environment variables)
- `LLM_MODEL` (default `mlx-community/Qwen3-8B-4bit`). Any mlx-lm chat model works; a bigger one (e.g. a 14B 4-bit) gives better feedback if you have the RAM.
- `WHISPER_MODEL` (default `mlx-community/whisper-large-v3-turbo`)
- `PHONEME_MODEL` (default `facebook/wav2vec2-xlsr-53-espeak-cv-ft`)
- `PORT` (default 8765)

## Limits, honestly
- Scores are calibrated heuristics, not certified exam scores. Use them to track your own progress and find patterns.
- The expected pronunciation is espeak's US or UK standard, so a different but valid accent can lose a few points on some vowels.
- The reference voices are TTS. Kokoro sounds natural, but it doesn't always rise on yes/no questions. Treat its tone as one good model, not the only correct one.
- Word-stress detection is experimental and only flags clear cases.
- Library labels come from an 8B local model, so some will be off (a neutral line tagged "affectionate"). Usefulness ranking and filters still work well at scale. Set `LLM_MODEL` to a larger model for better labels.
- Ingestion runs at roughly 40–50 clean clips a minute, and 35–60% of candidates pass the checks (fewer for films with heavy music).
