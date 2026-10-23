import os

# Use Homebrew's espeak-ng everywhere (phoneme scoring and Kokoro TTS) so both see the same
# phoneme inventory. Kokoro's bundled espeak can't find its data when the path contains spaces.
_BREW = "/opt/homebrew"
if os.path.exists(f"{_BREW}/lib/libespeak-ng.dylib"):
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", f"{_BREW}/lib/libespeak-ng.dylib")
    os.environ.setdefault("ESPEAK_DATA_PATH", f"{_BREW}/share/espeak-ng-data")
    try:
        import espeakng_loader

        espeakng_loader.get_library_path = lambda: os.environ["PHONEMIZER_ESPEAK_LIBRARY"]
        espeakng_loader.get_data_path = lambda: os.environ["ESPEAK_DATA_PATH"]
    except ImportError:
        pass
