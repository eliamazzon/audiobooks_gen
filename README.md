# Chirp 3 HD TTS Test

Minimal setup to test Google Cloud Chirp 3 HD TTS with SSML and voice controls.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up authentication:

**Option A: Service Account (original script)**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

**Option B: API Key (alternative script)**
```bash
export GOOGLE_TTS_API_KEY="your-api-key-here"
```
Or edit `test_chirp3_hd_api_key.py` and set `API_KEY` directly.

3. Run the test:

**With service account:**
```bash
python test_chirp3_hd.py
```

**With API key:**
```bash
python test_chirp3_hd_api_key.py
```

## Features Tested

- **SSML support**: `<speak>`, `<say-as>`, `<p>`, `<s>`, `<phoneme>`, `<sub>`
- **Long audio generation**: Using SSML paragraphs for structured narration
- **Pause control**: Using markup tags `[pause]`, `[pause short]`, `[pause long]`
- **Pace control**: Using `speaking_rate` (0.25 to 2.0)
- **Custom pronunciations**: Using IPA or X-SAMPA phonetic encoding

