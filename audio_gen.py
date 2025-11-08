import requests
import json
import os
import base64
import argparse
import time
import tempfile
import re
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or GOOGLE_API_KEY
BASE_URL = "https://texttospeech.googleapis.com/v1"

VOICE_NAME = "en-us-Chirp3-HD-Algenib"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_audio_extension(audio_format):
    """Get file extension for audio format."""
    extension_map = {
        "wav": ".wav",
        "mp3": ".mp3",
        "ogg": ".ogg",
        "mulaw": ".ulaw",
        "alaw": ".alaw",
    }
    return extension_map.get(audio_format.lower(), ".wav")

def synthesize_speech(text=None, markup=None, speaking_rate=1.0, pace=None, voice_name=VOICE_NAME, audio_format="wav"):
    """Synthesize speech using REST API with API key.
    
    Args:
        text: Plain text input
        markup: Markup input with pause tags
        speaking_rate: Speech rate (0.25 to 2.0)
        pace: Pace control for Chirp3-HD - "slow", "fast", "x-slow" (only for markup/text)
        voice_name: Voice name (default: en-us-Chirp3-HD-Leda)
        audio_format: Output format - "wav", "mp3", "ogg", "mulaw", "alaw" (default: "wav")
                     Note: MP3 uses default bitrate (API doesn't support bitrate control)
    
    Returns:
        dict: Response containing audioContent (base64 encoded)
    """
    
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Please set GOOGLE_API_KEY environment variable.")
    
    # Map format strings to API encoding values
    format_map = {
        "wav": "LINEAR16",
        "mp3": "MP3",
        "ogg": "OGG_OPUS",
        "mulaw": "MULAW",
        "alaw": "ALAW",
    }
    
    audio_encoding = format_map.get(audio_format.lower())
    if not audio_encoding:
        raise ValueError(f"Unsupported audio format: {audio_format}. Supported: {list(format_map.keys())}")
    
    url = f"{BASE_URL}/text:synthesize?key={GOOGLE_API_KEY}"
    
    input_data = {}
    if markup:
        input_data["markup"] = markup
    elif text:
        input_data["text"] = text
    else:
        raise ValueError("Must provide text or markup")
    
    voice_config = {
        "languageCode": "en-US",
        "name": voice_name,
    }
    
    effective_rate = speaking_rate
    if pace:
        pace_rate_map = {
            "slow": 0.9,
            "x-slow": 0.7,
            "fast": 1.5,
        }
        if pace in pace_rate_map:
            effective_rate = pace_rate_map[pace]
    
    payload = {
        "input": input_data,
        "voice": voice_config,
        "audioConfig": {
            "audioEncoding": audio_encoding,
            "speakingRate": effective_rate,
        }
    }
    
    response = requests.post(url, json=payload)
    
    if not response.ok:
        print(f"Error response: {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()
    
    return response.json()

def save_audio(result, output_file):
    """Save audio content from API response to file."""
    with open(output_file, "wb") as out:
        out.write(base64.b64decode(result["audioContent"]))
    
    audio_bytes = len(base64.b64decode(result["audioContent"]))
    print(f"Audio saved to {output_file}")
    print(f"Audio length: {audio_bytes} bytes ({audio_bytes / 1024:.2f} KB)")


def get_epub_title(epub_path):
    """Extract book title from EPUB file.
    
    Args:
        epub_path: Path to EPUB file
        
    Returns:
        str: Book title, or None if not found
    """
    book = epub.read_epub(epub_path)
    title = book.get_metadata('DC', 'title')
    if title and len(title) > 0:
        return title[0][0]
    return None


def parse_epub(epub_path):
    """Extract all text content from EPUB file.
    
    Args:
        epub_path: Path to EPUB file
        
    Returns:
        str: Combined text content from all chapters
    """
    print(f"Parsing EPUB: {epub_path}")
    book = epub.read_epub(epub_path)
    text_parts = []
    
    exclude_patterns = ['toc.', 'nav.', '-toc', '-nav', 'contents.xhtml', 'index.xhtml']
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            item_name = item.get_name().lower()
            item_id = item.get_id().lower() if item.get_id() else ''
            
            print(f"Processing: {item.get_name()} (id: {item_id})")
            
            if any(pattern in item_name or pattern in item_id for pattern in exclude_patterns):
                print(f"  -> Skipping (matched exclusion pattern)")
                continue
            
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            if 'table of contents' in text.lower()[:200] or 'contents' in text.lower()[:100]:
                print(f"  -> Skipping (detected TOC content)")
                continue
            
            if text:
                print(f"  -> Including ({len(text)} chars)")
                text_parts.append(text)
    
    combined_text = ' '.join(text_parts)
    print(f"Extracted {len(combined_text)} characters from EPUB")
    return combined_text


def clean_text_with_gemini(text, model_name="gemini-2.5-flash-lite"):
    """Clean text using Gemini LLM to remove non-readable content.
    
    Args:
        text: Raw text to clean
        model_name: Gemini model to use (default: gemini-2.5-flash-lite)
        
    Returns:
        str: Cleaned text
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please set GEMINI_API_KEY environment variable.")
    
    prompt = """
You are a helpful assistant that pre-processes text for an audiobook.
Your task is to remove all content that should not be read aloud in an audiobook, 
and fix broken words coming from the OCR process.

Remove all content like:
- Table of contents entries
- Page numbers
- Image descriptions and alt text
- Links and URLs
- Footnote numbers and references
- Footnotes
- Table of contents entries
- Headers and footers
- Any navigation elements
- Copyright notices
- Ebook ISBN
- Publisher information
- Author information


Keep only the actual readable text content that should be narrated. Return only the cleaned text, no explanations."""
    
    max_chunk_tokens = 5000
    chunk_size = max_chunk_tokens * 3
    
    if len(text) <= chunk_size:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt + "\n\nText to clean:\n" + text)
        time.sleep(4)
        cleaned = response.text.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        return cleaned
    
    print(f"Text is large, cleaning in chunks...")
    cleaned_chunks = []
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    model = genai.GenerativeModel(model_name)
    for i, chunk in enumerate(chunks):
        print(f"Cleaning chunk {i+1}/{len(chunks)}...")
        response = model.generate_content(prompt + "\n\nText to clean:\n" + chunk)
        cleaned = response.text.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        cleaned_chunks.append(cleaned)
        if i < len(chunks) - 1:
            time.sleep(4)
    
    return ' '.join(cleaned_chunks)


def text_to_markup_with_gemini(text, model_name="gemini-2.5-flash-lite"):
    """Convert text to markup format with natural pauses using Gemini.
    
    Args:
        text: Plain text to convert
        model_name: Gemini model to use
        
    Returns:
        str: Markup formatted text with pause tags
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please set GEMINI_API_KEY environment variable.")
    
    prompt = """Convert this text into natural audiobook narration markup. Follow these rules EXACTLY:

1. Remove all quotes (", ', ", "), parentheses (), and braces {} from the text
2. Replace em-dashes and en-dashes with spaces
3. Add pause tags in EXACTLY this format - SINGLE square brackets ONLY:
   - [pause short] after sentences or commas for brief pauses
   - [pause] between paragraphs or scene changes
   - [pause long] after chapter headings or major section breaks

4. CRITICAL: Use SINGLE brackets [pause long] NOT double brackets [[pause long]]
5. CRITICAL: Pause tags MUST be in square brackets like [pause] NOT "pause"
6. Use pauses to create natural rhythm:
   - After questions: [pause]
   - Before dramatic reveals: [pause long]
   - Between dialogue exchanges: [pause short]
   - After exclamations: [pause short]
   - After chapter titles: [pause long]

7. Do NOT write the words "pause short" or "pause" as plain text - they MUST be in [brackets]
8. Keep the text clean and readable for speech synthesis
9. Preserve all actual content - only remove punctuation that would be read aloud
10. Return ONLY the formatted text with pause tags, no explanations

Example input: Chapter 1. "Hello," she said (quietly).
Example output: Chapter 1 [pause long] Hello, she said quietly. [pause]

WRONG: Chapter 1 pause long Hello, she said quietly pause
WRONG: Chapter 1 [[pause long]] Hello, she said quietly [[pause]]
RIGHT: Chapter 1 [pause long] Hello, she said quietly. [pause]

Text to convert:
"""
    
    max_chunk_tokens = 5000
    chunk_size = max_chunk_tokens * 3
    
    if len(text) <= chunk_size:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt + text)
        time.sleep(4)
        markup = response.text.strip()
        if markup.startswith('"') and markup.endswith('"'):
            markup = markup[1:-1]
        #markup = fix_pause_tags(markup)
        print(markup)
        return markup
    
    print(f"Text is large, converting in chunks...")
    markup_chunks = []
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    model = genai.GenerativeModel(model_name)
    for i, chunk in enumerate(chunks):
        print(f"Converting chunk {i+1}/{len(chunks)}...")
        response = model.generate_content(prompt + chunk)
        markup = response.text.strip()
        if markup.startswith('"') and markup.endswith('"'):
            markup = markup[1:-1]
        #markup = fix_pause_tags(markup)
        markup_chunks.append(markup)
        if i < len(chunks) - 1:
            time.sleep(4)
    
    return ' '.join(markup_chunks)


def chunk_content(formatted_text, max_chars=4000):
    """Split formatted text into fixed-size chunks.
    
    Args:
        formatted_text: Markup text to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        list: List of chunk strings
    """
    sentences = re.split(r'(\[pause (?:short|long)\])', formatted_text)
    sentences = [s for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for part in sentences:
        part_size = len(part)
        test_size = current_chunk_size + part_size
        
        if test_size > max_chars and current_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = [part]
            current_chunk_size = part_size
        else:
            current_chunk.append(part)
            current_chunk_size += part_size
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks if chunks else [formatted_text]


def concatenate_audio_files(audio_files, output_file):
    """Concatenate multiple WAV files into a single file.
    
    Args:
        audio_files: List of WAV file paths
        output_file: Output file path
    """
    print(f"Concatenating {len(audio_files)} audio files...")
    
    if not audio_files:
        raise ValueError("No audio files to concatenate")
    
    with wave.open(audio_files[0], 'rb') as first_wav:
        params = first_wav.getparams()
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        frames = first_wav.readframes(nframes)
    
    for audio_file in audio_files[1:]:
        with wave.open(audio_file, 'rb') as wav_file:
            file_params = wav_file.getparams()
            file_nchannels, file_sampwidth, file_framerate, file_nframes, file_comptype, file_compname = file_params
            
            if (file_nchannels != nchannels or 
                file_sampwidth != sampwidth or 
                file_framerate != framerate or
                file_comptype != comptype or
                file_compname != compname):
                raise ValueError(f"Audio file {audio_file} has incompatible format: "
                               f"expected ({nchannels}, {sampwidth}, {framerate}, {comptype}, {compname}), "
                               f"got ({file_nchannels}, {file_sampwidth}, {file_framerate}, {file_comptype}, {file_compname})")
            frames += wav_file.readframes(file_nframes)
    
    with wave.open(output_file, 'wb') as out_wav:
        out_wav.setparams((nchannels, sampwidth, framerate, len(frames) // (sampwidth * nchannels), comptype, compname))
        out_wav.writeframes(frames)
    
    print(f"Combined audio saved to {output_file}")


def process_chunk(chunk_index, chunk, temp_dir, pace=None):
    """Process a single chunk and generate audio.
    
    Args:
        chunk_index: Index of the chunk
        chunk: Markup chunk text
        temp_dir: Temporary directory for audio files
        pace: Pace control (slow, fast, x-slow)
        
    Returns:
        tuple: (chunk_index, temp_file_path) or (chunk_index, None) if error
    """
    temp_file = os.path.join(temp_dir, f"chunk_{chunk_index:05d}.wav")
    try:
        print("chunk: ", chunk)
        result = synthesize_speech(markup=chunk, pace=pace)
        save_audio(result, temp_file)
        return (chunk_index, temp_file)
    except Exception as e:
        print(f"Error processing chunk {chunk_index + 1}: {e}")
        print(f"Chunk length: {len(chunk)} characters")
        print(f"Chunk content:\n{chunk}")
        return (chunk_index, None)


def convert_epub_to_audiobook(epub_path, output_file, pace=None, text_only=False):
    """Convert EPUB file to audiobook.
    
    Args:
        epub_path: Path to EPUB file
        output_file: Output audio file path
        pace: Pace control for markup - "slow", "fast", "x-slow" (default: None)
        use_gemini_markup: Use Gemini for intelligent markup conversion (default: True)
        text_only: Only run text preprocessing without audio generation (default: False)
    """
    print(f"Starting EPUB {'text preprocessing' if text_only else 'to audiobook conversion'}...")
    print(f"Input: {epub_path}")
    if not text_only:
        print(f"Output: {output_file}")
    if pace:
        print(f"Pace: {pace}")
    
    text = parse_epub(epub_path)
    
    print("Cleaning text with Gemini...")
    cleaned_text = clean_text_with_gemini(text)
    print(f"Cleaned text: {len(cleaned_text)} characters")

    print("Converting to markup with Gemini (natural narration)...")
    formatted_text = text_to_markup_with_gemini(cleaned_text)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    epub_basename = os.path.splitext(os.path.basename(epub_path))[0]
    debug_filename = f"{epub_basename}_{timestamp}_debug.txt"
    with open(debug_filename, 'w', encoding='utf-8') as f:
        f.write(formatted_text)
    print(f"Debug text saved to: {debug_filename}")
    
    if text_only:
        print("Text preprocessing complete! Skipping audio generation.")
        return
    
    print("Chunking for TTS...")
    chunks = chunk_content(formatted_text, max_chars=3500)
    print(f"Created {len(chunks)} chunks")
    
    temp_dir = tempfile.mkdtemp()
    
    print("Generating audio in parallel...")
    max_workers = min(10, len(chunks))
    temp_audio_files = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_chunk, i, chunk, temp_dir, pace): i
            for i, chunk in enumerate(chunks)
        }
        
        completed = 0
        for future in as_completed(futures):
            chunk_index, temp_file = future.result()
            completed += 1
            if temp_file:
                temp_audio_files[chunk_index] = temp_file
                print(f"Completed chunk {chunk_index + 1}/{len(chunks)}")
            else:
                print(f"Failed to process chunk {chunk_index + 1}")
                raise Exception(f"Failed to generate audio for chunk {chunk_index + 1}")
    
    temp_audio_files = [f for f in temp_audio_files if f is not None]
    temp_audio_files.sort()
    
    print("Concatenating audio files...")
    concatenate_audio_files(temp_audio_files, output_file)
    
    print("Cleaning up temporary files...")
    for temp_file in temp_audio_files:
        os.remove(temp_file)
    os.rmdir(temp_dir)
    
    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EPUB to audiobook")
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument("-o", "--output", default=None, help="Output audio file (default: book title)")
    parser.add_argument("-p", "--pace", choices=["slow", "fast", "x-slow"], default=None,
                       help="Reading speed: 'slow', 'fast', 'x-slow' (default: normal)")
    parser.add_argument("--text-only", action="store_true",
                       help="Only run text preprocessing without generating audio (saves debug file)")
    args = parser.parse_args()
    
    if args.output is None and not args.text_only:
        title = get_epub_title(args.epub_path)
        if title:
            sanitized_title = re.sub(r'[<>:"/\\|?*]', '', title).strip()
            if not sanitized_title:
                sanitized_title = "output"
            args.output = f"{sanitized_title}.wav"
        else:
            args.output = "output.wav"
    
    convert_epub_to_audiobook(args.epub_path, args.output, pace=args.pace, 
                             text_only=args.text_only)
    