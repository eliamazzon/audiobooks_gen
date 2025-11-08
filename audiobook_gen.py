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

def synthesize_speech(text=None, ssml=None, markup=None, speaking_rate=1.0, voice_name="en-us-Chirp3-HD-Aoede", audio_format="wav"):
    """Synthesize speech using REST API with API key.
    
    Args:
        text: Plain text input
        ssml: SSML formatted input
        markup: Markup input with pause tags
        speaking_rate: Speech rate (0.25 to 2.0)
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
    if ssml:
        input_data["ssml"] = ssml
    elif markup:
        input_data["markup"] = markup
    elif text:
        input_data["text"] = text
    else:
        raise ValueError("Must provide text, ssml, or markup")
    
    payload = {
        "input": input_data,
        "voice": {
            "languageCode": "en-US",
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": audio_encoding,
            "speakingRate": speaking_rate,
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
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            if text:
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
    
    prompt = """Remove all content that should not be read aloud in an audiobook:
- Page numbers
- Image descriptions and alt text
- Links and URLs
- Footnote numbers and references
- Table of contents entries
- Headers and footers
- Any navigation elements

Keep only the actual readable text content that should be narrated. Return only the cleaned text, no explanations."""
    
    max_chunk_tokens = 200000
    chunk_size = max_chunk_tokens * 3
    
    if len(text) <= chunk_size:
        print("Cleaning text with Gemini...")
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


def text_to_ssml(text):
    """Convert text to SSML format with natural pauses.
    
    Args:
        text: Plain text to convert
        
    Returns:
        str: SSML formatted text
    """
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) == 1:
        paragraphs = [text]
    
    ssml_parts = ['<speak>']
    max_sentence_length = 500
    
    for para_idx, para in enumerate(paragraphs):
        if not para.strip():
            continue
        para = para.strip()
        
        is_heading = (len(para) < 100 and 
                     (para.isupper() or 
                      re.match(r'^(Chapter|Part|Section|Book)\s+\d+', para, re.IGNORECASE) or
                      re.match(r'^(Chapter|Part|Section|Book)\s+[IVXLCDM]+', para, re.IGNORECASE) or
                      re.match(r'^\d+\.?\s*$', para)))
        
        ssml_parts.append('<p>')
        
        sentences = re.split(r'([.!?]+(?:\s+|$))', para)
        sentences = [s for s in sentences if s.strip()]
        
        current_sentence = ''
        for part in sentences:
            current_sentence += part
            if part.strip() and re.search(r'[.!?]\s*$', part.strip()):
                sentence_text = current_sentence.strip()
                if len(sentence_text) > max_sentence_length:
                    words = sentence_text.split()
                    current_words = []
                    for word in words:
                        test_sentence = ' '.join(current_words + [word])
                        if len(test_sentence) > max_sentence_length and current_words:
                            ssml_parts.append(f'<s>{" ".join(current_words)}</s>')
                            current_words = [word]
                        else:
                            current_words.append(word)
                    if current_words:
                        ssml_parts.append(f'<s>{" ".join(current_words)}</s>')
                else:
                    ssml_parts.append(f'<s>{sentence_text}</s>')
                current_sentence = ''
        
        if current_sentence.strip():
            sentence_text = current_sentence.strip()
            if len(sentence_text) > max_sentence_length:
                words = sentence_text.split()
                current_words = []
                for word in words:
                    test_sentence = ' '.join(current_words + [word])
                    if len(test_sentence) > max_sentence_length and current_words:
                        ssml_parts.append(f'<s>{" ".join(current_words)}</s>')
                        current_words = [word]
                    else:
                        current_words.append(word)
                if current_words:
                    ssml_parts.append(f'<s>{" ".join(current_words)}</s>')
            else:
                ssml_parts.append(f'<s>{sentence_text}</s>')
        
        ssml_parts.append('</p>')
        
        if is_heading and para_idx < len(paragraphs) - 1:
            ssml_parts.append('<p></p>')
    
    ssml_parts.append('</speak>')
    return '\n'.join(ssml_parts)


def chunk_content(ssml_text, max_chars=4000):
    """Split SSML text into fixed-size chunks preserving paragraph structure.
    
    Args:
        ssml_text: SSML text to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        list: List of SSML chunk strings (each wrapped in <speak> tags)
    """
    if not ssml_text.strip().startswith('<speak>'):
        ssml_text = '<speak>' + ssml_text + '</speak>'
    
    inner_content = ssml_text.replace('<speak>', '').replace('</speak>', '').strip()
    
    paragraphs = re.findall(r'<p>.*?</p>', inner_content, re.DOTALL)
    
    if not paragraphs:
        return [ssml_text]
    
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        test_size = current_chunk_size + para_size + 20
        
        if test_size > max_chars and current_chunk:
            chunk_content = ''.join(current_chunk)
            chunks.append('<speak>' + chunk_content + '</speak>')
            current_chunk = [para]
            current_chunk_size = para_size
        elif para_size > max_chars:
            if current_chunk:
                chunk_content = ''.join(current_chunk)
                chunks.append('<speak>' + chunk_content + '</speak>')
                current_chunk = []
                current_chunk_size = 0
            
            sentences = re.findall(r'<s>.*?</s>', para, re.DOTALL)
            if sentences:
                sub_chunk_sentences = []
                sub_chunk_size = 0
                for sentence in sentences:
                    sentence_size = len(sentence)
                    if sub_chunk_size + sentence_size + 20 > max_chars and sub_chunk_sentences:
                        sub_chunk_content = '<p>' + ''.join(sub_chunk_sentences) + '</p>'
                        chunks.append('<speak>' + sub_chunk_content + '</speak>')
                        sub_chunk_sentences = [sentence]
                        sub_chunk_size = sentence_size
                    else:
                        sub_chunk_sentences.append(sentence)
                        sub_chunk_size += sentence_size
                if sub_chunk_sentences:
                    sub_chunk_content = '<p>' + ''.join(sub_chunk_sentences) + '</p>'
                    chunks.append('<speak>' + sub_chunk_content + '</speak>')
            else:
                chunks.append('<speak>' + para + '</speak>')
        else:
            current_chunk.append(para)
            current_chunk_size += para_size
    
    if current_chunk:
        chunk_content = ''.join(current_chunk)
        chunks.append('<speak>' + chunk_content + '</speak>')
    
    return chunks if chunks else [ssml_text]


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


def process_chunk(chunk_index, chunk, temp_dir):
    """Process a single SSML chunk and generate audio.
    
    Args:
        chunk_index: Index of the chunk
        chunk: SSML chunk text
        temp_dir: Temporary directory for audio files
        
    Returns:
        tuple: (chunk_index, temp_file_path) or (chunk_index, None) if error
    """
    temp_file = os.path.join(temp_dir, f"chunk_{chunk_index:05d}.wav")
    try:
        result = synthesize_speech(ssml=chunk)
        save_audio(result, temp_file)
        return (chunk_index, temp_file)
    except Exception as e:
        print(f"Error processing chunk {chunk_index + 1}: {e}")
        print(f"Chunk length: {len(chunk)} characters")
        print(f"Chunk content:\n{chunk}")
        return (chunk_index, None)


def convert_epub_to_audiobook(epub_path, output_file):
    """Convert EPUB file to audiobook.
    
    Args:
        epub_path: Path to EPUB file
        output_file: Output audio file path
    """
    print(f"Starting EPUB to audiobook conversion...")
    print(f"Input: {epub_path}")
    print(f"Output: {output_file}")
    
    text = parse_epub(epub_path)
    
    print("Cleaning text with Gemini...")
    cleaned_text = clean_text_with_gemini(text)
    print(f"Cleaned text: {len(cleaned_text)} characters")
    
    print("Converting to SSML...")
    ssml = text_to_ssml(cleaned_text)
    
    print("Chunking SSML for TTS...")
    ssml_chunks = chunk_content(ssml, max_chars=3500)
    print(f"Created {len(ssml_chunks)} chunks")
    
    temp_dir = tempfile.mkdtemp()
    
    print("Generating audio in parallel...")
    max_workers = min(10, len(ssml_chunks))
    temp_audio_files = [None] * len(ssml_chunks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_chunk, i, chunk, temp_dir): i
            for i, chunk in enumerate(ssml_chunks)
        }
        
        completed = 0
        for future in as_completed(futures):
            chunk_index, temp_file = future.result()
            completed += 1
            if temp_file:
                temp_audio_files[chunk_index] = temp_file
                print(f"Completed chunk {chunk_index + 1}/{len(ssml_chunks)}")
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
    args = parser.parse_args()
    
    if args.output is None:
        title = get_epub_title(args.epub_path)
        if title:
            sanitized_title = re.sub(r'[<>:"/\\|?*]', '', title).strip()
            if not sanitized_title:
                sanitized_title = "output"
            args.output = f"{sanitized_title}.wav"
        else:
            args.output = "output.wav"
    
    convert_epub_to_audiobook(args.epub_path, args.output)
    