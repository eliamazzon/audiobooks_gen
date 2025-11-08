import requests
import json
import os
import argparse
import time
import re
import uuid
from urllib.parse import quote
from dotenv import load_dotenv
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or GOOGLE_API_KEY
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
BASE_URL = "https://texttospeech.googleapis.com/v1"
GCS_BASE_URL = "https://storage.googleapis.com"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


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


def synthesize_long_audio(ssml, output_gcs_uri, voice_name="en-us-Chirp3-HD-Aoede", speaking_rate=1.0):
    """Submit long audio synthesis request using REST API.
    
    Args:
        ssml: SSML formatted text
        output_gcs_uri: GCS URI where audio will be saved (gs://bucket/path)
        voice_name: Voice name (default: en-us-Chirp3-HD-Aoede)
        speaking_rate: Speech rate (0.25 to 2.0)
        
    Returns:
        str: Operation name for polling
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Please set GOOGLE_API_KEY environment variable.")
    
    url = f"{BASE_URL}/projects/-/locations/global:synthesizeLongAudio?key={GOOGLE_API_KEY}"
    
    payload = {
        "input": {
            "ssml": ssml
        },
        "voice": {
            "languageCode": "en-US",
            "name": voice_name
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "speakingRate": speaking_rate
        },
        "outputGcsUri": output_gcs_uri
    }
    
    response = requests.post(url, json=payload)
    
    if not response.ok:
        print(f"Error response: {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()
    
    result = response.json()
    operation_name = result.get("name")
    
    if not operation_name:
        raise ValueError("No operation name returned from API")
    
    print(f"Long audio synthesis started. Operation: {operation_name}")
    return operation_name


def wait_for_completion(operation_name, poll_interval=5):
    """Wait for long audio synthesis operation to complete.
    
    Args:
        operation_name: Operation name from synthesize_long_audio
        poll_interval: Seconds between status checks
        
    Returns:
        dict: Operation result when complete
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")
    
    url = f"{BASE_URL}/{operation_name}?key={GOOGLE_API_KEY}"
    
    print("Waiting for operation to complete...")
    start_time = time.time()
    
    while True:
        response = requests.get(url)
        
        if not response.ok:
            print(f"Error checking operation status: {response.status_code}")
            print(f"Response: {response.text}")
            response.raise_for_status()
        
        operation = response.json()
        
        if "done" in operation and operation["done"]:
            elapsed = time.time() - start_time
            print(f"Operation completed in {elapsed:.1f} seconds")
            
            if "error" in operation:
                error = operation["error"]
                raise Exception(f"Operation failed: {error.get('message', 'Unknown error')}")
            
            return operation
        
        if "response" in operation:
            elapsed = time.time() - start_time
            print(f"Still processing... ({elapsed:.0f}s elapsed)")
        
        time.sleep(poll_interval)


def get_oauth_token():
    """Get OAuth2 access token using API key.
    
    Returns:
        str: OAuth2 access token
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")
    
    url = "https://oauth2.googleapis.com/token"
    data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": GOOGLE_API_KEY
    }
    
    response = requests.post(url, data=data)
    if response.ok:
        return response.json().get("access_token")
    
    raise ValueError("Could not obtain OAuth2 token. Please ensure GOOGLE_API_KEY is valid.")


def download_from_gcs(gcs_uri, output_file):
    """Download audio file from Google Cloud Storage.
    
    Args:
        gcs_uri: GCS URI (gs://bucket/path)
        output_file: Local file path to save audio
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    parts = gcs_uri[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
    
    bucket = parts[0]
    object_path = parts[1]
    
    url = f"{GCS_BASE_URL}/storage/v1/b/{bucket}/o/{quote(object_path, safe='')}?alt=media"
    
    print(f"Downloading audio from GCS: {gcs_uri}")
    
    try:
        if GOOGLE_API_KEY:
            token = get_oauth_token()
            headers = {"Authorization": f"Bearer {token}"}
        else:
            headers = {}
        
        response = requests.get(url, headers=headers, stream=True)
        
        if not response.ok:
            if response.status_code == 401:
                raise ValueError("Authentication failed. Please check your GOOGLE_API_KEY or use service account credentials.")
            print(f"Error downloading from GCS: {response.status_code}")
            print(f"Response: {response.text}")
            response.raise_for_status()
        
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(output_file)
        print(f"Audio downloaded to {output_file} ({file_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: GCS download requires proper authentication.")
        print("You may need to use service account credentials or make the bucket publicly readable.")
        raise


def delete_from_gcs(gcs_uri):
    """Delete file from Google Cloud Storage.
    
    Args:
        gcs_uri: GCS URI (gs://bucket/path)
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    parts = gcs_uri[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
    
    bucket = parts[0]
    object_path = parts[1]
    
    url = f"{GCS_BASE_URL}/storage/v1/b/{bucket}/o/{quote(object_path, safe='')}"
    
    try:
        if GOOGLE_API_KEY:
            token = get_oauth_token()
            headers = {"Authorization": f"Bearer {token}"}
        else:
            headers = {}
        
        response = requests.delete(url, headers=headers)
        
        if response.ok:
            print(f"Deleted file from GCS: {gcs_uri}")
        else:
            print(f"Warning: Could not delete file from GCS: {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not delete file from GCS: {e}")


def convert_epub_to_audiobook_long(epub_path, output_file, bucket_name=None, keep_gcs=False):
    """Convert EPUB file to audiobook using Long Audio API.
    
    Args:
        epub_path: Path to EPUB file
        output_file: Output audio file path
        bucket_name: GCS bucket name (defaults to GCS_BUCKET_NAME env var)
        keep_gcs: If True, keep the file in GCS after downloading
    """
    print(f"Starting EPUB to audiobook conversion (Long Audio API)...")
    print(f"Input: {epub_path}")
    print(f"Output: {output_file}")
    
    bucket = bucket_name or GCS_BUCKET_NAME
    if not bucket:
        raise ValueError("GCS bucket name required. Set GCS_BUCKET_NAME environment variable or use --bucket argument.")
    
    text = parse_epub(epub_path)
    
    print("Cleaning text with Gemini...")
    cleaned_text = clean_text_with_gemini(text)
    print(f"Cleaned text: {len(cleaned_text)} characters")
    
    print("Converting to SSML...")
    ssml = text_to_ssml(cleaned_text)
    print(f"SSML length: {len(ssml)} characters")
    
    unique_id = str(uuid.uuid4())[:8]
    gcs_path = f"audiobooks/{unique_id}_{os.path.basename(output_file)}"
    gcs_uri = f"gs://{bucket}/{gcs_path}"
    
    print(f"Submitting to Long Audio API...")
    print(f"Output will be saved to: {gcs_uri}")
    
    operation_name = synthesize_long_audio(ssml, gcs_uri)
    
    wait_for_completion(operation_name)
    
    print("Downloading audio from GCS...")
    download_from_gcs(gcs_uri, output_file)
    
    if not keep_gcs:
        print("Cleaning up GCS file...")
        delete_from_gcs(gcs_uri)
    
    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EPUB to audiobook using Long Audio API")
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument("-o", "--output", default=None, help="Output audio file (default: book title)")
    parser.add_argument("--bucket", default=None, help="GCS bucket name (defaults to GCS_BUCKET_NAME env var)")
    parser.add_argument("--keep-gcs", action="store_true", help="Keep audio file in GCS after downloading")
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
    
    convert_epub_to_audiobook_long(args.epub_path, args.output, args.bucket, args.keep_gcs)

