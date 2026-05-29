#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests

try:
    from google.cloud import texttospeech
except Exception:  # pragma: no cover - optional dependency at runtime
    texttospeech = None


MAX_SEGMENT_CHARS = 1400
SEGMENT_RETRIES = 4
REQUEST_TIMEOUT = 120


class SynthesisError(RuntimeError):
    pass


@dataclass
class Provider:
    name: str
    voice: str

    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError


class ElevenLabsProvider(Provider):
    def __init__(self, api_key: str, voice_id: str):
        super().__init__(name="elevenlabs", voice=voice_id)
        self.api_key = api_key
        self.voice_id = voice_id

    def synthesize(self, text: str) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code >= 400:
            raise SynthesisError(f"ElevenLabs request failed ({response.status_code}): {response.text[:400]}")
        return response.content


class OpenAIProvider(Provider):
    def __init__(self, api_key: str, voice: str):
        normalized_voice = voice.lower().strip()
        if normalized_voice not in {"onyx", "nova"}:
            raise SynthesisError("OPENAI_TTS_VOICE must be onyx or nova")
        super().__init__(name="openai", voice=normalized_voice)
        self.api_key = api_key

    def synthesize(self, text: str) -> bytes:
        url = "https://api.openai.com/v1/audio/speech"
        payload = {
            "model": "tts-1-hd",
            "voice": self.voice,
            "input": text,
            "format": "mp3",
        }
        headers = {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code >= 400:
            raise SynthesisError(f"OpenAI request failed ({response.status_code}): {response.text[:400]}")
        return response.content


class GoogleProvider(Provider):
    def __init__(self, credentials_json: str, voice: str):
        if texttospeech is None:
            raise SynthesisError("google-cloud-texttospeech is not installed")
        if "wavenet" not in voice.lower():
            raise SynthesisError("GOOGLE_CLOUD_TTS_VOICE must be a WaveNet voice")

        super().__init__(name="google-cloud-tts", voice=voice)
        self._cred_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self._cred_file.write(credentials_json.encode("utf-8"))
        self._cred_file.flush()
        self._cred_file.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._cred_file.name
        self.client = texttospeech.TextToSpeechClient()

    def synthesize(self, text: str) -> bytes:
        response = self.client.synthesize_speech(
            request={
                "input": texttospeech.SynthesisInput(text=text),
                "voice": texttospeech.VoiceSelectionParams(
                    language_code=self.voice.split("-")[0] + "-" + self.voice.split("-")[1],
                    name=self.voice,
                ),
                "audio_config": texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=1.0,
                    pitch=0.0,
                ),
            }
        )
        return response.audio_content


def split_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(sentence) <= max_chars:
                current = sentence
            else:
                words = sentence.split()
                word_acc = ""
                for word in words:
                    candidate_word = f"{word_acc} {word}".strip()
                    if len(candidate_word) <= max_chars:
                        word_acc = candidate_word
                    else:
                        if word_acc:
                            chunks.append(word_acc)
                        word_acc = word
                current = word_acc

        if current:
            chunks.append(current)

    return chunks


def synthesize_with_retries(provider: Provider, text: str) -> bytes:
    last_error = None
    for attempt in range(1, SEGMENT_RETRIES + 1):
        try:
            return provider.synthesize(text)
        except Exception as exc:  # pragma: no cover - external API handling
            last_error = exc
            if attempt == SEGMENT_RETRIES:
                break
            sleep_seconds = attempt * 2
            print(f"Segment failed on attempt {attempt}/{SEGMENT_RETRIES}; retrying in {sleep_seconds}s", flush=True)
            time.sleep(sleep_seconds)
    raise SynthesisError(f"Segment failed after {SEGMENT_RETRIES} retries using {provider.name}/{provider.voice}: {last_error}")


def build_provider() -> Provider:
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    elevenlabs_voice = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_voice = os.getenv("OPENAI_TTS_VOICE", "onyx").strip()
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    google_voice = os.getenv("GOOGLE_CLOUD_TTS_VOICE", "en-US-Wavenet-D").strip()

    if elevenlabs_key and elevenlabs_voice:
        return ElevenLabsProvider(elevenlabs_key, elevenlabs_voice)
    if openai_key:
        return OpenAIProvider(openai_key, openai_voice)
    if google_creds:
        return GoogleProvider(google_creds, google_voice)

    raise SynthesisError(
        "No TTS provider configured. Set ElevenLabs (key+voice), OpenAI API key, or Google credentials secret."
    )


def read_chapter_files(source_path: Path) -> List[Path]:
    if source_path.is_file():
        return [source_path]

    if not source_path.exists():
        raise SynthesisError(f"Source path does not exist: {source_path}")

    candidates = sorted(
        p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )
    if not candidates:
        raise SynthesisError(f"No .txt or .md files found under: {source_path}")
    return candidates


def concat_mp3_files(segment_paths: List[Path], output_path: Path) -> None:
    concat_list_path = output_path.parent / "concat.txt"
    with concat_list_path.open("w", encoding="utf-8") as handle:
        for path in segment_paths:
            handle.write(f"file '{path.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "160k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an audiobook MP3 from text files")
    parser.add_argument("--source", required=True, help="Path to a source text file or directory")
    parser.add_argument("--output", default="dist/audiobook.mp3", help="Output MP3 path")
    parser.add_argument("--work-dir", default="dist/work", help="Directory for chapter/segment artifacts")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    work_dir = Path(args.work_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    provider = build_provider()
    print(f"Using provider: {provider.name} | voice: {provider.voice}", flush=True)

    chapter_files = read_chapter_files(source_path)
    all_segments: List[Path] = []

    for chapter_index, chapter_file in enumerate(chapter_files, start=1):
        chapter_text = chapter_file.read_text(encoding="utf-8")
        segments = split_text(chapter_text)
        if not segments:
            continue

        print(f"Chapter {chapter_index}: {chapter_file} -> {len(segments)} segment(s)", flush=True)
        for segment_index, segment_text in enumerate(segments, start=1):
            audio = synthesize_with_retries(provider, segment_text)
            segment_path = work_dir / f"chapter{chapter_index:04d}_segment{segment_index:05d}.mp3"
            segment_path.write_bytes(audio)
            all_segments.append(segment_path)

    if not all_segments:
        raise SynthesisError("No audio was generated from the provided source files")

    concat_mp3_files(all_segments, output_path)

    metadata = {
        "provider": provider.name,
        "voice": provider.voice,
        "chapters": len(chapter_files),
        "segments": len(all_segments),
        "source": str(source_path),
        "output": str(output_path),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
