#!/usr/bin/env python3
"""
Simple wrapper for VibeVoice Realtime TTS inference.

Usage examples:
    # Generate speech from inline text
    python generate_speech.py --text "Hello, this is a test of VibeVoice."

    # Generate speech from a text file
    python generate_speech.py --file input.txt

    # Specify a voice and output path
    python generate_speech.py --text "Good morning!" --voice emma --output morning.wav

    # List available voices
    python generate_speech.py --list-voices
"""

import argparse
import copy
import glob
import os
import sys
import time

import torch

VOICES_DIR = os.path.join(os.path.dirname(__file__), "demo", "voices", "streaming_model")
DEFAULT_MODEL = "microsoft/VibeVoice-Realtime-0.5B"


def get_available_voices():
    """Scan the voices directory and return a dict of name -> path."""
    voices = {}
    if not os.path.isdir(VOICES_DIR):
        return voices
    for pt_file in sorted(glob.glob(os.path.join(VOICES_DIR, "*.pt"))):
        name = os.path.splitext(os.path.basename(pt_file))[0].lower()
        voices[name] = os.path.abspath(pt_file)
    return voices


def resolve_voice(voice_name, voices):
    """Resolve a voice name (case-insensitive, partial match)."""
    key = voice_name.lower()
    # Exact match
    if key in voices:
        return voices[key]
    # Partial match
    matches = [n for n in voices if key in n]
    if len(matches) == 1:
        return voices[matches[0]]
    if len(matches) > 1:
        print(f"Error: '{voice_name}' matches multiple voices: {', '.join(matches)}")
        sys.exit(1)
    print(f"Error: No voice found matching '{voice_name}'.")
    print(f"Available voices: {', '.join(voices.keys())}")
    sys.exit(1)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech audio from text using VibeVoice Realtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_speech.py --text "Hello world"
  python generate_speech.py --file article.txt --voice davis --output article.wav
  python generate_speech.py --list-voices
        """,
    )
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--file", type=str, help="Path to a .txt file to synthesize")
    parser.add_argument("--voice", type=str, default="carter",
                        help="Voice name (default: carter). Use --list-voices to see options")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output .wav file path (default: output.wav)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda | mps | cpu (auto-detected if omitted)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (default: 1.5)")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices and exit")

    args = parser.parse_args()
    voices = get_available_voices()

    # --list-voices
    if args.list_voices:
        if not voices:
            print("No voice presets found. Run: bash demo/download_experimental_voices.sh")
        else:
            print("Available voices:")
            for name in voices:
                print(f"  {name}")
        return

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        parser.error("Provide --text or --file")

    if not text:
        print("Error: Input text is empty.")
        sys.exit(1)

    # Normalize quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Resolve voice
    voice_path = resolve_voice(args.voice, voices)
    voice_display = os.path.splitext(os.path.basename(voice_path))[0]

    # Device
    device = args.device or pick_device()
    if device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU.")
        device = "cpu"

    print(f"Device:  {device}")
    print(f"Voice:   {voice_display}")
    print(f"Model:   {args.model}")
    print(f"Output:  {args.output}")
    print(f"Text:    {text[:120]}{'...' if len(text) > 120 else ''}")
    print()

    # -- Load model --
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    print("Loading processor...")
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model)

    if device == "mps":
        load_dtype = torch.float32
        attn_impl = "sdpa"
    elif device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl = "sdpa"

    print("Loading model...")
    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model, torch_dtype=load_dtype,
                attn_implementation=attn_impl, device_map=None,
            )
            model.to("mps")
        elif device == "cuda":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model, torch_dtype=load_dtype,
                device_map="cuda", attn_implementation=attn_impl,
            )
        else:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model, torch_dtype=load_dtype,
                device_map="cpu", attn_implementation=attn_impl,
            )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"flash_attention_2 failed ({e}), retrying with sdpa...")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                args.model, torch_dtype=load_dtype,
                device_map=(device if device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if device == "mps":
                model.to("mps")
        else:
            raise

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    # -- Load voice preset --
    target_device = device
    all_prefilled_outputs = torch.load(voice_path, map_location=target_device, weights_only=False)

    # -- Prepare inputs --
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    # -- Generate --
    print("Generating speech...")
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs)
        if all_prefilled_outputs is not None
        else None,
    )
    elapsed = time.time() - start_time

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        print("Error: No audio output generated.")
        sys.exit(1)

    # -- Save --
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=args.output)

    # Stats
    sample_rate = 24000
    audio_samples = (
        outputs.speech_outputs[0].shape[-1]
        if len(outputs.speech_outputs[0].shape) > 0
        else len(outputs.speech_outputs[0])
    )
    duration = audio_samples / sample_rate
    rtf = elapsed / duration if duration > 0 else float("inf")

    print()
    print(f"Saved:     {args.output}")
    print(f"Duration:  {duration:.1f}s")
    print(f"Gen time:  {elapsed:.1f}s")
    print(f"RTF:       {rtf:.2f}x")


if __name__ == "__main__":
    main()
