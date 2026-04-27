#!/usr/bin/env python3
"""
Generate speech from text using VibeVoice-1.5B (long-form multi-speaker TTS).

Usage examples:
    # Generate speech from inline text (single speaker)
    python generate_speech_1_5b.py --text "Hello, this is a test of VibeVoice."

    # Generate speech from a text file
    python generate_speech_1_5b.py --file input.txt

    # Multi-speaker dialogue (use "Speaker N:" prefix)
    python generate_speech_1_5b.py --text "Speaker 0: Hello there! Speaker 1: Hi, how are you?"

    # Specify output path and parameters
    python generate_speech_1_5b.py --text "Good morning!" -o morning.wav --cfg-scale 1.3

    # Use voice cloning with a reference audio file
    python generate_speech_1_5b.py --text "Hello world" --voice-audio reference.wav

Setup (CUDA / GPU acceleration):
    The script auto-detects CUDA. If torch was installed without GPU support
    (torch.cuda.is_available() returns False), reinstall it for your CUDA version:

    1. Check your driver's CUDA version:
           nvidia-smi          (look for "CUDA Version" in the top-right)

    2. Install the matching torch wheel:
           # CUDA 11.8
           pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
           # CUDA 12.1
           pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
           # CUDA 12.8 (nightly, for CUDA 12.8/13.x drivers)
           pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

       For the canonical command for your exact version, visit:
           https://pytorch.org/get-started/locally/

    3. Verify:
           python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
           # Should print: True  12.x
"""

import argparse
import math
import os
import re
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

DEFAULT_MODEL = "microsoft/VibeVoice-1.5B"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def sample_speech_tokens(model, condition, neg_condition, cfg_scale=1.3):
    """
    Sample one acoustic latent frame via classifier-free guided diffusion.
    Mirrors the 0.5B streaming model's sample_speech_tokens exactly.
    """
    scheduler = model.model.noise_scheduler
    scheduler.set_timesteps(model._ddpm_inference_steps)
    pred_head = model.model.prediction_head

    # condition / neg_condition: (B, hidden_dim)
    cond = torch.cat([condition, neg_condition], dim=0).to(pred_head.device)
    speech = torch.randn(cond.shape[0], model.config.acoustic_vae_dim, device=cond.device, dtype=cond.dtype)

    for t in scheduler.timesteps:
        half = speech[: len(speech) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = pred_head(combined, t.repeat(combined.shape[0]).to(combined), condition=cond)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        speech = scheduler.step(eps, t, speech).prev_sample

    return speech[: len(speech) // 2]


@torch.no_grad()
def generate_speech(model, processor, text, device, cfg_scale=1.3,
                    diffusion_steps=20, voice_audio=None, verbose=True,
                    max_speech_tokens=4096):
    """
    Full inference pipeline for VibeVoice-1.5B.

    The model autoregressively predicts text tokens via lm_head. When it emits
    speech_diffusion_id, the diffusion head generates an acoustic latent frame
    which is fed back as the next input embedding. On speech_end_id or EOS the
    speech segment ends. All acoustic latents are decoded through the acoustic
    tokenizer to produce the final waveform.
    """
    model._ddpm_inference_steps = diffusion_steps

    tokenizer = processor.tokenizer
    speech_start_id = tokenizer.speech_start_id
    speech_end_id = tokenizer.speech_end_id
    speech_diffusion_id = tokenizer.speech_diffusion_id

    # ---- Build input tokens via the processor ----
    inputs = processor(
        text=text,
        voice_samples=[voice_audio] if voice_audio is not None else None,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_ids = inputs["input_ids"].to(device)                     # (1, S)
    attention_mask = inputs["attention_mask"].to(device)            # (1, S)
    speech_input_mask = inputs["speech_input_mask"].to(device)      # (1, S) bool

    # ---- Build input embeddings, splicing in voice prompt features ----
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)  # (1, S, D)

    if inputs.get("speech_tensors") is not None and inputs["speech_tensors"] is not None:
        speech_tensors = inputs["speech_tensors"].to(device).to(model.dtype)
        speech_masks = inputs["speech_masks"].to(device)

        # Encode voice audio → acoustic latents → connector embeddings
        _audio_features, connect_features = model.forward_speech_features(
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            speech_type="audio",
            return_unmask=True,
        )

        # Replace the speech_diffusion_id placeholder positions with voice embeddings
        inputs_embeds[speech_input_mask] = connect_features[speech_masks]

        # Also add semantic pathway
        sem_frames = model.model.semantic_tokenizer(speech_tensors.unsqueeze(1) if speech_tensors.dim() == 2 else speech_tensors)
        if isinstance(sem_frames, tuple):
            sem_latents = sem_frames[1] if len(sem_frames) > 1 else sem_frames[0]
        else:
            sem_latents = sem_frames
        sem_connect = model.model.semantic_connector(sem_latents)
        # Add semantic features to the voice prompt positions
        inputs_embeds[speech_input_mask] = inputs_embeds[speech_input_mask] + sem_connect[speech_masks]

    # ---- Prefill ----
    seq_len = inputs_embeds.shape[1]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    cache_position = torch.arange(seq_len, device=device)

    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
        cache_position=cache_position,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state  # (1, S, D)

    # ---- Negative (unconditional) context for CFG ----
    # Use a single padding token as the unconditional prompt
    neg_token_id = tokenizer.pad_id if isinstance(tokenizer.pad_id, int) and tokenizer.pad_id >= 0 else 0
    neg_ids = torch.tensor([[neg_token_id]], device=device)
    neg_embeds = embed_layer(neg_ids)

    neg_outputs = model.model(
        inputs_embeds=neg_embeds,
        attention_mask=torch.ones(1, 1, device=device, dtype=torch.long),
        position_ids=torch.zeros(1, 1, device=device, dtype=torch.long),
        use_cache=True,
        return_dict=True,
        cache_position=torch.tensor([0], device=device),
    )
    neg_past = neg_outputs.past_key_values
    neg_hidden = neg_outputs.last_hidden_state

    # ---- Autoregressive generation ----
    all_speech_latents = []
    cur_pos = seq_len
    neg_pos = 1

    if verbose:
        pbar = tqdm(total=max_speech_tokens, desc="Generating speech", leave=False)

    speech_count = 0
    last_token_id = input_ids[0, -1].item()

    for step in range(max_speech_tokens):
        # Get logits from the last hidden state
        logits = model.lm_head(hidden[:, -1:, :])  # (1, 1, vocab)
        next_token_id = logits[0, 0].argmax(dim=-1).item()

        # Termination
        if next_token_id == speech_end_id:
            if verbose:
                pbar.set_description(f"Speech end after {speech_count} frames")
            break
        eos_ids = [tokenizer.eos_id] if hasattr(tokenizer, 'eos_id') else []
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if eos_id is not None:
            eos_ids.append(eos_id)
        if next_token_id in eos_ids:
            break

        if next_token_id == speech_diffusion_id:
            # ---- Diffusion: generate one acoustic latent frame ----
            condition = hidden[:, -1, :]         # (1, D)
            neg_condition = neg_hidden[:, -1, :]  # (1, D)
            speech_latent = sample_speech_tokens(
                model, condition, neg_condition, cfg_scale=cfg_scale,
            )  # (1, vae_dim)
            all_speech_latents.append(speech_latent)
            speech_count += 1

            # Feed the acoustic latent back as the next input embedding
            next_embeds = model.model.acoustic_connector(speech_latent.unsqueeze(1))  # (1, 1, D)
        else:
            # Regular text token — embed it
            next_token = torch.tensor([[next_token_id]], device=device)
            next_embeds = embed_layer(next_token)  # (1, 1, D)

        # Step the main (conditional) KV cache forward
        new_attn = torch.ones(1, 1, device=device, dtype=torch.long)
        full_attn = torch.cat([attention_mask, new_attn], dim=-1)
        new_pos_id = full_attn.long().cumsum(-1)[:, -1:]  - 1
        new_cache_pos = torch.tensor([cur_pos], device=device)

        outputs = model.model(
            inputs_embeds=next_embeds,
            attention_mask=full_attn,
            position_ids=new_pos_id,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            cache_position=new_cache_pos,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state
        attention_mask = full_attn
        cur_pos += 1

        # Step the negative KV cache forward (always feed same token/embed)
        if next_token_id == speech_diffusion_id:
            neg_next_embeds = model.model.acoustic_connector(speech_latent.unsqueeze(1))
        else:
            neg_next_embeds = embed_layer(torch.tensor([[neg_token_id]], device=device))

        neg_full_attn = torch.ones(1, neg_pos + 1, device=device, dtype=torch.long)
        neg_new_pos = torch.tensor([[neg_pos]], device=device, dtype=torch.long)
        neg_cache_pos = torch.tensor([neg_pos], device=device)

        neg_out = model.model(
            inputs_embeds=neg_next_embeds,
            attention_mask=neg_full_attn,
            position_ids=neg_new_pos,
            past_key_values=neg_past,
            use_cache=True,
            return_dict=True,
            cache_position=neg_cache_pos,
        )
        neg_past = neg_out.past_key_values
        neg_hidden = neg_out.last_hidden_state
        neg_pos += 1

        if verbose:
            pbar.update(1)
            pbar.set_description(f"Generated {speech_count} speech frames")

    if verbose:
        pbar.close()

    if not all_speech_latents:
        return None

    # ---- Decode acoustic latents to waveform ----
    latent_stack = torch.cat(all_speech_latents, dim=0).unsqueeze(0)  # (1, N, vae_dim)

    # Un-scale: reverse of encode's (x + bias) * scale  →  x = x'/scale - bias
    scaling = model.model.speech_scaling_factor.to(latent_stack.device)
    bias = model.model.speech_bias_factor.to(latent_stack.device)
    latent_stack = latent_stack / scaling - bias

    audio = model.model.acoustic_tokenizer.decode(
        latent_stack.to(model.model.acoustic_tokenizer.device),
        use_cache=False,
    )

    return audio[0]  # (T,) waveform


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech from text using VibeVoice-1.5B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_speech_1_5b.py --text "Hello world"
  python generate_speech_1_5b.py --file article.txt --output article.wav
  python generate_speech_1_5b.py --text "Speaker 0: Hi! Speaker 1: Hello!" --cfg-scale 1.3

Multi-speaker format (in text files or --text):
  Speaker 0: Hello everyone, welcome to the show.
  Speaker 1: Thanks for having me!
        """,
    )
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--file", type=str, help="Path to a .txt file to synthesize")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output .wav file path (default: output.wav)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda | mps | cpu (auto-detected if omitted)")
    parser.add_argument("--cfg-scale", type=float, default=1.3,
                        help="Classifier-free guidance scale (default: 1.3)")
    parser.add_argument("--diffusion-steps", type=int, default=20,
                        help="Number of diffusion denoising steps (default: 20)")
    parser.add_argument("--max-speech-tokens", type=int, default=4096,
                        help="Maximum number of speech latent frames to generate (default: 4096)")
    parser.add_argument("--voice-audio", type=str, default=None,
                        help="Path to a reference audio file for voice cloning (3-30s, clear speech)")

    args = parser.parse_args()

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

    # If text doesn't have speaker tags, wrap it as Speaker 0
    if not re.search(r'Speaker\s+\d+\s*:', text, re.IGNORECASE):
        text = f"Speaker 0: {text}"

    # Device
    device = args.device or pick_device()
    if device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU.")
        device = "cpu"

    # Load voice audio if provided
    voice_audio = None
    if args.voice_audio:
        try:
            import librosa
            voice_audio, _ = librosa.load(args.voice_audio, sr=24000, mono=True)
        except ImportError:
            import soundfile as sf
            voice_audio, sr = sf.read(args.voice_audio)
            if sr != 24000:
                # Simple resampling fallback
                import scipy.signal
                voice_audio = scipy.signal.resample(
                    voice_audio, int(len(voice_audio) * 24000 / sr)
                ).astype(np.float32)

    print(f"Device:           {device}")
    print(f"Model:            {args.model}")
    print(f"Output:           {args.output}")
    print(f"CFG scale:        {args.cfg_scale}")
    print(f"Diffusion steps:  {args.diffusion_steps}")
    if args.voice_audio:
        print(f"Voice audio:      {args.voice_audio}")
    print(f"Text:             {text[:120]}{'...' if len(text) > 120 else ''}")
    print()

    # -- Load model --
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    print("Loading processor...")
    processor = VibeVoiceProcessor.from_pretrained(args.model)

    if device == "mps":
        load_dtype = torch.float32
        attn_impl = "sdpa"
    elif device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl = "sdpa"

    print("Loading model (downloads ~5.4 GB on first run)...")
    try:
        if device == "mps":
            model = VibeVoiceForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=load_dtype,
                attn_implementation=attn_impl, device_map=None,
            )
            model.to("mps")
        elif device == "cuda":
            model = VibeVoiceForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=load_dtype,
                device_map="cuda", attn_implementation=attn_impl,
            )
        else:
            model = VibeVoiceForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=load_dtype,
                device_map="cpu", attn_implementation=attn_impl,
            )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"flash_attention_2 failed ({e}), retrying with sdpa...")
            model = VibeVoiceForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=load_dtype,
                device_map=(device if device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if device == "mps":
                model.to("mps")
        else:
            raise

    model.eval()

    # -- Generate speech --
    print("Generating speech...")
    start_time = time.time()

    audio = generate_speech(
        model=model,
        processor=processor,
        text=text,
        device=device,
        cfg_scale=args.cfg_scale,
        diffusion_steps=args.diffusion_steps,
        voice_audio=voice_audio,
        verbose=True,
        max_speech_tokens=args.max_speech_tokens,
    )

    elapsed = time.time() - start_time

    if audio is None:
        print("Error: No speech output generated (model produced no speech tokens).")
        sys.exit(1)

    # -- Save --
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    processor.save_audio(audio, output_path=args.output)

    # Stats
    sample_rate = 24000
    if isinstance(audio, torch.Tensor):
        audio_samples = audio.shape[-1]
    else:
        audio_samples = len(audio)
    duration = audio_samples / sample_rate
    rtf = elapsed / duration if duration > 0 else float("inf")

    print()
    print(f"Saved:      {args.output}")
    print(f"Duration:   {duration:.1f}s")
    print(f"Gen time:   {elapsed:.1f}s")
    print(f"RTF:        {rtf:.2f}x")


if __name__ == "__main__":
    main()
