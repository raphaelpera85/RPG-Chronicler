# Audio DSP and Agent Pipeline Rules

## 1. Feature Dimension Invariants & Backwards Compatibility
- The base acoustic feature extractor (`extract_acoustic_features`) must remain strictly fixed to 38 dimensions to ensure backward compatibility with stored voice profiles in `perfis_vozes.json`.
- Any extended or experimental feature extraction (e.g. formants, HNR, higher-order spectral moments) must be implemented as a separate opt-in function (`extract_extended_acoustic_features` - 52D) to avoid invalidating existing user-trained models or existing unit tests.

## 2. Numerical Stability in Audio Filters (IIR / Biquad)
- When designing and applying digital biquad/IIR filters (especially high-pass, shelf, or parametric EQ filters under 100 Hz), always use `float64` (`np.float64`) for coefficients and signal arrays in `scipy.signal.filtfilt` or `lfilter`.
- Low-frequency biquad filters on 32-bit floats suffer from poles close to the unit circle, causing numerical overflow or NaN artifacts.

## 3. Broadcast Loudness Standardization (EBU R128 / Podcasts)
- Final podcast masters must comply with EBU R128 / ITU-R BS.1770-4:
  - Target Integrated Loudness: -16 LUFS (industry standard for spoken podcasts on Spotify/Apple Podcasts/YouTube).
  - True-Peak ceiling: <= -1.0 dBFS to prevent inter-sample clipping when compressed to AAC/MP3.
  - Implement dual gating: absolute threshold at -70 LKFS and relative threshold at -10 LKFS below non-gated loudness.

## 4. Modular Specialists Runtime Decoupling
- Audiovisual and editorial specialists (Show Notes, Viral Clips, Lore Keeper, Mastering Engineer, Video Generator) must execute as decoupled agents in `agents/specialists.py`.
- Each agent must handle its exceptions independently and serialize its output artifacts into the session directory (`runs/<run-id>/`) without breaking the primary transcription and diarization completion pipeline.

## 5. High-Performance Windows Module & DLL Scanning
- On Windows startup, avoid unbounded recursive `os.walk` across complete virtual environment packages. Filter top-level package directories with `os.scandir` for specific vendor prefixes (`nvidia*`, `torch*`, `ctranslate2*`) to prevent 30-40 second load penalties.

## 6. Non-Blocking FFmpeg Subprocess Execution
- Any subprocess calling `ffmpeg` on Windows must specify `-nostdin` and pass `stdin=subprocess.DEVNULL`.
- Complex video generation filters (`overlay`, `showwaves`, `color`) must use `shortest=1` and an explicit output duration (`-t {duration}`) to prevent infinite stream generation.

## 7. Embedding Variance and Numerical Validity Invariant
- Voice feature vectors are only valid if they exhibit finite values and real variance (`np.std(arr) >= 1e-4`). Flat constant vectors or silent DC inputs must be rejected to prevent contaminating cumulative profile centroids.

## 8. Acting-Invariant Voice Diarization (Roleplay-Tolerant Matching)
- In tabletop RPGs, players frequently modulate their vocal pitch (F0) dramatically when roleplaying monsters, shouting, or whispering.
- Diarization matching algorithms must calculate an acting-tolerant distance metric that down-weights pitch/F0 variations (weight ~0.20) and prioritizes anatomical vocal tract formants (MFCCs with weight ~1.60) to avoid misclassifying expressive players as unknown speakers while preserving the 38-D embedding representation.

## 9. Whisper Vocabulary Prompt Biasing
- Speech recognition for RPG campaigns must inject campaign-specific proper nouns (NPCs, character names, factions, locations, and system terms) extracted from `biblia_personagens_e_cenarios.md` into Whisper's `initial_prompt`.
- This eliminates phonetic hallucinations (e.g. converting "Tiefling" into common words) with zero additional inference latency.

## 10. Phonetic & Mechanical Normalization
- Conversational dice expressions and RPG jargon (e.g. "um d 20", "20 natural", "classe de armadura") must be normalized into standard RPG notation ("1d20", "20 natural", "CA", "PV", "CD") immediately following transcription to optimize downstream NLP, lore graph extraction, and search indexing.

## 11. Passive Audio Monitoring & Resilient Hardware Probing
- Audio device enumeration must be resilient against missing host APIs or faulty sound drivers; always query with protective fallbacks.
- Microphone monitoring before recording must run in lightweight passive streams that calculate RMS volume for visual meters without allocating queue buffers or disk writers, and must be terminated cleanly before full recording begins.

## 12. Self-Contained Offline Publishing Bundles
- Session exports (ZIP bundles) must be standalone and self-contained, including responsive HTML portals (`index.html`), interactive relationship graphs, subtitles (`.srt`, `.vtt`, `.ass`), and audio, enabling players to consume campaign content locally without needing Python or external server installations.

## 13. Zero-Dependency Embedded Companion Server Architecture
- Local LAN companion web servers designed for player interaction (smartphones, tablets at the table) should rely exclusively on Python standard library `http.server.ThreadingHTTPServer` running on a background daemon thread.
- Avoid external web frameworks (FastAPI, Flask) in desktop client environments to preserve rapid startup times, eliminate dependency conflicts, and ensure graceful shutdown upon application exit (`WM_DELETE_WINDOW`).

## 14. Native PowerShell TTS Bridge for Windows Environments
- Text-to-Speech narration for session recaps in Windows desktop applications without heavy external pip wheels (`pyttsx3`, `pydub`, `soundfile`) should interface natively with Windows `System.Speech.Synthesis.SpeechSynthesizer` via PowerShell subprocess execution.
- Temporary script files or inputs must be strictly encoded in UTF-8 (`[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`) to avoid character mangling with accented Portuguese phonemes.

## 15. Lossless Archival Storage Optimization
- Raw session WAV captures should be compressible to FLAC (compression level 8) using the native system `ffmpeg` binary (`-nostdin`, `-y`).
- Compression must compute and report verifiable metrics (`orig_size_bytes`, `new_size_bytes`, `saved_bytes`, `compression_ratio_pct`) and only remove source files if explicitly opted-in and verification succeeds.

## 16. Turnkey VTT and Personal Knowledge Management Interoperability
- Campaign exports must support direct ingestion into standard RPG and note-taking tooling:
  - **Obsidian Vault**: Generates structured folders (`Sessoes/`, `Entidades/`, `Destaques/`) with canonical YAML frontmatter. Entity names must be sorted by length descending before injecting wikilinks `[[...]]` to prevent partial substring collision.
  - **Foundry VTT**: Produces valid `JournalEntry` JSON schema compatible with Foundry v10/v11/v12 including styled HTML pages for recap, combat highlights, and dialogue transcript.

## 17. Human-in-the-Loop Timeline Highlighting for Viral Clips
- Live operator hotkeys (`<F9>` Epic, `<F10>` Critical, `<F11>` Plot Twist) create timestamped anchor points (`TimelineHighlightManager`) that directly feed downstream viral clip extractors (`SocialClipsViralScoutAgent`).
- Human temporal markers should confer substantial score boosts, aligning algorithmic clip curation with the table's genuine dramatic highlights.

