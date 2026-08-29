# Black History Factory v2

Four independent Google Colab processors for fast-paced ~90-second vertical historical/mythological shorts. **The self/host-video feature has been removed.** Every finished video ends with a blank background source card.

## What changed from the original ZIP

- Split the monolithic Colab into four independent notebooks.
- Replaced the old single-job pipeline with Drive job manifests/states.
- Qwen prepares up to 40 jobs from your existing Drive topic list.
- SDXL-Lightning only creates missing scene images.
- Audio is a separate processor; narration is generated with Piper and optional music/ambience/SFX are supplied from Drive libraries.
- Video assembly is a separate processor.
- Default format is 80–100 seconds, 175–220 spoken words, 1080x1920, 30 fps.
- Fast pacing uses many short scenes, hard cuts, zoom/pan motion and limited crossfades.
- Historical accuracy is separated from cinematic art direction.
- The final source card is blank/black and lists the research sources; if no specific source exists, it says so rather than inventing one.
- No host/self video code remains.

## Colab notebooks

1. `colab/01_Qwen_Processor.ipynb`
   - Reads unused topics from `01_TOPICS/topics.json`.
   - Creates a job folder and performs research + fact check + visual bible + narration + scene plan.
   - Stops when 40 Qwen-ready jobs exist by default.

2. `colab/02_Image_Processor.ipynb`
   - Loads `ByteDance/SDXL-Lightning`.
   - Processes only `QWEN_READY`/`IMAGES_PARTIAL` jobs.
   - Skips any `scene_###.png` that already exists, so a Colab reset does not regenerate completed images.

3. `colab/03_Audio_Processor.ipynb`
   - Uses Piper for scene narration.
   - Saves one WAV per scene and a concatenated narration track.
   - Put royalty-free audio into the Drive library folders for later mixing.

4. `colab/04_Video_Processor.ipynb`
   - Converts each still into a moving clip using zoom/pan.
   - Uses the actual narration WAV duration to time each visual.
   - Adds subtitles and optional background music.
   - Appends a 4.5-second blank source card with silence.
   - Copies the final MP4 to `05_OUTPUT/completed/`.

## Google Drive layout

```text
BLACK_HISTORY_FACTORY/
├── 00_CONFIG/
│   └── config.json
├── 01_TOPICS/
│   ├── topics.json
│   ├── used_topics.json
│   └── rejected_topics.json
├── 02_JOBS/
│   └── BH000001/
│       ├── job.json
│       ├── state/
│       ├── 01_research/
│       ├── 02_script/
│       ├── 03_scenes/
│       ├── 04_images/
│       ├── 05_audio/
│       ├── 06_video/
│       └── 07_thumbnail.png
├── 04_AUDIO_LIBRARY/
│   ├── music/
│   ├── ambience/
│   └── sfx/
├── 05_OUTPUT/
│   ├── completed/
│   └── failed/
├── 06_STATUS/
└── 07_LOGS/
```

The ZIP includes starter `00_CONFIG/config.json` and empty topic logs. **Replace `01_TOPICS/topics.json` with your existing topic list** or copy your existing file into that location. Do not delete it if it already contains your topics.

## Topic format

```json
[
  {
    "id": "T0001",
    "title": "Example topic",
    "category": "technology",
    "region": "East Africa",
    "period": "18th century",
    "description": "Specific surprising angle to investigate",
    "aliases": []
  }
]
```

## Audio library

Use only audio you are allowed to use. Suggested naming:

```text
04_AUDIO_LIBRARY/music/ancient-documentary.mp3
04_AUDIO_LIBRARY/ambience/fire-furnace.wav
04_AUDIO_LIBRARY/sfx/metal-strike.wav
```

The current v2 automatically uses the music library for global background music. The ambience/SFX folders are reserved for the next audio-mixing expansion, so the architecture already has a place for them without coupling them to Qwen or video rendering.

## Run order

Start Qwen first. Once it creates jobs, start Image, Audio and Video. You may run all four concurrently. Each processor only touches the files belonging to its stage.

### If a Colab times out

Simply reconnect/re-run that notebook. Existing files are checked before work starts. For example, if Image Processor generated 18 of 30 images before disconnecting, the next run skips those 18 and continues with the remaining 12.

## Visual direction

`00_CONFIG/config.json` contains the locked `art_style`. It is deliberately separated from historical facts. Qwen's visual bible determines what architecture, clothing, materials, people and environment are appropriate; the art style determines cinematic presentation. "Epic" therefore means lighting, scale, atmosphere and composition—not fabricated historical details.

## GitHub dashboard

`dashboard/` is a static status UI. Keep private Drive paths and tokens out of the public repository. The current ZIP leaves GitHub publishing intentionally non-destructive: you can connect your existing repository later and push only public status JSON. Public MP4 playback/download requires public storage for the video itself; a private Drive path cannot be used directly by GitHub Pages.


## v2.1 compatibility fixes

The supplied v2 ZIP had several cross-file API mismatches that could stop Cell 2 before Qwen loaded:
- `research_engine.py` now exports `RESEARCH_SCHEMA_KEYS` and `VALID_CLASSIFICATIONS`.
- `DrivePaths` now provides `research_raw()` and `research_verified()` aliases expected by `fact_checker.py`.
- `QwenClient` is loaded through its actual `QwenClient.load(...)` API.
- `script_engine.py` now has the missing `write_text_atomic()` utility.
- The four Colab notebooks install their processor-specific dependencies.
- SDXL-Lightning uses the appropriate Euler trailing scheduler.
- The visual bible accepts the newer structured `art_style` configuration.
- Drive folder names are consistent with the processor code: `04_AUDIO_LIBRARY`, `05_OUTPUT`, `06_STATUS`, `07_LOGS`.

For the first test, `prepared_job_target` is set to 1. Change it to 40 only after one complete video is approved.
