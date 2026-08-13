# Black History Factory

An automated documentary generator: picks an unusual, non-repeating topic in
Black/African history and culture, researches it, writes a narration, breaks
it into scenes, generates images, generates narration audio, and renders an
MP4 — all from a Google Colab notebook, checkpointed so it survives Colab
disconnects, with a small GitHub Pages dashboard to watch progress from your
phone.

## What's implemented so far

**Phase A (foundation)**, **Phase B (topics)**, **Phase C (research)**,
**Phase D (writing)**, and **Phase E (media)** are done — every pipeline
stage is a real implementation. From Phase F/G/H:

- **Job archiving** (Phase F) — every completed job gets a manifest at
  `09_LOGS/{job_id}.manifest.json`: topic metadata, scene count, sources
  cited, output paths, and a full stage-by-stage timing history (when the
  job entered each stage), separate from the checkpoint file (which is
  working state for resume logic, not a long-term record).
- **Stalled-session detection** (Phase G) — the dashboard no longer trusts
  a stale `"running"` status at face value. If `current.json` hasn't been
  updated in over 3 minutes while still claiming to be running, the
  dashboard shows "possibly disconnected" with the last known checkpoint
  and "Colab session likely ended" — matching the spec's paused-state
  example — computed client-side in `app.js` (tested against 5 scenarios
  in real Node).
- **Local self-test suite** (Phase H tooling) — `tests/self_test.py` runs
  the full pipeline against a mock model in under 10 seconds, no GPU
  needed, covering the scenarios listed in step 9 below. Run it after any
  change to `factory/` before burning Colab GPU time on a real test.

What's still open: the live GitHub Pages push (`factory/github.py`) and a
full multi-video soak test both need to run against a real repo/token and
real Colab/Drive to actually verify — that can't be done outside Colab.

- `factory/qwen_client.py` — loads Qwen once in Colab (GPU runtime
  required) and wraps generation with JSON extraction/repair and retry
  logic, since models routinely wrap JSON in prose or markdown fences.
- `factory/research_engine.py` / `fact_checker.py` — generate and review
  the research package. Any claim missing (or with an invalid)
  `classification` defaults to `"uncertain"` rather than ever being
  silently treated as established fact.
- `factory/script_engine.py` — generates the narration script as prose
  from the verified research package.
- `factory/visual_bible.py` — generates topic-specific architecture,
  clothing, materials, and environment from Qwen. `style` and the base
  `lighting` mood are **always** taken from `config.art_style`, never the
  model — series-wide visual consistency is a locked, one-time decision.
  Cached per job so it's computed once, not recomputed on every resume.
- `factory/scene_engine.py` — Qwen handles narration segmentation and
  blocking; `image_prompt` is composed **programmatically** from the
  visual bible + each scene's own fields, guaranteeing every scene
  reliably carries the locked style.
- `factory/image_engine.py` — loads FLUX.1 Schnell (4-step distilled
  model, viable per-scene inside one Colab session) and generates from
  each scene's `image_prompt`. Per-image skip-if-exists + an
  `on_progress` callback so the checkpoint advances after every single
  image, matching the spec's "checkpoint after every image."
- `factory/audio_engine.py` — loads a Piper TTS voice and synthesizes
  per-scene narration audio. Same per-file skip + progress-callback
  pattern as images.
- `factory/video_engine.py` — real FFmpeg pipeline: per-scene Ken Burns
  zoom/pan (matched to each scene's `camera` field) sized to that scene's
  *actual* audio duration, crossfade-concatenated, subtitles burned in
  from scene timing, background music mixed in under narration if a music
  file is present in `05_AUDIO/music/`. Tested locally against real
  ffmpeg — output verified as correct 1920x1080 h264+aac with accurate
  crossfade timing and correctly-timed subtitles.
- `factory/thumbnail_engine.py` — composites the episode title over a
  chosen frame with a legibility gradient, using PIL. Falls back to a
  plain frame copy if anything in the overlay path fails.
- A stage failure anywhere in the pipeline marks the checkpoint
  `status: "error"` with the message, and does **not** mark the topic
  used — re-running picks up and retries that exact job.
- Every engine also has a "no model loaded" fallback path (writes clearly
  labeled placeholders), so the whole pipeline — including checkpointing,
  resume, and now real video/thumbnail rendering — can be exercised and
  tested with zero GPU/model available. Set `USE_QWEN`/`USE_FLUX`/
  `USE_PIPER` to `False` in Colab Cell 4 to use this path selectively.
- External source cross-checking (the academic-papers → museums → ... →
  general-web hierarchy from the spec) is still not wired in — `sources`
  is currently whatever Qwen reports citing, which needs human
  spot-checking until a retrieval step is added.

## Repository layout

```
black-history-factory/
├── README.md
├── requirements.txt
├── .gitignore
├── colab/
│   └── Black_History_Factory.ipynb
├── factory/
│   ├── __init__.py
│   ├── config.py          # loads 00_CONFIG/config.json from Drive
│   ├── drive.py            # Drive path helpers, mount check
│   ├── checkpoint.py       # per-job checkpoint read/write, skip logic
│   ├── status.py           # writes 08_STATUS/current.json + history.json
│   ├── utils.py            # id generation, slugify, json helpers
│   ├── topic_engine.py     # topic DB, dedupe, selection (Phase B — implemented)
│   ├── qwen_client.py      # Qwen loader + JSON-safe generation (Phase C — implemented)
│   ├── research_engine.py  # research generation (Phase C — implemented)
│   ├── fact_checker.py     # classification review (Phase C — implemented)
│   ├── script_engine.py    # narration generation (Phase D — implemented)
│   ├── scene_engine.py     # scene breakdown + image prompts (Phase D — implemented)
│   ├── visual_bible.py     # per-topic visual details, locked style (Phase D — implemented)
│   ├── image_engine.py     # FLUX.1 Schnell image generation (Phase E — implemented)
│   ├── audio_engine.py     # Piper TTS narration audio (Phase E — implemented)
│   ├── video_engine.py     # FFmpeg assembly: pan/zoom, crossfade, subs, music (Phase E — implemented)
│   ├── thumbnail_engine.py # PIL title-card compositing (Phase E — implemented)
│   ├── github.py           # pushes status JSON to the dashboard repo
│   └── main.py             # the job loop / orchestrator
├── prompts/
│   ├── ART_STYLE.md         # the series art style, why, and where it's set in code
│   └── *.txt                # editable prompt templates
├── tests/
│   └── self_test.py        # local no-GPU integration test (Phase H tooling)
└── dashboard/
    ├── index.html
    ├── style.css
    ├── app.js
    └── data/
        ├── current.json
        └── history.json
```

Only `factory/` (code) lives in GitHub. Generated content — images, audio,
video, research JSON — lives on Google Drive under `BLACK_HISTORY_FACTORY/`,
never in the repo (see `.gitignore`).

## Install — step by step

### 1. Create the GitHub repository

1. On GitHub, create a new repo named `black-history-factory` (public if you
   want the dashboard to be visible, private is fine too if you enable Pages
   with a paid plan or use a separate public dashboard-only repo).
2. Clone it locally, then copy in everything from this delivered folder:
   ```bash
   git clone https://github.com/<you>/black-history-factory.git
   cd black-history-factory
   # copy in README.md, requirements.txt, .gitignore, colab/, factory/, prompts/, dashboard/
   git add .
   git commit -m "Phase A: foundation skeleton"
   git push
   ```
3. Enable GitHub Pages: repo **Settings → Pages → Source: Deploy from a
   branch → main / (root or /dashboard)**. If you keep `dashboard/` where it
   is, set the Pages source folder to `/dashboard`. Your dashboard will be
   live at `https://<you>.github.io/black-history-factory/`.

### 2. Create a GitHub token so Colab can push status updates

1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained
   token → grant it **Contents: read and write** on this one repo only.
2. Save the token somewhere safe — you'll paste it into Colab as a secret
   (Colab's "Secrets" panel, key name `GH_TOKEN`), never hard-coded in the
   notebook.

### 3. Create the Google Drive folder structure

In `My Drive`, create `BLACK_HISTORY_FACTORY/` with these subfolders (empty
is fine — the code creates files inside them, but Drive won't auto-create the
folders themselves the first run, so make them once):

```
BLACK_HISTORY_FACTORY/
├── 00_CONFIG/
├── 01_TOPICS/
├── 02_RESEARCH/{raw,verified,sources}/
├── 03_SCRIPTS/{narration,scenes}/
├── 04_IMAGES/{generating,completed}/
├── 05_AUDIO/{narration,music}/
├── 06_VIDEOS/{rendering,completed}/
├── 07_THUMBNAILS/
├── 08_STATUS/
└── 09_LOGS/
```

Quickest way: run this once in a Colab cell after mounting Drive (also in
notebook Cell 1):

```python
import os
base = "/content/drive/MyDrive/BLACK_HISTORY_FACTORY"
for p in ["00_CONFIG","01_TOPICS","02_RESEARCH/raw","02_RESEARCH/verified",
          "02_RESEARCH/sources","03_SCRIPTS/narration","03_SCRIPTS/scenes",
          "04_IMAGES/generating","04_IMAGES/completed","05_AUDIO/narration",
          "05_AUDIO/music","06_VIDEOS/rendering","06_VIDEOS/completed",
          "07_THUMBNAILS","08_STATUS","09_LOGS"]:
    os.makedirs(f"{base}/{p}", exist_ok=True)
```

### 4. Seed the config and topic database

Copy `00_CONFIG/config.json` (included in this delivery) into
`BLACK_HISTORY_FACTORY/00_CONFIG/config.json` on Drive, and put a starter
`topics.json` into `BLACK_HISTORY_FACTORY/01_TOPICS/topics.json` (a small
seed list is included — the topic engine can also generate more over time
once Phase C's AI research engine is wired in).

`config.json` ships with **portrait dimensions** (1080×1920 video,
896×1600 source images — both close to the same aspect ratio, so
`video_engine.py`'s Ken Burns step only lightly crops rather than
reframing) and the locked series **art style** — see
`prompts/ART_STYLE.md` for the full writeup of what it is, why it was
chosen, and the one place in code to change it if you want a different
look. If `config.json` on Drive predates a setting added later (like
`art_style` was, originally), `Config.load()` detects the missing key and
rewrites the file with it automatically — so the file on disk never
silently drifts out of sync with what the code actually uses.

Each seed topic also carries a `description` field — the specific research
angle for that episode (e.g. Great Zimbabwe's entry isn't just "the
walls," it names the mortarless dry-stone technique, the wall dimensions,
and the colonial-era refusal to credit African builders as the actual
angle to research). This flows straight into the research prompt
(`prompts/research.txt`) so Qwen has a concrete angle instead of just a
title to work from. Add this field to any new topics you write by hand,
and the topic-generation prompt (`prompts/topic_generation.txt`) now asks
for it from any AI-generated topics too.

### 5. Open the notebook in Colab

1. **Set the runtime to GPU**: Runtime → Change runtime type → T4 GPU (or
   better). Qwen2.5-7B-Instruct needs ~16GB VRAM in bf16; if you're on a
   free-tier T4 and hit an out-of-memory error, edit Cell 4 to load a
   smaller checkpoint (`QwenClient.load("Qwen/Qwen2.5-3B-Instruct")`).
2. Upload `colab/Black_History_Factory.ipynb` to Colab, or open it straight
   from GitHub (Colab → File → Open notebook → GitHub → paste repo URL).
3. Cell 1 mounts Drive and creates the folder tree if missing.
4. Cell 2 clones/pulls `factory/` from GitHub so the notebook always runs
   the latest code, and installs `requirements.txt`.
5. Cell 3 loads `00_CONFIG/config.json`.
6. **Cell 4 loads Qwen** (set `USE_QWEN = False` at the top of the cell to
   skip this and run everything against placeholders instead — useful for
   testing checkpointing/resume without waiting on model downloads).
7. Cell 5 checks `08_STATUS/current.json` — if a job was mid-flight, it
   resumes that job; otherwise it starts a fresh one.
8. Cell 6 runs `factory.main.run()`, which loops: pick topic → research →
   fact-check → write → plan scenes → visual bible → images → audio →
   render → thumbnail → mark used → status update → push to GitHub → repeat.
9. Cell 7 just prints/pretty-displays `current.json` so you can watch
   progress without leaving the notebook.

With `USE_QWEN = True`, research, fact-checking, narration, visual bible,
and scene planning all run for real. With `USE_FLUX = True` and
`USE_PIPER = True`, images and narration audio are real too, and
`video_engine.py`/`thumbnail_engine.py` always run for real regardless
(they only need ffmpeg/PIL, not a loaded model) — so at this point in the
build order, a full run with all three enabled produces an actual finished
episode end-to-end.

### 8. A note on Piper voices

Piper needs a downloaded voice model (`.onnx` + `.onnx.json`), not just the
Python package. Pick one from the
[Piper voices catalog](https://github.com/rhasspy/piper/blob/master/VOICES.md)
matching `config.language`, download both files to the same path in Colab
(see the commented `wget` lines in Cell 4), and point `PIPER_VOICE_PATH` at
the `.onnx` file.

### 9. Run the local self-test before spending Colab GPU time

```bash
python3 tests/self_test.py
```

Runs the full pipeline against a mock Qwen client (no model download, no
GPU needed) but through the real ffmpeg/PIL video and thumbnail engines.
Covers: a full job completing end-to-end, two sequential jobs getting
distinct ids/topics, resuming a job that "crashed" mid-pipeline, a hard
model failure correctly marking `status: "error"` without consuming the
topic, retrying after that failure resuming the same job, the pipeline
still working with zero models loaded, and the locked art style never
being overridden by a model-supplied value. Runs in under 10 seconds and
never touches your real `00_CONFIG`/`01_TOPICS` — it works in a throwaway
temp copy. Run this after any change to `factory/` before testing against
real Colab.

### 6. Verify resumability (Phase H, step "force-stop / restart")

1. Run the factory, let it get partway through a job (interrupt the Colab
   runtime manually, or use "Manage sessions → Terminate").
2. Reconnect, re-run cells 1–6.
3. Confirm in `current.json` / the printed log that it picked up at the same
   `stage` and `scene`/`current` count instead of restarting the job.

### 7. Plug in the real engines, one at a time

Follow the build order in the spec (Phase C → D → E). Each stub file has a
docstring with its exact expected input and output shape, so replacing a
stub with a real Qwen call, FLUX call, Piper TTS call, or FFmpeg pipeline
should not require touching `main.py`, `checkpoint.py`, or `status.py`.

## Requirements

See `requirements.txt`. Model-specific dependencies (Qwen, FLUX, Piper,
FFmpeg build tools) are commented out until you reach the phase that needs
them, so the base skeleton installs fast.
