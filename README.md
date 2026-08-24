# Black History Factory

A Google Colab-based automated documentary generator: picks an unusual,
non-repeating topic in Black/African history and culture, researches it,
writes a narration, breaks it into scenes, generates images and narration
audio, and renders an MP4 -- checkpointed so it survives Colab disconnects,
with a GitHub Pages dashboard to watch progress from your phone.

## Current state

Every pipeline stage is a real implementation, not a stub:

- **Topics** (`factory/topic_engine.py`) -- database, dedupe, selection.
- **Research** (`factory/research_engine.py`, `fact_checker.py`) --
  Qwen-generated research with fact classification
  (established_fact/archaeological_evidence/scholarly_interpretation/
  oral_tradition/mythology/uncertain). Any claim missing or with an invalid
  classification defaults to `uncertain` rather than being treated as fact.
- **Writing** (`factory/script_engine.py`, `visual_bible.py`,
  `scene_engine.py`) -- narration, a locked series-wide art style (see
  `prompts/ART_STYLE.md`), and scene breakdown with a carefully
  budgeted image-generation prompt (see scene_engine.py's module
  docstring -- this took several rounds to get right, see Troubleshooting).
- **Media** (`factory/image_engine.py`, `audio_engine.py`,
  `video_engine.py`, `thumbnail_engine.py`) -- FLUX or SDXL-Lightning for
  images (pick one in Cell 4), Piper TTS for narration audio, real FFmpeg
  assembly (Ken Burns pan/zoom, crossfades, subtitles, music mixing), PIL
  thumbnail compositing.
- **Job management** (`factory/main.py`, `checkpoint.py`, `status.py`) --
  full checkpoint/resume, per-completed-job manifest archiving, Qwen
  GPU-memory offload/restore around the image-generation stage.
- **Dashboard** (`dashboard/`) -- GitHub Pages status page with
  stalled-session detection (flags a job as "possibly disconnected" if no
  status update in 3+ minutes while still claiming to run).
- **Local self-test** (`tests/self_test.py`) -- runs the whole pipeline
  against a mock model in ~10 seconds, no GPU needed. Run this after any
  code change before spending real Colab GPU time.

## Repository layout

```
black-history-factory/
├── README.md
├── requirements.txt
├── .gitignore
├── colab/Black_History_Factory.ipynb
├── factory/            # all pipeline code -- see docstrings for contracts
├── prompts/             # editable prompt templates + ART_STYLE.md
├── tests/self_test.py   # local no-GPU integration test
├── dashboard/            # GitHub Pages status page
├── 00_CONFIG/config.json # seed config -- copy to Drive once
└── 01_TOPICS/topics.json # small seed topic list -- see note below
```

**Note on topics.json**: the seed file in this repo is intentionally small.
The real, full topic database (hundreds of curated topics) lives on your
Google Drive at `BLACK_HISTORY_FACTORY/01_TOPICS/topics.json` and is
managed separately from this repo -- `git pull` in Cell 2 never touches
your Drive, so your real topic list is safe across code updates.

## Install -- step by step

### 1. Create the GitHub repository, push this code

Create a repo, push everything in this folder to it.

### 2. Create a GitHub token for the dashboard (optional)

GitHub → Settings → Developer settings → Personal access tokens →
Fine-grained token → Contents: read/write on this one repo. Add it to
Colab's Secrets panel (key icon in the sidebar) as `GH_TOKEN`. Also set
`github_repo` in `config.json` to your actual `username/repo-name` --
it ships blank/placeholder by default, and the dashboard silently does
nothing if it's not set correctly.

### 3. Enable GitHub Pages

Repo Settings → Pages → Deploy from a branch → set the folder to
`/dashboard`. Dashboard will be live at
`https://<you>.github.io/black-history-factory/`.

### 4. Seed your Google Drive

Run Cell 1 once to create the folder tree. Then upload your real
`topics.json` to `BLACK_HISTORY_FACTORY/01_TOPICS/topics.json` on Drive
(the repo's seed is just a small placeholder -- see the layout note
above). `config.json` self-heals: `Config.load()` writes it fresh if
missing, and fills in any keys an older file is missing, so you rarely
need to touch it directly.

### 5. Open the notebook in Colab, run Cells 1-7 in order

Cell 4 loads models. Recommended default (see Cell 4's comments): **Qwen
2.5-3B-Instruct + SDXL-Lightning** together total ~13GB, comfortably
fitting a free-tier T4 (14.56GB) with real margin -- see Troubleshooting
below for why this is the recommendation, not just a suggestion.

### 6. Verify resumability

Interrupt execution mid-run (Runtime → Interrupt execution), then re-run
Cells 1, 3, 5, 6 -- Cell 5 should say "Resuming in-progress job" and pick
up from wherever it stopped.

### 7. Run the local self-test before any future code change

```bash
python3 tests/self_test.py
```

No GPU needed, runs in ~10 seconds, covers 8 scenarios including
checkpoint/resume, failure handling, the locked art style never being
overridden, and Qwen's GPU offload firing at the right points.

## Troubleshooting

This section reflects real problems hit while building this project --
each one is a genuine lesson, not hypothetical.

**`credential propagation was unsuccessful` on `drive.mount()`**
A Colab-side auth handshake failure. Try `force_remount=True`; if that
fails, disconnect/delete the runtime and reconnect; check for an ad
blocker or third-party-cookie blocking on `colab.research.google.com`.

**`GatedRepoError` / `401 Unauthorized` loading FLUX**
FLUX.1-schnell is gated -- accept its license on its Hugging Face page,
create a token, add it as a Colab secret named `HF_TOKEN`, **restart the
runtime** (secrets don't retroactively apply mid-session).

**`OutOfMemoryError` loading or running FLUX/Qwen**
The root cause, worked out the hard way over several rounds: FLUX's
12B-parameter transformer alone is ~24GB in bf16 -- more than a T4's
14.56GB by itself, before Qwen is even considered. `enable_model_cpu_offload()`
alone can't fix this (it manages *when* things move between CPU/GPU, not
the size of the thing itself). The real fix was 4-bit quantizing FLUX's
transformer + T5 encoder (see `load_flux()`), or better, switching to
**SDXL-Lightning** (~7GB total, fits without any quantization tricks) --
which is now the recommended default. Separately, Qwen at full precision
(~14-16GB) also doesn't leave room for anything else, which is why
**Qwen 2.5-3B** (not 7B) is the other half of the recommended default --
even at full precision it's only ~6GB, so the combination fits with real
margin instead of depending on quantization working perfectly.

If you still hit OOM with the recommended defaults: restart the runtime
first (a previously-OOM'd session can leave the GPU in a bad state), then
check Cell 4's printed "Qwen GPU memory after load" line -- under 8GB
means quantization/sizing is working as expected.

**A memory bug specific to `load_sdxl_lightning()`**
An earlier version of this function loaded the base SDXL pipeline fully
onto the GPU, THEN loaded the Lightning UNet checkpoint directly onto the
GPU as a second copy before swapping it in -- briefly holding two UNets in
VRAM at once. Fixed by loading the Lightning weights to CPU first and
letting `load_state_dict()` copy them into the already-allocated GPU
parameters in place.

**Image prompts silently losing content / every image looking the same**
`scene_engine.py`'s `_compose_image_prompt()` went through several
corrections: (1) it was originally sized for FLUX's 256-token T5 budget,
which badly overshoots SDXL-Lightning's CLIP-only 77-token limit now that
SDXL is the default; (2) the naive fix used a generic ~1.3 tokens/word
estimate, but this project's actual vocabulary (specialized/foreign terms)
empirically tokenizes at ~2.07 tokens/word -- nearly double; (3) with the
old prompt ordering, EVERY scene-specific detail (location, characters,
objects) was what got silently truncated away, while only the shared style
boilerplate survived, meaning every image in a video looked nearly
identical. Fixed by reprioritizing scene-specific content first, dropping
the least-informative fields entirely, and adding a hard word-count
safety cap verified against real observed tokenization data. Also removed
`"camera movement: X"` from the image prompt entirely -- it was never
consumed by the image model; `video_engine.py` reads `scene["camera"]`
directly for its own zoompan filter after the still image is generated.

**`KeyError: slice(None, 1500, None)` during visual_bible generation**
A genuinely confusing error message: Qwen occasionally returns `overview`
as a nested object instead of plain text, and `overview[:1500]` on a dict
raises this (Python 3.12 made `slice` objects hashable, so a dict slice
now raises `KeyError` instead of the clearer old `TypeError`). Fixed with
explicit type coercion in `research_engine.py`'s `_normalize()` and a
defensive check in `visual_bible.py`.

**Job fails with "No unused topics remain" even though you uploaded topics.json**
Almost always either: (1) the file wasn't uploaded to the exact path
`BLACK_HISTORY_FACTORY/01_TOPICS/topics.json`, (2) Google Drive kept a
duplicate-named file and the mount is resolving to a stale one, or (3)
every topic really is already marked `used: true` from an earlier run.
Diagnose directly in a Colab cell:
```python
from factory.utils import read_json
topics = read_json(paths.topics_json, default=[])
print(len(topics), sum(1 for t in topics if t.get("used")))
```

**"Your disk is almost full"**
Model weights are cached on the Colab runtime's local disk, separate from
Drive. Qwen 3B (~6GB) + SDXL-Lightning (~7GB) is dramatically lighter than
the original Qwen 7B + FLUX combination (~49GB) -- another reason the
recommended default combination helps beyond just VRAM.

## Requirements

See `requirements.txt`.
