# Black History Factory — 4 Independent Colab Processors

This version is deliberately split into four independent processors. The old monolithic runner/checkpoint architecture has been removed.

## Pipeline

1. **Qwen Processor** — research → fact check → visual bible → narration → scene planning → `QWEN_READY`
2. **Image Processor** — generate missing scene images → `IMAGES_READY`
3. **Audio Processor** — generate scene narration WAVs → duration check → `AUDIO_READY`
4. **Video Processor** — motion → subtitles → optional music → black source card → final duration check → `COMPLETED`

Every processor writes its current stage to `06_STATUS/current.json` and prints the same information in Colab. Existing files are reused after a reset, so a processor can continue instead of starting its stage from zero.

The final video does **not** include the creator/self clip. It ends with a black source card.

## First test

Keep `prepared_job_target` at `1`. Run one job through all four notebooks and inspect the resulting video. Only after the first video is acceptable should you change it to `40` and use the loop cells.

## Drive

```text
BLACK_HISTORY_FACTORY/
├── 00_CONFIG/config.json
├── 01_TOPICS/topics.json
├── 01_TOPICS/used_topics.json
├── 01_TOPICS/rejected_topics.json
├── 02_JOBS/BH000001/
│   ├── job.json
│   ├── state/
│   ├── 01_research/
│   ├── 02_script/
│   ├── 03_scenes/
│   ├── 04_images/
│   ├── 05_audio/
│   └── 06_video/
├── 04_AUDIO_LIBRARY/music/
├── 04_AUDIO_LIBRARY/ambience/
├── 04_AUDIO_LIBRARY/sfx/
├── 05_OUTPUT/completed/
├── 05_OUTPUT/failed/
├── 06_STATUS/current.json
└── 07_LOGS/
```

## Important accuracy limitation

The Qwen fact checker is still an LLM review. It does **not** independently verify claims against live external sources. Do not treat the generated documentary as publication-grade historical research until a retrieval/source-verification stage is added.

## Qwen model

The project defaults to `Qwen/Qwen2.5-1.5B-Instruct`, which is intentionally small for free Colab. The model is officially supported through Transformers and has structured-data/JSON capabilities. citeturn0search0
