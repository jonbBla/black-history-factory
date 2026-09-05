# Simplified Qwen replacement package

## Replace
- factory/qwen_pipeline.py
- factory/research_engine.py
- factory/fact_checker.py
- factory/script_engine.py
- factory/scene_engine.py
- factory/qwen_client.py
- factory/image_engine.py
- factory/topic_engine.py
- 00_CONFIG/config.json
- prompts/research.txt
- prompts/fact_check.txt
- prompts/narration.txt
- prompts/scene_planning.txt
- prompts/image_prompt.txt
- colab/01_Qwen_Processor.ipynb

## Remove
- factory/visual_bible.py
- factory/checkpoint.py
- factory/main.py
- prompts/visual_bible.txt
- factory/_pycache_/ (entire directory)

## Keep
Keep the independent downstream system: audio_engine.py, video_engine.py, source_card.py, thumbnail_engine.py, drive.py, config.py, status.py, github.py, utils.py, and the Image/Audio/Video notebooks.

## Output contract
`02_JOBS/<JOB>/03_scenes/scenes.json` is a JSON LIST. Every scene contains:
`scene_id`, `scene_number`, `narration`, `word_count`, `visual_description`, `image_prompt`, `image_description`, `camera`.

The Image Processor reads `visual_description` directly and generates only missing scene images.

## Important
The repository config controls defaults, but `Config.load()` loads the existing Drive config if it already exists. If your Drive has an older `00_CONFIG/config.json`, replace/update that Drive copy too, otherwise its old duration/scene limits will override the repository defaults.

After replacing the files in a running Colab, restart the runtime before testing, or use the cache-clearing import cell. Test one job until it reaches `QWEN_READY` before enabling the multi-job loop.
