Black History Factory — Qwen pipeline replacement files

Replace these files in the repository:
factory/qwen_pipeline.py
factory/research_engine.py
factory/fact_checker.py
factory/script_engine.py
factory/scene_engine.py
factory/status.py
00_CONFIG/config.json

Notebook Cell 3 (Qwen model load) does not need changing.
Notebook Cell 4 does not need changing.

Important: Config.load() uses the canonical config at
Google Drive/MyDrive/BLACK_HISTORY_FACTORY/00_CONFIG/config.json.
If that file already exists, changing the repository config alone will NOT
change the active Drive settings. Set scene_count_min=18, scene_count_max=22,
narration_words_min=170, narration_words_max=220 there as well.

The new Qwen pipeline is:
source search -> Qwen evidence dossier -> Qwen fact validation -> visual bible
-> Qwen narration (6 attempts) -> Qwen intelligent scene plan -> Qwen image
prompt per scene -> scenes.json.

The scene planner does NOT use deterministic Python segmentation.
It validates that Qwen's scene narration uses the same words in the same order.
Each scene must be 4-14 words and the total scene count is 18-22.

Scene output includes both scene_id and image_prompt so the existing Image,
Audio and Video notebooks can consume it.
