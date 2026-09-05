from pathlib import Path
from google.colab import drive


class DrivePaths:
    """
    Centralized Google Drive paths for Black History Factory.

    Every processor uses this class so that all four processors
    agree on exactly where job artifacts are stored.
    """

    SUBFOLDERS = [
        "00_CONFIG",
        "01_TOPICS",
        "02_JOBS",
        "03_AUDIO_LIBRARY/music",
        "03_AUDIO_LIBRARY/ambience",
        "03_AUDIO_LIBRARY/sfx",
        "04_OUTPUT/completed",
        "04_OUTPUT/failed",
        "05_STATUS",
        "06_LOGS",
    ]

    JOB_SUBFOLDERS = [
        "01_research",
        "02_script",
        "03_scenes",
        "04_images",
        "05_audio",
        "06_video",
        "06_video/clips",
        "07_thumbnail",
        "state",
    ]

    def __init__(self, root):
        self.root = Path(root)

    # ------------------------------------------------------------------
    # DRIVE SETUP
    # ------------------------------------------------------------------

    def ensure_tree(self):
        """
        Create the global Black History Factory directory structure.
        Safe to call repeatedly.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        for folder in self.SUBFOLDERS:
            (self.root / folder).mkdir(parents=True, exist_ok=True)

        return self.root

    def prepare_job(self, job_id):
        """
        Create the complete directory structure for one job.

        This is the permanent fix for errors such as:

        FileNotFoundError:
        .../02_JOBS/BH000001/02_script/narration.txt

        Safe to call repeatedly.
        """
        job_root = self.job(job_id)

        job_root.mkdir(parents=True, exist_ok=True)

        for folder in self.JOB_SUBFOLDERS:
            (job_root / folder).mkdir(parents=True, exist_ok=True)

        return job_root

    # ------------------------------------------------------------------
    # JOB ROOT
    # ------------------------------------------------------------------

    def job(self, job_id):
        return self.root / "02_JOBS" / job_id

    # ------------------------------------------------------------------
    # JOB MANIFEST / STATE
    # ------------------------------------------------------------------

    def manifest(self, job_id):
        return self.job(job_id) / "job.json"

    def state(self, job_id):
        return self.job(job_id) / "state"

    # ------------------------------------------------------------------
    # RESEARCH
    # ------------------------------------------------------------------

    def research_dir(self, job_id):
        return self.job(job_id) / "01_research"

    def research(self, job_id):
        return self.research_dir(job_id) / "research.json"

    def sources(self, job_id):
        return self.research_dir(job_id) / "sources.json"

    def verified(self, job_id):
        return self.research_dir(job_id) / "verified.json"

    # ------------------------------------------------------------------
    # SCRIPT / NARRATION
    # ------------------------------------------------------------------

    def script_dir(self, job_id):
        return self.job(job_id) / "02_script"

    def narration(self, job_id):
        return self.script_dir(job_id) / "narration.txt"

    # ------------------------------------------------------------------
    # SCENES
    # ------------------------------------------------------------------

    def scenes_dir(self, job_id):
        return self.job(job_id) / "03_scenes"

    def scenes(self, job_id):
        return self.scenes_dir(job_id) / "scenes.json"

    # ------------------------------------------------------------------
    # IMAGES
    # ------------------------------------------------------------------

    def images_dir(self, job_id):
        return self.job(job_id) / "04_images"

    def image(self, job_id, scene_id):
        return self.images_dir(job_id) / f"{scene_id}.png"

    # ------------------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------------------

    def audio_dir(self, job_id):
        return self.job(job_id) / "05_audio"

    def audio(self, job_id, scene_id):
        return self.audio_dir(job_id) / f"{scene_id}.wav"

    def narration_audio(self, job_id):
        return self.audio_dir(job_id) / "narration.wav"

    # ------------------------------------------------------------------
    # VIDEO
    # ------------------------------------------------------------------

    def video_dir(self, job_id):
        return self.job(job_id) / "06_video"

    def clips_dir(self, job_id):
        return self.video_dir(job_id) / "clips"

    def video(self, job_id):
        return self.video_dir(job_id) / f"{job_id}.mp4"

    # ------------------------------------------------------------------
    # THUMBNAIL
    # ------------------------------------------------------------------

    def thumbnail_dir(self, job_id):
        return self.job(job_id) / "07_thumbnail"

    def thumbnail(self, job_id):
        return self.thumbnail_dir(job_id) / f"{job_id}.jpg"

    # ------------------------------------------------------------------
    # AUDIO LIBRARY
    # ------------------------------------------------------------------

    def music_dir(self):
        return self.root / "03_AUDIO_LIBRARY" / "music"

    def ambience_dir(self):
        return self.root / "03_AUDIO_LIBRARY" / "ambience"

    def sfx_dir(self):
        return self.root / "03_AUDIO_LIBRARY" / "sfx"

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    def completed_dir(self):
        return self.root / "04_OUTPUT" / "completed"

    def failed_dir(self):
        return self.root / "04_OUTPUT" / "failed"

    def completed(self, job_id):
        return self.completed_dir() / f"{job_id}.mp4"

    def failed(self, job_id):
        return self.failed_dir() / f"{job_id}"

    # ------------------------------------------------------------------
    # STATUS / LOGS
    # ------------------------------------------------------------------

    def status_dir(self):
        return self.root / "05_STATUS"

    def current_status(self):
        return self.status_dir() / "current.json"

    def history_status(self):
        return self.status_dir() / "history.json"

    def logs_dir(self):
        return self.root / "06_LOGS"

    def log(self, job_id):
        return self.logs_dir() / f"{job_id}.log"


# ----------------------------------------------------------------------
# GOOGLE DRIVE MOUNT
# ----------------------------------------------------------------------

def mount_drive():
    """
    Mount Google Drive and return the MyDrive path.
    """
    drive.mount("/content/drive")

    mydrive = Path("/content/drive/MyDrive")
    return mydrive
