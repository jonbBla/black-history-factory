from pathlib import Path
import json
from datetime import datetime, timezone

try:
    from google.colab import drive as colab_drive
except Exception:
    colab_drive = None


# ============================================================
# GOOGLE DRIVE
# ============================================================

def mount_drive(mount_point="/content/drive"):
    """
    Mount Google Drive in Colab and return the mounted path.
    """
    if colab_drive is None:
        raise RuntimeError(
            "Google Colab Drive is unavailable. "
            "Run this code inside Google Colab."
        )

    colab_drive.mount(mount_point, force_remount=False)
    return Path(mount_point) / "MyDrive"


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

GLOBAL_FOLDERS = [
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

JOB_FOLDERS = [
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


# ============================================================
# DRIVE PATHS
# ============================================================

class DrivePaths:
    """
    Central path manager for the Black History Factory.

    Every processor should use this class instead of constructing
    Google Drive paths manually.
    """

    def __init__(self, root):
        self.root = Path(root)

    # --------------------------------------------------------
    # Root / global paths
    # --------------------------------------------------------

    @property
    def config_dir(self):
        return self.root / "00_CONFIG"

    @property
    def topics_dir(self):
        return self.root / "01_TOPICS"

    @property
    def jobs_dir(self):
        return self.root / "02_JOBS"

    @property
    def audio_library_dir(self):
        return self.root / "03_AUDIO_LIBRARY"

    @property
    def output_dir(self):
        return self.root / "04_OUTPUT"

    @property
    def status_dir(self):
        return self.root / "05_STATUS"

    @property
    def logs_dir(self):
        return self.root / "06_LOGS"

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    @property
    def config(self):
        return self.config_dir / "config.json"

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    @property
    def topics(self):
        return self.topics_dir / "topics.json"

    @property
    def used_topics(self):
        return self.topics_dir / "used.json"

    @property
    def claimed_topics(self):
        return self.topics_dir / "claimed.json"

    @property
    def rejected_topics(self):
        return self.topics_dir / "rejected.json"

    # --------------------------------------------------------
    # Audio library
    # --------------------------------------------------------

    @property
    def music_library(self):
        return self.audio_library_dir / "music"

    @property
    def ambience_library(self):
        return self.audio_library_dir / "ambience"

    @property
    def sfx_library(self):
        return self.audio_library_dir / "sfx"

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    @property
    def completed_dir(self):
        return self.output_dir / "completed"

    @property
    def failed_dir(self):
        return self.output_dir / "failed"

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    @property
    def status_qwen(self):
        return self.status_dir / "qwen.json"

    @property
    def status_image(self):
        return self.status_dir / "image.json"

    @property
    def status_audio(self):
        return self.status_dir / "audio.json"

    @property
    def status_video(self):
        return self.status_dir / "video.json"

    # Compatibility aliases
    @property
    def status_current(self):
        """
        Compatibility path used by the processors.

        Older versions of the pipeline expected a current.json file.
        Keep it available while the V2 processor-specific status files
        remain the primary status mechanism.
        """
        return self.status_dir / "current.json"

    @property
    def status_history(self):
        return self.status_dir / "history.json"

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    def log(self, name="factory.log"):
        return self.logs_dir / name

    # ========================================================
    # JOB PATHS
    # ========================================================

    def job(self, job_id):
        return self.jobs_dir / str(job_id)

    def prepare_job(self, job_id):
        """
        Create the complete directory tree for a job.

        This MUST be called before any processor attempts to write
        files belonging to a job.
        """
        job_root = self.job(job_id)

        for folder in JOB_FOLDERS:
            (job_root / folder).mkdir(parents=True, exist_ok=True)

        return job_root

    # --------------------------------------------------------
    # Job metadata/state
    # --------------------------------------------------------

    def manifest(self, job_id):
        return self.job(job_id) / "job.json"

    def state(self, job_id):
        return self.job(job_id) / "state"

    def state_file(self, job_id):
        return self.state(job_id) / "state.json"

    # --------------------------------------------------------
    # Research
    # --------------------------------------------------------

    def research_dir(self, job_id):
        return self.job(job_id) / "01_research"

    def research(self, job_id):
        return self.research_dir(job_id) / "research.json"

    def sources(self, job_id):
        return self.research_dir(job_id) / "sources.json"

    def verified(self, job_id):
        return self.research_dir(job_id) / "verified.json"

    # --------------------------------------------------------
    # Script
    # --------------------------------------------------------

    def script_dir(self, job_id):
        return self.job(job_id) / "02_script"

    def narration(self, job_id):
        return self.script_dir(job_id) / "narration.txt"

    def script(self, job_id):
        return self.narration(job_id)

    # --------------------------------------------------------
    # Scenes
    # --------------------------------------------------------

    def scenes_dir(self, job_id):
        return self.job(job_id) / "03_scenes"

    def scenes(self, job_id):
        return self.scenes_dir(job_id) / "scenes.json"

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    def images_dir(self, job_id):
        return self.job(job_id) / "04_images"

    def image(self, job_id, scene_number):
        return self.images_dir(job_id) / f"scene_{int(scene_number):03d}.png"

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    def audio_dir(self, job_id):
        return self.job(job_id) / "05_audio"

    def audio(self, job_id):
        return self.audio_dir(job_id) / "audio.json"

    def narration_audio(self, job_id):
        return self.audio_dir(job_id) / "narration.wav"

    def music_audio(self, job_id):
        return self.audio_dir(job_id) / "music.wav"

    def ambience_audio(self, job_id):
        return self.audio_dir(job_id) / "ambience.wav"

    def sfx_audio(self, job_id):
        return self.audio_dir(job_id) / "sfx.wav"

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    def video_dir(self, job_id):
        return self.job(job_id) / "06_video"

    def clips_dir(self, job_id):
        return self.video_dir(job_id) / "clips"

    def video(self, job_id):
        return self.video_dir(job_id) / "final.mp4"

    def final_video(self, job_id):
        return self.video(job_id)

    # --------------------------------------------------------
    # Thumbnail
    # --------------------------------------------------------

    def thumbnail_dir(self, job_id):
        return self.job(job_id) / "07_thumbnail"

    def thumbnail(self, job_id):
        return self.thumbnail_dir(job_id) / "thumbnail.jpg"

    # ========================================================
    # STATUS HELPERS
    # ========================================================

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _read_json(path, default=None):
        path = Path(path)

        if not path.exists():
            return default

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def _write_json(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(path.suffix + ".tmp")

        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        tmp.replace(path)

    def update_status(
        self,
        processor,
        status,
        current_job=None,
        completed=None,
        extra=None,
    ):
        """
        Update processor status.

        processor:
            qwen / image / audio / video
        """

        processor = str(processor).lower().strip()

        status_paths = {
            "qwen": self.status_qwen,
            "image": self.status_image,
            "audio": self.status_audio,
            "video": self.status_video,
        }

        path = status_paths.get(processor)

        if path is None:
            raise ValueError(
                f"Unknown processor '{processor}'. "
                f"Expected one of: {', '.join(status_paths)}"
            )

        existing = self._read_json(path, {}) or {}

        data = {
            "processor": processor,
            "status": status,
            "current_job": current_job,
            "completed": (
                existing.get("completed", 0)
                if completed is None
                else completed
            ),
            "updated_at": self._now(),
        }

        if extra:
            data.update(extra)

        self._write_json(path, data)

        # Maintain compatibility with older code that expects
        # a shared current.json file.
        current = self._read_json(self.status_current, {}) or {}

        current[processor] = data
        current["updated_at"] = self._now()

        self._write_json(self.status_current, current)

        return data

    def status_current_data(self):
        """
        Return the shared current status object.
        """
        return self._read_json(self.status_current, {})

    def set_status(self, processor, status, job_id=None, **extra):
        """
        Compatibility wrapper.
        """
        return self.update_status(
            processor=processor,
            status=status,
            current_job=job_id,
            extra=extra or None,
        )

    # ========================================================
    # TREE CREATION
    # ========================================================

    def ensure_tree(self):
        """
        Create/verify the complete global Drive structure.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        for folder in GLOBAL_FOLDERS:
            (self.root / folder).mkdir(
                parents=True,
                exist_ok=True,
            )

        # Create the processor status files if they don't exist.
        defaults = {
            self.status_qwen: {
                "processor": "qwen",
                "status": "idle",
                "current_job": None,
                "completed": 0,
                "updated_at": None,
            },
            self.status_image: {
                "processor": "image",
                "status": "idle",
                "current_job": None,
                "completed": 0,
                "updated_at": None,
            },
            self.status_audio: {
                "processor": "audio",
                "status": "idle",
                "current_job": None,
                "completed": 0,
                "updated_at": None,
            },
            self.status_video: {
                "processor": "video",
                "status": "idle",
                "current_job": None,
                "completed": 0,
                "updated_at": None,
            },
        }

        for path, data in defaults.items():
            if not path.exists():
                self._write_json(path, data)

        if not self.status_current.exists():
            self._write_json(
                self.status_current,
                {
                    "qwen": defaults[self.status_qwen],
                    "image": defaults[self.status_image],
                    "audio": defaults[self.status_audio],
                    "video": defaults[self.status_video],
                    "updated_at": None,
                },
            )

        # Topic tracking files.
        topic_defaults = {
            self.topics: [],
            self.used_topics: [],
            self.claimed_topics: {},
            self.rejected_topics: [],
        }

        for path, data in topic_defaults.items():
            if not path.exists():
                self._write_json(path, data)

        return self.root
