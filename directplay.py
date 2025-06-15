#!/usr/bin/env python3

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Dict, List

# ------------------------- pretty logging ------------------------------ #


class Action(Enum):
    SKIP = "skip"
    ENCODE = "encode"


class EncodeOptions:
    def __init__(self, audio: bool = False, video: bool = False, subtitles: bool = False):
        self.audio: bool = audio
        self.video: bool = video
        self.subtitles: bool = subtitles

    def has_any_encode(self) -> bool:
        return self.audio or self.video or self.subtitles


class C:
    RESET = "\033[0m"
    GRN = "\033[32m"
    CYN = "\033[36m"
    YEL = "\033[33m"
    RED = "\033[31m"


def log(symbol: str, color: str, action: str, path: Path):
    print(f'{color}{symbol} {action}: "{path}"{C.RESET}')


def get_script_dir():
    """Returns the directory containing the current script"""
    return Path(__file__).parent.resolve()


def setup_ffmpeg_path():
    """Prepend script directory to PATH"""
    script_dir = str(get_script_dir())
    os.environ["PATH"] = script_dir + os.pathsep + os.environ["PATH"]
    print(f"Added script directory to PATH: {script_dir}")


# Docs:
# For codec supports overview https://jellyfin.org/docs/general/clients/codec-support
# https://trac.ffmpeg.org/wiki/Encode/AV1
# `ffmpeg -h encoder=av1_nvenc`


# ---------------------------- constants ---------------------------------- #

DEFAULT_EXTS = {".mkv", ".mp4", ".webm"}
IGNORE_SUFFIXES = {'.bak', '.tmp'}  # Files containing these will be skipped
H264_LEVEL_THRESHOLD = 41  # High 4.1 is Chromecast safe
MAX_HEIGHT = 1080  # Down-scale 4 K → 1080p to stay in 4.1

# target settings, move those to command line?
TARGET_CONTAINER: str = '.webm'
DEFAULT_VIDEO_ENCODER: str = "av1_nvenc"
TARGET_SUBTITLE_FORMAT: str = 'webvtt'
TARGET_AUDIO_ENCODER: str = 'libopus'

# ------------------------------------------------------------------------- #


class EncoderConfig:
    def __init__(self, codec_name: str, options: List[str]):
        self.codec_name = codec_name
        self.options = options


class DefaultEncoderConfigManager:
    # Static data array
    ENCODER_CONFIGS: Dict[str, EncoderConfig] = {
        "libx264": EncoderConfig(
            codec_name="h264",
            options=[
                "-preset", "medium",
                "-crf", "19", 
                "-pix_fmt", "yuv420p"  # Enforce 8bit format for firefox compatibility
            ] 
        ),
        "h264_nvenc": EncoderConfig(
            codec_name="hevc",  # or h264?
            options=[ 
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "0"
                "-tune", "hq",
                # color format is handled in build_encode_cmd for now for nvenc
            ]
        ),
        #  ffmpeg -h encoder=av1_nvenc
        "av1_nvenc": EncoderConfig(
            codec_name="av1",
            options=[
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "0",
                "-tune", "hq",
                "-highbitdepth", "true",
                "-multipass", "qres",
            ]
        ),
        "libopus": EncoderConfig(
            codec_name="opus",
            options=[
                "-b:a", "192k" # assuming stereo
            ]
        ),
    }

    @staticmethod
    def get(encoder_name: str) -> EncoderConfig:
        if encoder_name not in DefaultEncoderConfigManager.ENCODER_CONFIGS:
            raise ValueError(f"Encoder '{encoder_name}' not found in the configuration")

        return DefaultEncoderConfigManager.ENCODER_CONFIGS[encoder_name]
    
    @staticmethod
    def get_codec_name(encoder_name: str) -> str:
        if encoder_name not in DefaultEncoderConfigManager.ENCODER_CONFIGS:
            raise ValueError(f"Encoder '{encoder_name}' not found in the configuration")
        
        return DefaultEncoderConfigManager.ENCODER_CONFIGS[encoder_name].codec_name

    @staticmethod
    def get_supported_encoders() -> List[str]:
        return list(DefaultEncoderConfigManager.ENCODER_CONFIGS.keys())


def run(cmd: List[str]) -> str:
    """Run command, return stdout text, raise on error."""
    return subprocess.check_output(cmd, text=True)


def ffprobe(path: Path) -> dict:
    """Return ffprobe metadata for video & audio streams."""
    return json.loads(
        run(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_streams",
                "-print_format",
                "json",
                str(path),
            ]
        )
    )


class StreamValidator:

    @staticmethod
    def are_subtitles_ok(streams: List[dict]) -> bool:
        subs = [s for s in streams if s["codec_type"] == "subtitle"]

        if not subs:
            return True  # No subs → no flags needed

        # ok if every sub is already target format
        return all(s["codec_name"] == TARGET_SUBTITLE_FORMAT for s in subs)

    @staticmethod
    def _validate_h264(v: dict) -> bool:
        return (
            float(v.get("level", H264_LEVEL_THRESHOLD)) <= H264_LEVEL_THRESHOLD
            and int(v.get("coded_height", MAX_HEIGHT)) <= MAX_HEIGHT
        )

    @staticmethod
    def _validate_av1(v: dict) -> bool:
        return (
            float(v.get("level", H264_LEVEL_THRESHOLD)) <= H264_LEVEL_THRESHOLD
            and int(v.get("coded_height", MAX_HEIGHT)) <= MAX_HEIGHT
        )

    @staticmethod
    def is_video_ok(encoder: str, v: dict) -> bool:
        codec_name = v.get("codec_name")
        if codec_name is None:
            raise ValueError("codec_name is missing in the video info")

        if codec_name != DefaultEncoderConfigManager.get_codec_name(encoder):
            return False

        # Dictionary mapping codec names to their validation functions
        validation_functions = {
            "h264": StreamValidator._validate_h264,
            "av1": StreamValidator._validate_av1,
        }

        # Get the validation function for the given codec
        validate = validation_functions.get(codec_name)

        if validate:
            return validate(v)
        else:
            raise ValueError(f"Target codec is not supported by this script: {codec_name}")

    @staticmethod
    def is_audio_ok(streams: List[dict]) -> bool:
        for s in streams:
            if s["codec_type"] == "audio" and s["codec_name"] == DefaultEncoderConfigManager.get_codec_name(TARGET_AUDIO_ENCODER):
                return True
            
        return False


class Job:
    def __init__(self, src: Path, opts):
        self.src = src
        self.opts = opts
        try:
            self.meta = ffprobe(src)
            self.action, self.encode_options = self.classify()
        except subprocess.CalledProcessError:
            # Attempt auto-repair for MP4/MOV/M4V files with corrupted headers
            if src.suffix.lower() in {".mp4", ".m4v", ".mov"}:
                print(f"⚠️  Attempting to repair corrupted MP4 header: {src}")
                tmp = src.with_suffix(".tmp.mp4")
                repair = [
                    "ffmpeg",
                    "-y",
                    "-err_detect",
                    "ignore_err",
                    "-i",
                    str(src),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(tmp),
                ]
                if subprocess.run(repair).returncode == 0 and tmp.stat().st_size > 0:
                    tmp.rename(src)  # swap in the repaired file
                    print(f"✓  Successfully repaired: {src}")
                    self.meta = ffprobe(src)
                    self.action, self.encode_options = self.classify()  # try again
                else:
                    print(f"⚠️  Repair failed, skipping: {src}")
                    self.action = Action.SKIP
            # For MKV/WebM, try a simple remux
            elif src.suffix.lower() in {".mkv", ".webm"}:
                print(f"⚠️  Attempting to remux corrupted container: {src}")
                tmp = src.with_suffix(".tmp" + src.suffix)
                remux = ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(tmp)]
                if subprocess.run(remux).returncode == 0 and tmp.stat().st_size > 0:
                    tmp.rename(src)  # swap in the remuxed file
                    print(f"✓  Successfully remuxed: {src}")
                    self.meta = ffprobe(src)
                    self.action, self.encode_options = self.classify()  # try again
                else:
                    print(f"⚠️  Remux failed, skipping: {src}")
                    self.action = Action.SKIP
            else:
                print(f"⚠️  Unreadable or non-video file, skipping {src}")
                self.action = Action.SKIP

        self.dst: Path = src.with_suffix(TARGET_CONTAINER)

    def classify(self) -> tuple[Action, EncodeOptions]:
        """
        Decide what to do with `job.src`.
        """
         
        options = EncodeOptions(False, False, False)

        streams = self.meta["streams"]
        v = next((s for s in streams if s["codec_type"] == "video"), None)
        if v is None:  # no video stream at all
            log("⚠", C.YEL, "No video stream; skipping", self.src)
            return Action.SKIP, options

        if self.opts.skip_audio is False and not StreamValidator.is_audio_ok(streams):
            options.audio = True

        if self.opts.skip_video is False and not StreamValidator.is_video_ok(self.opts.encoder, v):
            options.video = True

        if self.opts.skip_subtitles is False and not StreamValidator.are_subtitles_ok(streams):
            options.subtitles = True

        if self.src.suffix.lower() == TARGET_CONTAINER and not options.has_any_encode():
            return Action.SKIP, options

        # Some remux or transcode to do
        return Action.ENCODE, options


# ---------------------------------------------------------------------- #

# helper: detect obvious stubs (0-byte or < 1 MiB)
def is_stub(p: Path) -> bool:
    try:
        return p.stat().st_size < 1 * 1024 * 1024  # < 1 MiB
    except FileNotFoundError:
        return True


def get_subtitle_flags(job: Job) -> list:
    """Returns FFmpeg flags based existing subtitle streams"""

    flags = []

    if TARGET_CONTAINER == ".webm" and TARGET_SUBTITLE_FORMAT != "webvtt":
        raise ValueError(f"Unsupported subtitle format for webm container: '{TARGET_SUBTITLE_FORMAT}'")

    if job.encode_options.subtitles:
        flags.extend([
            "-map",
            "0:s?",
            "-c:s",
            TARGET_SUBTITLE_FORMAT,
            "-ignore_unknown"
        ])
    else:
        # That's fine if there are none
        flags.extend([
            "-map",
            "0:s?",
            "-c:s",
            "copy",
            "-ignore_unknown"
        ])

    return flags


def get_bitrate(channels: int) -> str:
    """Returns appropriate AAC bitrate based on channel count"""
    return {
        1: "96k",  # Mono
        2: "128k",  # Stereo
        6: "384k",  # 5.1
        8: "512k"  # 7.1
    }.get(channels, f"{channels * 64}k")  # Default: 64k per channel


def get_audio_flags(job: Job) -> list:
    """Returns FFmpeg flags based existing audio streams"""

    flags = []

    flags.extend(
        [
            "-map",
            "0:a:0",
        ]
    )
    if job.encode_options.audio:
        audio_stream = next((s for s in job.meta["streams"] if s["codec_type"] == "audio"), None)
        if audio_stream:
            
            encoder_default_config = DefaultEncoderConfigManager.get(job.opts.encoder)

            # always downmix to stereo

            flags.extend([
                "-c:a", TARGET_AUDIO_ENCODER,
                "-ac", "2",
            ])
            flags.extend(encoder_default_config.options)
            # Test channel layout: with ffprobe -v error -show_entries stream=channel_layout,channels -of csv=p=0 <file>
        else:
            print(f"Failed to get audio streams for {job.src}")

         
        # just copy other streams if any
        flags.extend(
            [
                "-map",
                "0:a:1?",
                "-c:a:1",
                "copy",
                "-map",
                "0:a:2?",
                "-c:a:2",
                "copy",
            ]
        )
    else:
        flags.extend(
            [
                "-c:a",
                "copy",
            ]
        )

    return flags


def build_encode_cmd(job: Job):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-y",
        "-hwaccel",
        "auto",
    ]

    v = next(s for s in job.meta["streams"] if s["codec_type"] == "video")
    codec_name = v.get("codec_name")

    print(f'Found video codec {codec_name}')

    cmd.extend([
        "-fflags",
        "+genpts+fastseek",
        "-i",
        str(job.src),
        "-movflags",
        "+faststart",
        "-g",
        "60",
        "-map",
        "0:v:0",
    ])

    if job.encode_options.video:
        vfilters = []

        need_downscale = int(v.get("coded_height", 0)) > MAX_HEIGHT

        # Only use the CUDA scaler if we also decoded on CUDA (i.e. input was H.264)
        if job.opts.encoder == "h264_nvenc" and (codec_name == "h264" or codec_name == "hevc"):
            if need_downscale:
                vfilters.append(f"scale_cuda=-2:{MAX_HEIGHT}:format=yuv420p")
            else:
                vfilters.append("scale_cuda=format=yuv420p")
        else:
            # either CPU‐only path or non‐H.264 input → use normal scale
            if need_downscale:
                vfilters.append(f'scale=-2:{MAX_HEIGHT}')

        vf = ",".join(vfilters) if vfilters else None

        encoder_default_config = DefaultEncoderConfigManager.get(job.opts.encoder)

        cmd.extend(
            [
                "-c:v",
                job.opts.encoder,
                *encoder_default_config.options,
                # Let ffmpeg decide level
                # "-level",
                # "4.1",
            ]
            + (["-vf", vf] if vf else [])
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "copy",
            ]
        )

    if job.opts.sample != 0:
        cmd.extend(
            [
                "-t",
                f"{job.opts.sample}",
            ]
        )

    srt_file = job.src.with_suffix(".srt")
    if srt_file.exists():
        print(f'Found subtitle {srt_file}')
        cmd.extend([
            "-i", str(srt_file),
            "-map", "1:s",
        ])

    cmd.extend(get_audio_flags(job))
    cmd.extend(get_subtitle_flags(job))

    tmp_out = job.dst.with_stem(job.dst.stem + ".tmp")
    cmd.extend(
        [
            str(tmp_out),
        ]
    )
    return cmd, tmp_out


def process(job: Job):
    """
    Process a single Job:
      - skip if already compliant
      - remux or encode otherwise
      - on any ffmpeg error, log & return (i.e. skip the bad file)
    """

    print(f"Job current working directory: {shlex.quote(os.getcwd())}")

    # 1) If we don't need to touch it, skip early
    if job.action == Action.SKIP:
        log("✓", C.GRN, "Direct-play already supported", job.src)
        return
    
    # 2) If dst exists and is good, skip; if stub or bad, prepare to overwrite
    if job.dst.exists() and is_stub(job.dst):
        log("⚠", C.YEL, "Stub file detected; will overwrite", job.dst)
        job.dst.unlink()

    cmd: list[str] = []
    tmp_out: Path | None = None
    
    # 3) Build the ffmpeg command
    if job.action == Action.ENCODE:
        cmd, tmp_out = build_encode_cmd(job)

    log_action = str(job.action)
    if job.action == Action.ENCODE:
        log_action += f" (video: {job.encode_options.video}, audio: {job.encode_options.audio}, subtitles: {job.encode_options.subtitles})"

    log("→", C.CYN, log_action, job.src)

    # 4) Run ffmpeg, but don't let a non-zero exit kill the whole batch
    try:
        for x in cmd:
            if type(x) != str:
                print(f"Found non string value {x} in cmd")

        print_cmd = ' '.join(cmd)
        print(f"FFmpeg command: {print_cmd}")
        if job.opts.debug is True:
            with open(str(job.dst) + ".txt", "w") as f:
                f.write(print_cmd)

        subprocess.run(cmd, check=True, cwd=os.getcwd())
    except subprocess.CalledProcessError as e:
        # Log the failure and skip to the next file
        log("✗", C.RED, f"FFmpeg failed ({e.returncode}); skipping", job.src)
        return

    # 5) On success: atomic swap & optional delete
    if tmp_out and tmp_out.exists():
        if job.dst.exists():
            bak_file = job.dst.with_stem(job.dst.stem + ".bak")
            job.dst.replace(bak_file)
            log("⚠", C.YEL, "Existing file will renamed to", bak_file)

        tmp_out.rename(job.dst)

    if job.opts.delete_original and job.action == Action.ENCODE and job.src != job.dst:
        job.src.unlink()
        log("✘", C.RED, "Deleted original", job.src)


# ----------------------------- main -------------------------------------- #


def gather_files(roots, exts, limit: int) -> List[Path]:
    files = []

    for root in roots:
        for p in Path(root).rglob("*"):
            # skip dot-files created by macOS (._foo, .DS_Store, etc.)
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts and not any(
                    ignore in p.name for ignore in IGNORE_SUFFIXES):
                files.append(p)
                if limit != 0 and len(files) >= limit:
                    return files

    return files


def main():
    setup_ffmpeg_path()
    p = argparse.ArgumentParser(description="Pre-encode / remux videos for Chromecast.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("roots", nargs="+", help="Root directories to scan.")
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel workers.",
    )
    p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Max videos to process before stopping. 0 means means unlimited.",
    )
    p.add_argument(
        "-d",
        "--delete-original",
        action='store_true',
        default=False,
        help="Remove source file after successful processing.",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_VIDEO_ENCODER,
        help=f"Specify which encoder to use. Supported encoders are: {DefaultEncoderConfigManager.get_supported_encoders()}",
    )
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Presets, has to match the encoder. (ex for xh264: ultrafast…slow)",
    )
    p.add_argument(
        "--skip-video",
        "-sv",
        action='store_true',
        default=False,
        help="Never transcode video.",
    )
    p.add_argument(
        "--skip-subtitles",
        "-ss",
        action='store_true',
        default=False,
        help="Never transcode subtitles.",
    )
    p.add_argument(
        "--skip-audio",
        "--sa",
        action='store_true',
        default=False,
        help="Never transcode video.",
    )
    p.add_argument(
        "--exts",
        default=",".join(DEFAULT_EXTS),
        help="Comma-separated list of file extensions to consider.",
    )
    p.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Create a .txt file next to destination with ffmpeg command.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Create a segment sample of given length in second.",
    )
    opts = p.parse_args()

    # -- Make sure ffmpeg exists BEFORE calling it -------------
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        os_hint = {
            "Windows": "winget install Gyan.FFmpeg",
            "Darwin": "brew install ffmpeg",
            "Linux": "sudo apt install ffmpeg   # Debian/Ubuntu\n"
                     "  sudo dnf install ffmpeg   # Fedora/RHEL",
        }.get(platform.system(), "Install ffmpeg from your package manager")

        msg = textwrap.dedent(
            f"""
            ffmpeg/ffprobe not found in PATH.

            Quick install:
            {os_hint}

            Then re-run: {' '.join(map(str, sys.argv))}
        """
        ).strip()
        raise SystemExit(msg)

    # -- Detect hardware encoder once --------------------------
    encoders = run(["ffmpeg", "-v", "quiet", "-encoders"])
    if opts.encoder not in encoders:
        raise ValueError(f"Provided encoder {opts.encoder} does not exists or is not supported by current ffmpeg installation", f"Valid encoders are {encoders}")
    
    exts = {e if e.startswith(".") else f".{e}" for e in opts.exts.split(",")}

    print("Scanning …")
    files = gather_files(opts.roots, exts, opts.limit)
    print(f"Found {len(files)} candidate files.\n")

    with ProcessPoolExecutor(max_workers=opts.workers) as executor:
        jobs = [Job(f, opts) for f in files]
        results = executor.map(process, jobs)
        # print errors if any
        for result in results:
            print(result)

    print("\nDone.")


if __name__ == "__main__":
    main()
