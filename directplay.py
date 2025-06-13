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
from typing import List

# ------------------------- pretty logging ------------------------------ #
class EncodeOptions:
    def __init__(self, encode_audio: bool, encode_video: bool, encode_subtitles: bool):
        self.encode_audio = encode_audio
        self.encode_video = encode_video
        self.encode_subtitles = encode_subtitles


class Action(Enum):
    SKIP = "skip"
    REMUX = "remux"
    ENCODE = "encode"


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


# good source for choices https://jellyfin.org/docs/general/clients/codec-support

# ---------------------------- constants ---------------------------------- #

DEFAULT_EXTS = {".mkv", ".mp4"}
IGNORE_SUFFIXES = {'.bak', '.tmp'}  # Files containing these will be skipped
H264_LEVEL_THRESHOLD = 41  # High 4.1 is Chromecast safe
MAX_HEIGHT = 1080  # Down-scale 4 K → 1080p to stay in 4.1
TARGET_CONTAINER = '.mp4'
TARGET_SUBTITLE_FORMAT = 'mov_text'

# ------------------------------------------------------------------------- #


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


# ---------------------- util: decision helpers --------------------------- #

def are_subtitles_ok(streams: List[dict]) -> bool:
    subs = [s for s in streams if s["codec_type"] == "subtitle"]

    if not subs:
        return True  # No subs → no flags needed

    # ok if every sub is already target format
    return all(s["codec_name"] == TARGET_SUBTITLE_FORMAT for s in subs)


def is_h264_ok(v: dict) -> bool:
    return (
            v["codec_name"] == "h264"
            and float(v.get("level", H264_LEVEL_THRESHOLD)) <= H264_LEVEL_THRESHOLD
            and int(v.get("coded_height", MAX_HEIGHT)) <= MAX_HEIGHT
    )


def has_aac_stereo(streams: List[dict]) -> bool:
    for s in streams:
        if s["codec_type"] == "audio" and s["codec_name"] == "aac":
            return True
    return False


def classify(job) -> tuple[Action, EncodeOptions]:
    """
    Decide what to do with `job.src`.
    Returns: Action
    """
    encode_options = EncodeOptions(False, False, False)

    streams = job.meta["streams"]
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    if v is None:  # no video stream at all
        log("⚠", C.YEL, "No video stream; skipping", job.src)
        return Action.SKIP, encode_options

    if job.opts.skip_audio is False and not has_aac_stereo(streams):
        encode_options.encode_audio = True

    if job.opts.skip_video is False and not is_h264_ok(v):
        encode_options.encode_video = True

    if job.opts.skip_subtitles is False and not are_subtitles_ok(streams):
        encode_options.encode_subtitles = True

    if job.src.suffix.lower() == TARGET_CONTAINER and not encode_options.encode_video and not encode_options.encode_audio and not encode_options.encode_subtitles:
        return Action.SKIP, encode_options

    # Some remux or transcode to do
    return Action.ENCODE, encode_options


# ----------------------------- job type ---------------------------------- #


class Job:
    def __init__(self, src: Path, opts):
        self.src = src
        self.opts = opts
        try:
            self.meta = ffprobe(src)
            self.action, self.encoder_options = classify(self)
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
                    self.action, self.encoder_options = classify(self)  # try again
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
                    self.action, self.encoder_options = classify(self)  # try again
                else:
                    print(f"⚠️  Remux failed, skipping: {src}")
                    self.action = Action.SKIP
            else:
                print(f"⚠️  Unreadable or non-video file, skipping {src}")
                self.action = Action.SKIP

        self.dst = src.with_suffix(TARGET_CONTAINER) if self.action != Action.SKIP else None


# --------------------------- worker logic -------------------------------- #


def looks_ok(path: Path) -> bool:
    """
    Return True only if `path` is a playable MKV that already
    satisfies the direct-play profile (container + codecs).
    """
    try:
        meta = ffprobe(path)
        streams = meta["streams"]
        v = next((s for s in streams if s["codec_type"] == "video"), None)
        if v is None:
            return False

        container_ok = path.suffix.lower() == TARGET_CONTAINER
        video_ok = is_h264_ok(v)
        audio_ok = has_aac_stereo(streams)

        return container_ok and video_ok and audio_ok
    except subprocess.CalledProcessError:
        return False


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

    if job.encoder_options.encode_subtitles:
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
            TARGET_SUBTITLE_FORMAT,
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


def convert_layout_for_libfdk_aac(layout) -> str:
    return {
        "5.1(side)": "5.1",
        # ... Will need more for other layouts
    }.get(layout, layout)

def get_audio_flags(job: Job) -> list:
    """Returns FFmpeg flags based existing audio streams"""

    flags = []

    flags.extend(
        [
            "-map",
            "0:a:0",
        ]
    )
    if job.encoder_options.encode_audio:
        audio_stream = next((s for s in job.meta["streams"] if s["codec_type"] == "audio"), None)
        if audio_stream:
            channels = audio_stream.get("channels", 2)
            channel_layout = audio_stream.get("channel_layout", "")
            channel_layout = convert_layout_for_libfdk_aac(channel_layout)

            print(f"Encoding audio with {channels} channels & layout {channel_layout}")
            flags.extend([
                "-c:a:0", "libfdk_aac",
                "-b:a", get_bitrate(channels),  # Dynamic bitrate based on channels
                "-ac:0", str(channels),
                *(["-channel_layout", channel_layout] if channel_layout else []),
                "-metadata:s:a:0", f"channels={channels}"
            ])
            # test with ffprobe -v error -show_entries stream=channel_layout,channels -of csv=p=0 <file>
            # currently channels is borked
        else:
            print(f"Failed to get audio streams for {job.src}")
    else:
        flags.extend(
            [
                "-c:a:0",
                "copy",
            ]
        )

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

    return flags


def build_encode_cmd(job: Job):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-y",
    ]

    v = next(s for s in job.meta["streams"] if s["codec_type"] == "video")
    codec_name = v.get("codec_name")

    print(f'Found video codec {codec_name}')
    # Only use CUDA hwaccel if input is H.264 (AV1 on hardware will fail)
    if job.opts.encoder == "h264_nvenc":
        if codec_name == "h264" or codec_name == "hevc":
            cmd.extend(
                [
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_output_format",
                    "cuda",
                ]
            )

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

    if job.encoder_options.encode_video:
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
                vfilters.append(f"scale=-2:{MAX_HEIGHT}")

        vf = ",".join(vfilters) if vfilters else None

        # -------- encoder-specific branch --------
        if job.opts.encoder == "libx264":
            video_flags = [
                "-c:v",
                "libx264",
                "-preset",
                job.opts.preset,
                "-crf",
                str(job.opts.crf),
                "-pix_fmt",
                "yuv420p",  # Enforce 8bit format for firefox compatibility
            ]
        elif job.opts.encoder == "h264_nvenc":
            video_flags = [
                # --- NVENC encode flags ---
                "-c:v",
                "h264_nvenc",
                "-preset",
                job.opts.nvenc_preset,
                "-rc",
                "vbr",
                "-tune",
                "hq",
                # pix_fmt handled by format filter above
                "-cq",
                str(job.opts.crf),
                "-b:v",
                "0",
            ]

        else:
            log("✗", C.RED, f"Unknown encoder ({job.opts.encoder})", job.src)
            exit(1)

        cmd.extend(
            [
                *video_flags,
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
    if job.dst.exists():
        if looks_ok(job.dst):
            log("✓", C.GRN, "Output exists; skipping", job.dst)
            return
        elif is_stub(job.dst):
            log("⚠", C.YEL, "Stub file detected; will overwrite", job.dst)
            job.dst.unlink()

    # 3) Build the ffmpeg command
    if job.action == Action.ENCODE:
        cmd, tmp_out = build_encode_cmd(job)

    log_action = str(job.action)
    if job.action == Action.ENCODE:
        log_action += f" (video: {job.encoder_options.encode_video}, audio: {job.encoder_options.encode_audio}, subtitles: {job.encoder_options.encode_subtitles})"

    log("→", C.CYN, log_action, job.src)

    # 4) Run ffmpeg, but don't let a non-zero exit kill the whole batch
    try:
        print(f"FFmpeg command: {' '.join(cmd)}")
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
        "--crf",
        type=int,
        default=19,
        help="x264 Constant Rate Factor (lower = higher quality).",
    )
    p.add_argument(
        "-d",
        "--delete-original",
        action='store_true',
        default=False,
        help="Remove source file after successful processing.",
    )
    p.add_argument(
        "--hvenc",
        choices=("true", "false"),
        default="false",
        help="Use NVIDIA hvenc if possible.",
    )
    p.add_argument(
        "--preset",
        default="medium",
        help="preset: ultrafast…slow. Will be converted to p* for h264_nvenc",
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
    opts = p.parse_args()

    # -- Make sure ffmpeg exists BEFORE calling it -------------
    if not shutil.which("ffmpeg"):
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
    opts.encoder = "h264_nvenc" if (opts.hvenc is True and "h264_nvenc" in encoders) else "libx264"

    # quick helper for preset mapping when NVENC is active
    def _nv_map(x):
        map_ = {
            "ultrafast": "p1",
            "superfast": "p2",
            "veryfast": "p3",
            "faster": "p3",
            "fast": "p4",
            "medium": "p4",
            "slow": "p5",
            "slower": "p6",
            "veryslow": "p7",
        }
        if x not in map_:
            raise ValueError(
                f"Unsupported preset for NVENC conversion: '{x}'. "
                f"Valid options are: {', '.join(map_.keys())}"
            )
        return map_[x]  # Explicit dict access now safe

    # Add the NVENC preset mapping to the options
    opts.nvenc_preset = _nv_map(opts.preset)

    exts = {e if e.startswith(".") else f".{e}" for e in opts.exts.split(",")}

    print("Scanning …")
    files = gather_files(opts.roots, exts, opts.limit)
    print(f"Found {len(files)} candidate files.\n")

    with ProcessPoolExecutor(max_workers=opts.workers) as executor:
        jobs = [Job(f, opts) for f in files]
        results = executor.map(process, jobs)
        for result in results:
            print(result)

    print("\nDone.")


if __name__ == "__main__":
    main()
