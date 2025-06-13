#!/usr/bin/env python3
"""
encode.py
=============

Batch-optimise any video library so every file direct-plays on
Chromecast, smart-TVs and desktop/mobile browsers — no server-side
transcoding required.

Key features
------------
• Smart decision tree – tries lightning-fast remux first; falls back
  to H.264 re-encode only when codecs or resolution exceed safe limits.
• Recursive scan – pass one or more root folders; sub-directories
  handled automatically.
• Parallel workers (-j / --workers) – use all CPU cores by default or
  limit to keep the box responsive.
• Quality knobs – --crf and --preset expose x264 controls; defaults
  tuned for "Netflix-like" 1080 p quality.
• Safe by default – outputs <name>.mkv in the same folder; originals
  retained unless you add --delete-original true.
• Idempotent – re-runs skip files already compliant; perfect for
  nightly cron/systemd-timer jobs.

Dependencies
------------
ffmpeg and ffprobe must be in your PATH.
OR it will use the ones next to this file if any.

  Windows 10/11 : winget install Gyan.FFmpeg
  macOS (brew)  : brew install ffmpeg
  Debian/Ubuntu : sudo apt install ffmpeg
  Fedora/RHEL   : sudo dnf install -y ffmpeg ffprobe

Python 3.8 or newer is required.

Quick start
-----------
# dry-run, show what would happen
python3 directplay.py /mnt/media

# real conversion, two workers, better quality, delete sources after success
python3 directplay.py -j 2 --crf 18 --preset slow \
                      --delete-original true /mnt/media /video/new

CLI flags
---------
positional:
  roots                One or more library root folders.

optional:
  -j, --workers N      Parallel encode workers (default = all CPUs)
  --crf N              x264 Constant Rate Factor (quality slider, default 19)
  --preset P           x264 preset: ultrafast … slow (default medium)
  --delete-original    true|false  Remove source file on success (default false)
  --exts .mkv,.mp4     Comma-separated list of file extensions to consider

License
-------
Apache-2.0 — use it, fork it, profit.

Author
------
© 2025 ph33nx   https://github.com/ph33nx
"""
import platform, textwrap, sys
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List
from enum import Enum
import shlex


# ------------------------- pretty logging ------------------------------ #
class EncodeOptions:
    def __init__(self, encode_audio: bool, encode_video: bool):
        self.encode_audio = encode_audio
        self.encode_video = encode_video


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


# ---------------------------- constants ---------------------------------- #

DEFAULT_EXTS = {".mkv", ".mp4"}
IGNORE_SUFFIXES = {'.bak', '.tmp'}  # Files containing these will be skipped
H264_LEVEL_THRESHOLD = 41  # High 4.1 is Chromecast safe
MAX_HEIGHT = 1080  # Down-scale 4 K → 1080p to stay in 4.1
AUDIO_CHANNELS = 2  # stereo AAC


# ------------------------------------------------------------------------- #


def run(cmd: List[str]) -> str:
    "Run command, return stdout text, raise on error."
    return subprocess.check_output(cmd, text=True)


def ffprobe(path: Path) -> dict:
    "Return ffprobe metadata for video & audio streams."
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
    encode_options = EncodeOptions(False, False)

    streams = job.meta["streams"]
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    if v is None:  # no video stream at all
        log("⚠", C.YEL, "No video stream; skipping", job.src)
        return Action.SKIP, encode_options

    if not has_aac_stereo(streams) and False:  # Debug for now
        encode_options.encode_audio = True

    if not is_h264_ok(v):
        encode_options.encode_video = True

    if job.src.suffix.lower() == ".mkv" and not encode_options.encode_video and not encode_options.encode_audio:
        return Action.SKIP, encode_options

    if encode_options.encode_video or encode_options.encode_audio:
        return Action.ENCODE, encode_options

    # Remux path: codecs OK but container wrong
    return Action.REMUX, encode_options


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

        self.dst = src.with_suffix(".mkv") if self.action != Action.SKIP else None


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

        container_ok = path.suffix.lower() == ".mkv"
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
    subs = [s for s in job.meta["streams"] if s["codec_type"] == "subtitle"]

    if not subs:
        return flags  # No subs → no flags needed

    # Case 1: Already srt (MP4/MKV) → copy as-is
    if all(s["codec_name"] == "srt" for s in subs):
        flags.extend([
            "-map", "0:s?",
            "-c:s", "copy",
            "-disposition:s", "0"  # Reset defaults
        ])

    # Case 2: Convert text subs (mov_text/ASS) to srt
    else:
        flags.extend([
            "-map", "0:s?",
            "-c:s", "srt",  # Preferred format for MKV
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
    if job.encoder_options.encode_audio:
        audio_stream = next((s for s in job.meta["streams"] if s["codec_type"] == "audio"), None)
        if audio_stream:
            channels = audio_stream.get("channels", 2)
            channel_layout = audio_stream.get("channel_layout", "")

            layout = {
                1: "mono",
                2: "stereo",
                6: "5.1",
                8: "7.1"
            }.get(channels, "")

            print(f"Encoding audio with {channels} channels & layout {channel_layout}")
            flags.extend([
                "-c:a:0", "aac",
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

    if job.encoder_options.encode_video:
        vfilters = []

        v = next(s for s in job.meta["streams"] if s["codec_type"] == "video")
        need_downscale = int(v.get("coded_height", 0)) > MAX_HEIGHT

        codec_name = v.get("codec_name")
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
            ]
        elif job.opts.encoder == "h264_nvenc":
            video_flags = [
                # --- NVENC encode flags ---
                "-c:v",
                "h264_nvenc",
                "-preset",
                job.opts.nv_preset,
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

            print(f'CODEC {codec_name}')
            # Only use CUDA hwaccel if input is H.264 (AV1 on hardware will fail)
            if codec_name == "h264" or codec_name == "hevc":
                cmd.extend(
                    [
                        "-hwaccel",
                        "cuda",
                        "-hwaccel_output_format",
                        "cuda",
                    ]
                )
        else:
            log("✗", C.RED, f"Unknown encoder ({job.opts.encoder})", job.src)
            exit(1)

        cmd.extend([
            "-i",
            str(job.src),
        ])

        srt_file = job.src.with_suffix(".srt")
        if srt_file.exists():
            print(f'Found subtitle {srt_file}')
            cmd.extend([
                "-i", str(srt_file),
                "-map", "1:s",
            ])

        cmd.extend(
            [
                "-map",
                "0:v:0",
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
    if job.action == Action.REMUX:
        tmp_out = job.dst.with_stem(job.dst.stem + ".tmp")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-y",
            "-i",
            str(job.src),
            "-c",
            "copy",
            "-map_metadata",
            "0",
            tmp_out,
        ]
    else:
        cmd, tmp_out = build_encode_cmd(job)

    log_action = str(job.action)
    if job.action == Action.ENCODE:
        log_action += f" (video: {job.encoder_options.encode_video}, audio: {job.encoder_options.encode_audio})"

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


def gather_files(roots, exts) -> List[Path]:
    files = []

    for root in roots:
        for p in Path(root).rglob("*"):
            # skip dot-files created by macOS (._foo, .DS_Store, etc.)
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts and not any(
                    ignore in p.name for ignore in IGNORE_SUFFIXES):
                files.append(p)

    return files


def main():
    p = argparse.ArgumentParser(description="Pre-encode / remux videos for Chromecast.")
    p.add_argument("roots", nargs="+", help="Root directories to scan.")
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default = 1).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=19,
        help="x264 Constant Rate Factor (lower = higher quality, default 19).",
    )
    p.add_argument(
        "--preset",
        default="medium",
        help="x264 preset: ultrafast…slow (default medium).",
    )
    p.add_argument(
        "--delete-original",
        choices=("true", "false"),
        default="false",
        help="Remove source file after successful processing.",
    )
    p.add_argument(
        "--gpu",
        choices=("true", "false"),
        default="false",
        help="Use NVIDIA hvenc if possible.",
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
    opts.encoder = "h264_nvenc" if (opts.gpu == True and "h264_nvenc" in encoders) else "libx264"

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
        return map_.get(x, "p4")

    # Add the NVENC preset mapping to the options
    opts.nv_preset = _nv_map(opts.preset)

    opts.delete_original = opts.delete_original.lower() == "true"
    exts = {e if e.startswith(".") else f".{e}" for e in opts.exts.split(",")}

    print("Scanning …")
    files = gather_files(opts.roots, exts)
    print(f"Found {len(files)} candidate files.\n")

    with ProcessPoolExecutor(max_workers=opts.workers) as executor:
        jobs = [Job(f, opts) for f in files]
        results = executor.map(process, jobs)
        for result in results:
            print(result)

    print("\nDone.")


if __name__ == "__main__":
    main()
