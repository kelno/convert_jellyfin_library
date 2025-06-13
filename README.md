_Forked from ph33nx gist: https://gist.github.com/ph33nx/b0b7a0ff64bd3f7c8961675db22b4716_

directplay.py
=============

Batch-optimise any video library so every file direct-plays on
Chromecast, smart-TVs and desktop/mobile browsers — no server-side
transcoding required.

This script ensure the following:
- Video is using an mkv container
- Video codec is h264
- Audio codec is AAC (channel count and layour is kept)
- Subtitles are converted to srt format
- Subtitles files matching <video_name.srt> are also automatically added to the container (But not removed yet!)

Key features
------------

- **Smart decision tree** – tries lightning-fast remux first; falls back to H.264 re-encode only when codecs or resolution exceed safe limits.
- **Recursive scan** – pass one or more root folders; sub-directories handled automatically.
- **Parallel workers** (-j / --workers) – use 1 CPU cores by default.
- **Quality knobs** – --crf and --preset expose x264 controls; defaults tuned for "Netflix-like" 1080 p quality.
- **Safe by default** – outputs <name>.mkv in the same folder; originals retained as `<name>.bak.mkv` unless you specify --delete-original.
- **Idempotent** – re-runs skip files already compliant; perfect for nightly cron/systemd-timer jobs.

Dependencies
------------
- Python 3.8 or newer is required.
- ffmpeg and ffprobe must be in your PATH. 
- ffmpeg needs to be built with libfdk_aac. (See https://github.com/m-ab-s/media-autobuild_suite to build it yourself)

The current script directory will be added to path, so you can drop ffmpeg next to it if you want to use a different one.


Quick start
-----------

### Example:   
`python3 directplay.py -j 2 --crf 18 --preset slow --delete-original /mnt/media /video/new`

/!\ Currently, existing .srt files are added to the .mkv but not removed. They might be re added on subsequent runs!
 
CLI flags
---------
*Run this script with '-h' for up to date arguments.*

```
positional arguments:
  roots                 Root directories to scan.

options:
  -h, --help            show this help message and exit
  -j WORKERS, --workers WORKERS
                        Parallel workers. (default: 1)
  --limit LIMIT, -l LIMIT
                        Max videos to process before stopping. 0 means means unlimited. (default: 0)
  --crf CRF             x264 Constant Rate Factor (lower = higher quality). (default: 19)
  -d, --delete-original
                        Remove source file after successful processing. (default: False)
  --hvenc {true,false}  Use NVIDIA hvenc if possible. (default: false)
  --hvenc-preset HVENC_PRESET
                        hvenc preset: ultrafast…slow. (default: medium)
  --skip-video, -sv     Never transcode video. (default: False)
  --skip-audio, --sa    Never transcode video. (default: False)
  --exts EXTS           Comma-separated list of file extensions to consider. (default: .mp4,.mkv)
```

License
-------
Apache-2.0 — use it, fork it, profit.

Author
------
© 2025 ph33nx   https://github.com/ph33nx
© 2025 Kelno   https://github.com/kelno
