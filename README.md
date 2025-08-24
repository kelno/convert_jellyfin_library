Forked from ph33nx gist: https://gist.github.com/ph33nx/b0b7a0ff64bd3f7c8961675db22b4716

directplay.py
=============

Batch-optimise my video library so every file direct-plays on my devices, and hopefully most others as well.

This script ensure the following:
- Video is using a webm container
- Video is:
  - Codec AV1 (controlled by command line)
  - Max height 1080
  - Max level 4.1
- Audio codec is opus in stereo
- Subtitles are converted to webVTT format
- Subtitles files matching <video_name.srt> are also automatically added to the container (but not removed as of writing)

Key features
------------

- **Smart decision tree** – tries lightning-fast remux first; falls back to re-encode only when needed.
- **Recursive scan** – pass one or more root folders; sub-directories handled automatically.
- **Parallel workers** (-j / --workers) – use 1 CPU cores by default.
- **Quality knobs** – --crf and --preset expose x264 controls; defaults tuned for "Netflix-like" 1080 p quality.
- **Safe by default** – outputs <name>.webm in the same folder; originals retained as `<name>.bak.webm` unless you specify --delete-original.
- **Idempotent** – re-runs skip files already compliant; perfect for nightly cron/systemd-timer jobs.

Dependencies
------------
- Python 3.8 or newer is required.
- ffmpeg and ffprobe must be in your PATH. 

The current script directory will be added to path, so you can drop ffmpeg next to it if you want to use a specific one.


Quick start
-----------

### Example:   
This will process all files in given directories, with 2 threads, deleting the originals.  
`python directplay.py --workers 2 --delete-original /mnt/media /mnt/anotheroptionalmedia/`

/!\ Currently, existing .srt files are added to the container but not removed. They might be re added on subsequent runs!
 
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
  -d, --delete-original
                        Remove source file after successful processing. (default: False)
  --encoder ENCODER     Specify which encoder to use. Supported encoders are: ['libx264', 'h264_nvenc', 'av1_nvenc'] (default:
                        av1_nvenc)
  --preset PRESET       Presets, has to match the encoder. (ex for xh264: ultrafast…slow) (default: None)
  --skip-video, -sv     Never transcode video. (default: False)
  --skip-subtitles, -ss
                        Never transcode subtitles. (default: False)
  --skip-audio, --sa    Never transcode video. (default: False)
  --exts EXTS           Comma-separated list of file extensions to consider. (default: .webm,.mp4,.mkv)
  --debug               Create a .txt file next to destination with ffmpeg command. (default: False)
  --sample SAMPLE       Create a segment sample of given length in second. (default: 0)
```

License
-------
Apache-2.0 — use it, fork it, profit.

Author
------
© 2025 ph33nx   https://github.com/ph33nx
2025 Kelno   https://github.com/kelno
