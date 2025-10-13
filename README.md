# render-visual

A minimal command‑line renderer extracted from the **Musializer** project. It reads an audio file, performs an FFT‑based visualisation and streams the frames to FFmpeg to produce a video file.

## Dependencies (Linux)

- **raylib** (development headers) – provides the graphics/audio API.  
- **FFmpeg** – must be available in `$PATH` as the `ffmpeg` executable.  
- X11 development libraries (`libx11-dev`, `libxcursor-dev`, `libxrandr-dev`, `libxinerama-dev`, `libxi-dev`) – required by raylib.  
- Standard C tool‑chain (`gcc`, `make`, `pkg-config`).

Install the required packages on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install build-essential libraylib-dev libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev ffmpeg
```

## Build

```bash
make
```

The `Makefile` compiles `render-visual.c` into the `render-visual` executable.

## Usage

```bash
./render-visual [--rainbow-bg] [--lava] [--mirror] <input_audio_file> <output_video_file>
```

Example:

```bash
./render-visual song.wav song.mp4
```

Options:
  --rainbow-bg   Enable rainbow background animation.
  --lava         Enable lava background animation.
  --mirror       Enable mirrored bar effect.

The program will render the visualisation at 30 FPS, and encode the result with FFmpeg (using H.264 video and AAC audio).

## Notes

- The renderer is based on the Musializer codebase; see the Musializer repository for the full project.  
- The output video resolution is fixed to 1920×1080 (16:9) as defined by `RENDER_WIDTH` and `RENDER_HEIGHT` in `render-visual.c`.  
- If you encounter “could not write into ffmpeg pipe” errors, ensure FFmpeg is correctly installed and that you have write permissions in the current directory.

- Original project: [Musializer](https://github.com/tsoding/musializer)

## License

The source files are released under the same license as Musializer (see the Musializer repository).
