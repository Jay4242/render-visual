# render-visual

A minimal command‑line renderer extracted from the [Musializer](https://github.com/tsoding/musializer) project. It reads an audio file, performs an FFT‑based visualisation and streams the frames to FFmpeg to produce a video file.

![screenshot.jpg](screenshot.jpg)

## Dependencies (Linux)

- **raylib** – install following the instructions at https://www.raylib.com/.  
- **FFmpeg** – must be available in `$PATH` as the `ffmpeg` executable.  
- X11 development libraries (`libx11-dev`, `libxcursor-dev`, `libxrandr-dev`, `libxinerama-dev`, `libxi-dev`) – required by raylib.  
- Standard C tool‑chain (`gcc`, `make`, `pkg-config`).

Install the required packages on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install build-essential libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev ffmpeg
```

## Build

```bash
make
```

The `Makefile` compiles `render-visual.c` into the `render-visual` executable.

## Usage

```bash
./render-visual [--rainbow-bg] [--lava] [--mirror] [--vert] [--waveform] <input_audio_file> <output_video_file>
```

Example:

```bash
./render-visual song.wav song.mp4
```

Options:
  --rainbow-bg   Enable rainbow background animation.
  --lava         Enable lava background animation.
  --mirror       Enable mirrored bar effect.
  --vert         Render video in vertical orientation (swap width/height).
  --waveform     Enable waveform/oscilloscope visualizer.

The program will render the visualisation at 30 FPS, and encode the result with FFmpeg (using H.264 video and AAC audio).

## Notes

- The renderer is based on the [Musializer](https://github.com/tsoding/musializer) codebase; see the [Musializer](https://github.com/tsoding/musializer) repository for the full project.  
- The output video resolution is fixed to 1920×1080 (16:9) as defined by `RENDER_WIDTH` and `RENDER_HEIGHT` in `render-visual.c`.  
- Use `--vert` to swap width and height for a vertical video orientation.  
- Use `--waveform` to enable an oscilloscope‑style waveform visualizer.  
- If you encounter “could not write into ffmpeg pipe” errors, ensure FFmpeg is correctly installed and that you have write permissions in the current directory.

- Original project: [Musializer](https://github.com/tsoding/musializer)

## License

The source files are released under the same license as Musializer (see the Musializer repository).
