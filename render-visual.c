#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <errno.h>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdbool.h>
#include <getopt.h>

#include <raylib.h>
#include <rlgl.h>
static void fft_render_mirror(Rectangle boundary, size_t m);

static const char *circle_fs_source = "#version 330\n"
"\n"
"// Input vertex attributes (from vertex shader)\n"
"in vec2 fragTexCoord;\n"
"in vec4 fragColor;\n"
"\n"
"uniform float radius;\n"
"uniform float power;\n"
"\n"
"// Output fragment color\n"
"out vec4 finalColor;\n"
"\n"
"void main()\n"
"{\n"
"    float r = radius;\n"
"    vec2 p = fragTexCoord - vec2(0.5);\n"
"    if (length(p) <= 0.5) {\n"
"        float s = length(p) - r;\n"
"        if (s <= 0) {\n"
"            finalColor = fragColor*1.5;\n"
"        } else {\n"
"            float t = 1 - s / (0.5 - r);\n"
"            finalColor = mix(vec4(fragColor.xyz, 0), fragColor*1.5, pow(t, power));\n"
"        }\n"
"    } else {\n"
"        finalColor = vec4(0);\n"
"    }\n"
"}\n";

static const char *lava_fs_source = "#version 330\n"
"uniform vec2 resolution;\n"
"uniform float time;\n"
"\n"
"vec2 blob(float id) {\n"
"    // Add slight randomness to speed, radius, and angle for a more organic effect\n"
"    float baseSpeed = 0.3 + 0.2*id;\n"
"    float speed = baseSpeed + 0.05 * sin(time * 0.7 + id);\n"
"    float baseRadius = 0.2 + 0.05*id;\n"
"    float radius = baseRadius + 0.02 * sin(time * 1.3 + id * 2.0);\n"
"    float angle = time*speed + id*6.2831853 + 0.3 * sin(time * 1.1 + id);\n"
"    return vec2(cos(angle), sin(angle)) * radius + vec2(0.5);\n"
"}\n"
"\n"
"float field(vec2 uv) {\n"
"    float v = 0.0;\n"
"    for (int i = 0; i < 10; ++i) {\n"
"        vec2 b = blob(float(i));\n"
"        float d = length(uv - b);\n"
"        v += exp(-pow(d*8.0, 2.0));\n"
"    }\n"
"    return v;\n"
"}\n"
"\n"
"vec3 palette(float t) {\n"
"    return vec3(0.5+0.5*cos(6.2831*(t+vec3(0.0,0.33,0.66))));\n"
"}\n"
"\n"
"out vec4 fragColor;\n"
"void main() {\n"
"    vec2 uv = gl_FragCoord.xy / resolution;\n"
"    float v = field(uv);\n"
"    float mask = smoothstep(0.5, 0.55, v);\n"
"    vec3 col = palette(v);\n"
"    fragColor = vec4(col*mask, mask);\n"
"}\n";

static Shader lava;
static int lava_res_loc;
static int lava_time_loc;



 

#define FFT_SIZE (1<<13)

#define RENDER_FPS 30
#define RENDER_WIDTH 1920
#define RENDER_HEIGHT 1080

#define COLOR_BACKGROUND              GetColor(0x151515FF)

// Microsoft could not update their parser OMEGALUL:
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/complex-math-support?view=msvc-170#types-used-in-complex-math
#ifdef _MSC_VER
#    define Float_Complex _Fcomplex
#    define cbuild(re, im) _FCbuild(re, im)
#    define cfromreal(re) _FCbuild(re, 0)
#    define cfromimag(im) _FCbuild(0, im)
#    define mulcc _FCmulcc
#    define addcc(a, b) _FCbuild(crealf(a) + crealf(b), cimagf(a) + cimagf(b))
#    define subcc(a, b) _FCbuild(crealf(a) - crealf(b), cimagf(a) - cimagf(b))
#else
#    define Float_Complex float complex
#    define cbuild(re, im) ((re) + (im)*I)
#    define cfromreal(re) (re)
#    define cfromimag(im) ((im)*I)
#    define mulcc(a, b) ((a)*(b))
#    define addcc(a, b) ((a)+(b))
#    define subcc(a, b) ((a)-(b))
#endif

// FFMPEG related definitions (moved from ffmpeg.h and ffmpeg_posix.c)
#define READ_END 0
#define WRITE_END 1

typedef struct FFMPEG {
    int pipe;
    pid_t pid;
} FFMPEG;

FFMPEG *ffmpeg_start_rendering(const char *output_path, size_t width, size_t height, size_t fps, const char *sound_file_path)
{
    int pipefd[2];

    if (pipe(pipefd) < 0) {
        TraceLog(LOG_ERROR, "FFMPEG: Could not create a pipe: %s", strerror(errno));
        return NULL;
    }

    pid_t child = fork();
    if (child < 0) {
        TraceLog(LOG_ERROR, "FFMPEG: could not fork a child: %s", strerror(errno));
        return NULL;
    }

    if (child == 0) {
        if (dup2(pipefd[READ_END], STDIN_FILENO) < 0) {
            TraceLog(LOG_ERROR, "FFMPEG CHILD: could not reopen read end of pipe as stdin: %s", strerror(errno));
            exit(1);
        }
        close(pipefd[WRITE_END]);

        char resolution[64];
        snprintf(resolution, sizeof(resolution), "%zux%zu", width, height);
        char framerate[64];
        snprintf(framerate, sizeof(framerate), "%zu", fps);

        int ret = execlp("ffmpeg",
            "ffmpeg",
            "-loglevel", "verbose",
            "-y",

            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", resolution,
            "-r", framerate,
            "-i", "-",
            "-i", sound_file_path,

            "-c:v", "libx264",
            "-vb", "5000k",
            "-c:a", "aac",
            "-ab", "200k",
            "-pix_fmt", "yuv420p",
            output_path,

            NULL
        );
        if (ret < 0) {
            TraceLog(LOG_ERROR, "FFMPEG CHILD: could not run ffmpeg as a child process: %s", strerror(errno));
            exit(1);
        }
        assert(0 && "unreachable");
        exit(1);
    }

    if (close(pipefd[READ_END]) < 0) {
        TraceLog(LOG_WARNING, "FFMPEG: could not close read end of the pipe on the parent's end: %s", strerror(errno));
    }

    FFMPEG *ffmpeg = malloc(sizeof(FFMPEG));
    assert(ffmpeg != NULL && "Buy MORE RAM lol!!");
    ffmpeg->pid = child;
    ffmpeg->pipe = pipefd[WRITE_END];
    return ffmpeg;
}

bool ffmpeg_end_rendering(FFMPEG *ffmpeg, bool cancel)
{
    int pipe = ffmpeg->pipe;
    pid_t pid = ffmpeg->pid;

    free(ffmpeg);

    if (close(pipe) < 0) {
        TraceLog(LOG_WARNING, "FFMPEG: could not close write end of the pipe on the parent's end: %s", strerror(errno));
    }

    if (cancel) kill(pid, SIGKILL);

    for (;;) {
        int wstatus = 0;
        if (waitpid(pid, &wstatus, 0) < 0) {
            TraceLog(LOG_ERROR, "FFMPEG: could not wait for ffmpeg child process to finish: %s", strerror(errno));
            return false;
        }

        if (WIFEXITED(wstatus)) {
            int exit_status = WEXITSTATUS(wstatus);
            if (exit_status != 0) {
                TraceLog(LOG_ERROR, "FFMPEG: ffmpeg exited with code %d", exit_status);
                return false;
            }

            return true;
        }

        if (WIFSIGNALED(wstatus)) {
            TraceLog(LOG_ERROR, "FFMPEG: ffmpeg got terminated by %s", strsignal(WTERMSIG(wstatus)));
            return false;
        }
    }

    assert(0 && "unreachable");
}

bool ffmpeg_send_frame_flipped(FFMPEG *ffmpeg, void *data, size_t width, size_t height)
{
    for (size_t y = height; y > 0; --y) {
        // TODO: write() may not necessarily write the entire row. We may want to repeat the call.
        if (write(ffmpeg->pipe, (uint32_t*)data + (y - 1)*width, sizeof(uint32_t)*width) < 0) {
            TraceLog(LOG_ERROR, "FFMPEG: failed to write into ffmpeg pipe: %s", strerror(errno));
            return false;
        }
    }
    return true;
}

typedef struct {
    // Visualizer
    Shader circle;
    int circle_radius_location;
    int circle_power_location;

    // Renderer
    RenderTexture2D screen;
    Wave wave;
    float *wave_samples;
    size_t wave_cursor;
    FFMPEG *ffmpeg;
    // Waveform visualizer
    float *waveform;
    size_t waveform_pos;

    // FFT Analyzer
    float in_raw[FFT_SIZE];
    float in_win[FFT_SIZE];
    Float_Complex out_raw[FFT_SIZE];
    float out_log[FFT_SIZE];
    float out_smooth[FFT_SIZE];
    float out_smear[FFT_SIZE];
} Renderer;

static Renderer *r = NULL;
int render_width = RENDER_WIDTH;
int render_height = RENDER_HEIGHT;

static bool fft_settled(void)
{
    float eps = 1e-3;
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        if (r->out_smooth[i] > eps) return false;
        if (r->out_smear[i] > eps) return false;
    }
    return true;
}


// Ported from https://cp-algorithms.com/algebra/fft.html
static void fft(float in[], Float_Complex out[], size_t n)
{
    for(size_t i = 0; i < n; i++) {
        out[i] = cfromreal(in[i]);
    }

    for (size_t i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            Float_Complex temp = out[i];
            out[i] = out[j];
            out[j] = temp;
        }
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = 2 * PI / len;
        Float_Complex wlen = cbuild(cosf(ang), sinf(ang));
        for (size_t i = 0; i < n; i += len) {
            Float_Complex w = cfromreal(1);
            for (size_t j = 0; j < len / 2; j++) {
                Float_Complex u = out[i+j], v = mulcc(out[i+j+len/2], w);
                out[i+j] = addcc(u, v);
                out[i+j+len/2] = subcc(u, v);
                w = mulcc(w, wlen);
            }
        }
    }
}

static inline float amp(Float_Complex z)
{
    float a = crealf(z);
    float b = cimagf(z);
    return logf(a*a + b*b);
}

static size_t fft_analyze(float dt)
{
    // Apply the Hann Window on the Input - https://en.wikipedia.org/wiki/Hann_function
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        float t = (float)i/(FFT_SIZE - 1);
        float hann = 0.5 - 0.5*cosf(2*PI*t);
        r->in_win[i] = r->in_raw[i]*hann;
    }

    // FFT
    fft(r->in_win, r->out_raw, FFT_SIZE);

    // "Squash" into the Logarithmic Scale
    float step = 1.06;
    float lowf = 1.0f;
    size_t m = 0;
    float max_amp = 1.0f;
    for (float f = lowf; (size_t) f < FFT_SIZE/2; f = ceilf(f*step)) {
        float f1 = ceilf(f*step);
        float a = 0.0f;
        for (size_t q = (size_t) f; q < FFT_SIZE/2 && q < (size_t) f1; ++q) {
            float b = amp(r->out_raw[q]);
            if (b > a) a = b;
        }
        if (max_amp < a) max_amp = a;
        r->out_log[m++] = a;
    }

    // Normalize Frequencies to 0..1 range
    for (size_t i = 0; i < m; ++i) {
        r->out_log[i] /= max_amp;
    }

    // Smooth out and smear the values
    for (size_t i = 0; i < m; ++i) {
        float smoothness = 8;
        r->out_smooth[i] += (r->out_log[i] - r->out_smooth[i])*smoothness*dt;
        float smearness = 3;
        r->out_smear[i] += (r->out_smooth[i] - r->out_smear[i])*smearness*dt;
    }

    return m;
}

static void draw_background(Rectangle boundary, float t)
{
    // Simple vertical gradient that cycles hue over time (more vibrant)
    float hue = fmodf(t * 60.0f, 360.0f);
    Color top = ColorFromHSV(hue, 0.9f, 0.6f);
    Color bottom = ColorFromHSV(fmodf(hue + 180.0f, 360.0f), 0.9f, 0.4f);
    DrawRectangleGradientV((int)boundary.x, (int)boundary.y,
                           (int)boundary.width, (int)boundary.height,
                           top, bottom);
}
static void draw_waveform(Rectangle boundary, float t)
{
    (void)t; // suppress unused parameter warning
    // Draw the stored waveform as a line strip.
    // Scale samples (assumed in range [-1,1]) to half the height.
    float half_h = boundary.height * 0.5f;
    Vector2 *points = malloc(sizeof(Vector2) * render_width);
    for (size_t i = 0; i < (size_t)render_width; ++i) {
        size_t idx = (r->waveform_pos + i) % (size_t)render_width;
        float sample = r->waveform[idx];
        points[i].x = boundary.x + (float)i;
        points[i].y = boundary.y + half_h - sample * half_h;
    }
    DrawLineStrip(points, render_width, WHITE);
    free(points);
}
static void draw_lava_background(Rectangle boundary, float t)
{
    SetShaderValue(lava, lava_res_loc, (float[2]){ boundary.width, boundary.height }, SHADER_UNIFORM_VEC2);
    SetShaderValue(lava, lava_time_loc, (float[1]){ t }, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(lava);
    DrawRectangleRec(boundary, WHITE);
    EndShaderMode();
}

static void fft_render(Rectangle boundary, size_t m)
{
    // The width of a single bar
    float cell_width = boundary.width/m;

    // Global color parameters
    float saturation = 0.75f;
    float value = 1.0f;

    // Display the Bars
    for (size_t i = 0; i < m; ++i) {
        float t = r->out_smooth[i];
        float hue = (float)i/m;
        Color color = ColorFromHSV(hue*360, saturation, value);
        Vector2 startPos = {
            boundary.x + i*cell_width + cell_width/2,
            boundary.y + boundary.height - boundary.height*2/3*t,
        };
        Vector2 endPos = {
            boundary.x + i*cell_width + cell_width/2,
            boundary.y + boundary.height,
        };
        float thick = cell_width/3*sqrtf(t);
        DrawLineEx(startPos, endPos, thick, color);
    }

    Texture2D texture = { rlGetTextureIdDefault(), 1, 1, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };

    // Display the Smears
    SetShaderValue(r->circle, r->circle_radius_location, (float[1]){ 0.3f }, SHADER_UNIFORM_FLOAT);
    SetShaderValue(r->circle, r->circle_power_location, (float[1]){ 3.0f }, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(r->circle);
    for (size_t i = 0; i < m; ++i) {
        float start = r->out_smear[i];
        float end = r->out_smooth[i];
        float hue = (float)i/m;
        Color color = ColorFromHSV(hue*360, saturation, value);
        Vector2 startPos = {
            boundary.x + i*cell_width + cell_width/2,
            boundary.y + boundary.height - boundary.height*2/3*start,
        };
        Vector2 endPos = {
            boundary.x + i*cell_width + cell_width/2,
            boundary.y + boundary.height - boundary.height*2/3*end,
        };
        float radius = cell_width*3*sqrtf(end);
        Vector2 origin = {0};
        if (endPos.y >= startPos.y) {
            Rectangle dest = {
                .x = startPos.x - radius/2,
                .y = startPos.y,
                .width = radius,
                .height = endPos.y - startPos.y
            };
            Rectangle source = {0, 0, 1, 0.5};
            DrawTexturePro(texture, source, dest, origin, 0, color);
        } else {
            Rectangle dest = {
                .x = endPos.x - radius/2,
                .y = endPos.y,
                .width = radius,
                .height = startPos.y - endPos.y
            };
            Rectangle source = {0, 0.5, 1, 0.5};
            DrawTexturePro(texture, source, dest, origin, 0, color);
        }
    }
    EndShaderMode();

    // Display the Circles
    SetShaderValue(r->circle, r->circle_radius_location, (float[1]){ 0.07f }, SHADER_UNIFORM_FLOAT);
    SetShaderValue(r->circle, r->circle_power_location, (float[1]){ 5.0f }, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(r->circle);
    for (size_t i = 0; i < m; ++i) {
        float t = r->out_smooth[i];
        float hue = (float)i/m;
        Color color = ColorFromHSV(hue*360, saturation, value);
        Vector2 center = {
            boundary.x + i*cell_width + cell_width/2,
            boundary.y + boundary.height - boundary.height*2/3*t,
        };
        float radius = cell_width*6*sqrtf(t);
        Vector2 position = {
            .x = center.x - radius,
            .y = center.y - radius,
        };
        DrawTextureEx(texture, position, 0, 2*radius, color);
    }
    EndShaderMode();
}

static void fft_push(float frame)
{
    memmove(r->in_raw, r->in_raw + 1, (FFT_SIZE - 1)*sizeof(r->in_raw[0]));
    r->in_raw[FFT_SIZE-1] = frame;
}

/* Print usage information */
static void print_help(const char *progname)
{
    fprintf(stderr,
        "Usage: %s [--rainbow-bg] [--lava] [--mirror] [--vert] <input_audio_file> <output_video_file>\n"
        "\n"
        "Options:\n"
        "  --rainbow-bg   Enable rainbow background animation.\n"
        "  --lava         Enable lava background animation.\n"
        "  --mirror       Enable mirrored bar effect.\n"
        "  --vert         Render video in vertical orientation (swap width/height).\n"
        "  --waveform     Enable waveform/oscilloscope visualizer.\n"
        "  -h, --help    Show this help message.\n",
        progname);
}

int main(int argc, char *argv[])
{
    bool rainbow_bg = false;
    bool lava_bg = false;
    bool mirror = false;
    bool waveform = false;
    bool vertical = false;
    const char *input_audio_file = NULL;
    const char *output_video_file = NULL;

    /* Parse command‑line options */
    static const struct option long_opts[] = {
        {"rainbow-bg", no_argument, 0, 'r'},
        {"lava",       no_argument, 0, 'l'},
        {"mirror",     no_argument, 0, 'm'},
        {"waveform",   no_argument, 0, 'w'},
        {"vert",       no_argument, 0, 'v'},
        {"help",       no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "rhlmv", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'r':
            rainbow_bg = true;
            break;
        case 'l':
            lava_bg = true;
            break;
        case 'm':
            mirror = true;
            break;
        case 'w':
            waveform = true;
            break;
        case 'v':
            vertical = true;
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        case '?':
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            return 1;
        }
    }

    if (optind + 2 != argc) {
        fprintf(stderr, "Error: expected <input_audio_file> <output_video_file>\n");
        print_help(argv[0]);
        return 1;
    }

    input_audio_file  = argv[optind];
    output_video_file = argv[optind + 1];

    /* Determine render dimensions, possibly swapping for vertical mode */
    int render_width = RENDER_WIDTH;
    int render_height = RENDER_HEIGHT;
    if (vertical) {
        int tmp = render_width;
        render_width = render_height;
        render_height = tmp;
    }

    r = malloc(sizeof(*r));
    assert(r != NULL && "Buy more RAM lol");
    memset(r, 0, sizeof(*r));
    r->waveform = calloc(render_width, sizeof(float));
    r->waveform_pos = 0;

    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(render_width, render_height, "Musializer Renderer");
    InitAudioDevice();

    // Load assets
    // TODO: make resource loading more robust (e.g., use plug_load_resource)

    r->circle = LoadShaderFromMemory(NULL, circle_fs_source);
    r->circle_radius_location = GetShaderLocation(r->circle, "radius");
    r->circle_power_location = GetShaderLocation(r->circle, "power");
    lava = LoadShaderFromMemory(NULL, lava_fs_source);
    lava_res_loc = GetShaderLocation(lava, "resolution");
    lava_time_loc = GetShaderLocation(lava, "time");

    r->screen = LoadRenderTexture(render_width, render_height);

    // Load audio file
    r->wave = LoadWave(input_audio_file);
    if (r->wave.frameCount == 0) {
        fprintf(stderr, "Error: Could not load audio file %s\n", input_audio_file);
        return 1;
    }
    r->wave_samples = LoadWaveSamples(r->wave);
    r->wave_cursor = 0;

    // Start FFmpeg rendering
    r->ffmpeg = ffmpeg_start_rendering(output_video_file, render_width, render_height, RENDER_FPS, input_audio_file);
    if (r->ffmpeg == NULL) {
        fprintf(stderr, "Error: Could not start FFmpeg rendering\n");
        return 1;
    }

    SetTraceLogLevel(LOG_WARNING);

    // Rendering loop
    float bg_time = 0.0f;
    while (r->wave_cursor < r->wave.frameCount || !fft_settled()) {
        // Process audio chunks
        size_t chunk_size = r->wave.sampleRate/RENDER_FPS;
        float *fs = (float*)r->wave_samples;
        for (size_t i = 0; i < chunk_size; ++i) {
            float sample = 0.0f;
            if (r->wave_cursor < r->wave.frameCount) {
                sample = fs[r->wave_cursor*r->wave.channels + 0];
                fft_push(sample);
            } else {
                fft_push(0);
            }
            // Store sample for waveform visualizer
            r->waveform[r->waveform_pos] = sample;
            r->waveform_pos = (r->waveform_pos + 1) % render_width;

            r->wave_cursor += 1;
        }

        // Perform FFT analysis
        size_t m = fft_analyze(1.0f/RENDER_FPS);

        // Render to texture
        BeginTextureMode(r->screen);
        ClearBackground(COLOR_BACKGROUND);
        /* Render rainbow background first (if enabled) */
        if (rainbow_bg) {
            draw_background((Rectangle){0, 0, (float)render_width, (float)render_height}, bg_time);
        }
        if (waveform) {
            draw_waveform((Rectangle){0, 0, (float)render_width, (float)render_height}, bg_time);
        }
        /* Render lava background on top (if enabled) */
        if (lava_bg) {
            draw_lava_background((Rectangle){0, 0, (float)render_width, (float)render_height}, bg_time);
        }
        fft_render((Rectangle){0, 0, (float)render_width, (float)render_height}, m);
        if (mirror) {
            fft_render_mirror((Rectangle){0, 0, (float)render_width, (float)render_height}, m);
        }
        EndTextureMode();

        // Send frame to FFmpeg
        Image image = LoadImageFromTexture(r->screen.texture);
        if (!ffmpeg_send_frame_flipped(r->ffmpeg, image.data, image.width, image.height)) {
            fprintf(stderr, "Error: Could not send frame to FFmpeg\n");
            ffmpeg_end_rendering(r->ffmpeg, false);
            UnloadImage(image);
            return 1;
        }
        UnloadImage(image);
        bg_time += 1.0f / RENDER_FPS;
    }

    // Finish rendering
    if (!ffmpeg_end_rendering(r->ffmpeg, false)) {
        fprintf(stderr, "Error: Could not finalize FFmpeg rendering\n");
        return 1;
    }

    // Cleanup
    UnloadWave(r->wave);
    UnloadWaveSamples(r->wave_samples);
    UnloadShader(r->circle);
    UnloadShader(lava);
    UnloadRenderTexture(r->screen);
    CloseAudioDevice();
    CloseWindow();
    free(r->waveform);
    free(r);

    return 0;
}

/* Mirror version of fft_render: draws bars from the top, reversed horizontally */
static void fft_render_mirror(Rectangle boundary, size_t m)
{
    // Width of a single bar
    float cell_width = boundary.width / m;

    // Global color parameters (same as in fft_render)
    float saturation = 0.75f;
    float value = 1.0f;

    // Display the Bars (mirrored horizontally, drawing from top)
    for (size_t i = 0; i < m; ++i) {
        float t = r->out_smooth[i];
        float hue = (float)i / m;
        Color color = ColorFromHSV(hue * 360, saturation, value);
        float x_center = boundary.x + (m - 1 - i) * cell_width + cell_width / 2;
        Vector2 startPos = {
            x_center,
            boundary.y,
        };
        Vector2 endPos = {
            x_center,
            boundary.y + boundary.height * 2 / 3 * t,
        };
        float thick = cell_width / 3 * sqrtf(t);
        DrawLineEx(startPos, endPos, thick, color);
    }

    // Texture for smears and circles
    Texture2D texture = { rlGetTextureIdDefault(), 1, 1, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };

    // Display the Smears (mirrored)
    SetShaderValue(r->circle, r->circle_radius_location, (float[1]){ 0.3f }, SHADER_UNIFORM_FLOAT);
    SetShaderValue(r->circle, r->circle_power_location, (float[1]){ 3.0f }, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(r->circle);
    for (size_t i = 0; i < m; ++i) {
        float start = r->out_smear[i];
        float end = r->out_smooth[i];
        float hue = (float)i / m;
        Color color = ColorFromHSV(hue * 360, saturation, value);
        float x_center = boundary.x + (m - 1 - i) * cell_width + cell_width / 2;
        Vector2 startPos = {
            x_center,
            boundary.y + boundary.height * 2 / 3 * start,
        };
        Vector2 endPos = {
            x_center,
            boundary.y + boundary.height * 2 / 3 * end,
        };
        float radius = cell_width * 3 * sqrtf(end);
        Vector2 origin = {0};
        if (endPos.y >= startPos.y) {
            Rectangle dest = {
                .x = startPos.x - radius/2,
                .y = startPos.y,
                .width = radius,
                .height = endPos.y - startPos.y
            };
            Rectangle source = {0, 0, 1, 0.5};
            DrawTexturePro(texture, source, dest, origin, 0, color);
        } else {
            Rectangle dest = {
                .x = endPos.x - radius/2,
                .y = endPos.y,
                .width = radius,
                .height = startPos.y - endPos.y
            };
            Rectangle source = {0, 0.5, 1, 0.5};
            DrawTexturePro(texture, source, dest, origin, 0, color);
        }
    }
    EndShaderMode();

    // Display the Circles (mirrored)
    SetShaderValue(r->circle, r->circle_radius_location, (float[1]){ 0.07f }, SHADER_UNIFORM_FLOAT);
    SetShaderValue(r->circle, r->circle_power_location, (float[1]){ 5.0f }, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(r->circle);
    for (size_t i = 0; i < m; ++i) {
        float t = r->out_smooth[i];
        float hue = (float)i / m;
        Color color = ColorFromHSV(hue * 360, saturation, value);
        float x_center = boundary.x + (m - 1 - i) * cell_width + cell_width / 2;
        Vector2 center = {
            x_center,
            boundary.y + boundary.height * 2 / 3 * t,
        };
        float radius = cell_width * 6 * sqrtf(t);
        Vector2 position = {
            .x = center.x - radius,
            .y = center.y - radius,
        };
        DrawTextureEx(texture, position, 0, 2*radius, color);
    }
    EndShaderMode();
}
