import glfw
import numpy as np
from functools import partial
import soundfile as sf
from scipy.signal import spectrogram
from decord import VideoReader, gpu
import fastplotlib as fpl
from ipywidgets import VBox, HBox
import threading
import time
import cv2
import threading
from video_reader import LazyVideo
import os


# Functions
def make_specgram(audio, fps=125000):
    f, t, spec = spectrogram(audio, fs=fps, nfft=n_fft, nperseg=n_samples_bin, noverlap=n_samples_overlap,
                             return_onesided=True)
    # Remove the 0-frequency bin and flip the frequency axis so that high frequencies are at the top.
    f = f[1:][::-1]
    spec = np.flip(spec[1:], axis=0)
    spec = np.log(np.abs(spec) + 1e-12).astype(np.float32)
    return t, f, spec


# Compute the spectrogram from audio file.
# t_spec, f_spec, spec_data = make_specgram(audio_data, fps_audio)
# print(t_spec.shape)
# print(" - Spectrogram computed with shape:", spec_data.shape)


def compute_bins_for_window(fs, nperseg, noverlap, window_sec):
    """Computes the number of time bins in a given window duration."""
    delta_t = (nperseg - noverlap) / fs  # Time per bin
    bins_for_window = int(window_sec / delta_t)  # Total bins for desired duration
    return bins_for_window


# num_bins_window = compute_bins_for_window(fps_audio, n_samples_bin, n_samples_overlap, window_width_sec)
# #print(num_bins_window)
# initial_window = spec_data[:,:num_bins_window]

def get_spect_window(ev):
    t_frame_video = ev['t']
    t_sec = t_frame_video / 30  # how can I take out the 30?

    window_duration = 5.0
    dt = t_spec[1] - t_spec[0]
    window_size = int(window_duration / dt)

    center_idx = np.searchsorted(t_spec, t_sec)
    half_window = window_size // 2

    start_idx = max(0, center_idx - half_window)
    end_idx = min(len(t_spec), center_idx + half_window)

    # print(f"Frame: {t_frame_video}, Time (sec): {t_sec}, Index: {center_idx}, Window: ({start_idx}, {end_idx})")

    spec_slices: dict[str, np.ndarray] = dict()

    for loc in location_order:
        spec_data = specs[loc]

        spectrogram_slice = spec_data[:, start_idx:end_idx]

        # Pad if necessary to maintain fixed size
        if spectrogram_slice.shape[1] < window_size:
            pad_width = window_size - spectrogram_slice.shape[1]
            spectrogram_slice = np.pad(
                spectrogram_slice, ((0, 0), (0, pad_width)), mode="constant"
            )

        spec_slices[loc] = spectrogram_slice

    for loc in location_order[:2]:
        spec_fig1[loc].graphics[0].data = spec_slices[loc]

    for loc in location_order[2:]:
        spec_fig2[loc].graphics[0].data = spec_slices[loc]


# ==== Choose ====
exp_num = 237
file_num = 20
channel_numbers = [2, 0, 4, 5]  # Choose 4 channels

# === Video file naming rule based on channel number ===
video_prefix_map = {
    0: "video_center_",
    1: "video_center_",
    2: "video_gily_center_",
    3: "video_gily_center_",
    4: "video_nest_side_",
    5: "video_burrow_side_"
}

# maps int -> location
name_mapping = {
    0: "center-1",
    1: "center-1",
    2: "center-2",
    3: "center-2",
    4: "nest",
    5: "burrow"
}

location_order = ["center-1", "center-2", "nest", "burrow"]

base_path_audio = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Raw_data\experiment_{exp_num}\concatenated_data_cam_mic_sync\wavs"
base_path_video = fr"\\sanesstorage.cns.nyu.edu\archive\ginosar\Raw_data\experiment_{exp_num}\concatenated_data_cam_mic_sync"

# === spectrogram variables ===
fps_audio = 125000
fps_video = 29.9976  # 30
n_fft = 512
n_samples_bin = 512
n_samples_overlap = 256
window_width_sec = 5  # choose

# for all synchrinozed files
num_bins_window = compute_bins_for_window(fps_audio, n_samples_bin, n_samples_overlap, window_width_sec)

# === Collect paths ===
# maps location str -> path str
video_paths: dict[str, str] = dict()
audio_paths: dict[str, str] = dict()

file_num_str = f"{file_num:03d}"

for ch in channel_numbers:
    video_prefix = video_prefix_map[ch]
    if video_prefix is None:
        print(f"Warning: Invalid channel number {ch}, skipping.")
        continue

    video_path = os.path.join(base_path_video, f"{video_prefix}{file_num}.mp4")
    audio_path = os.path.join(base_path_audio, f"channel_{ch:02d}_file_{file_num_str}.wav")


    video_paths[name_mapping[ch]] = video_path
    audio_paths[name_mapping[ch]] = audio_path

# === Output the results ===
# print("Video paths:")
# for path in video_paths:
#     print(" ", path)

# print("\nAudio paths:")
# for path in audio_paths:
#     print(" ", path)


## Load video and audio ##
# maps location name -> LazyVideo
movies: dict[str, LazyVideo] = dict()

# maps location name -> full spectrogram
specs: dict[str, np.ndarray] = dict()

# gpu_context = gpu(0)

# --- Video loading ---
for location, path in video_paths.items():
    movies[location] = LazyVideo(path)

for location, path in audio_paths.items():
    audio_data, fps_audio = sf.read(path, dtype='float32')
    print(f"loaded: {path}")

    # Unpack the outputs of make_specgram
    t_spec, f_spec, spec_data = make_specgram(audio_data, fps_audio)

    specs[location] = spec_data



# Create the widget
spec_fig1 = fpl.Figure(
    size=(2000, 200),
    shape=(1, 2),
    names=[location_order[:2]],
    # controller_ids="sync"
)

spec_fig2 = fpl.Figure(
    size=(2000, 200),
    shape=(1, 2),
    names=[location_order[2:]],
    # controllers=[spec_fig1[0, 0].controller, spec_fig1[0, 0].controller]
)

for loc in location_order[:2]:
    spec_fig1[loc].add_image(
        data=specs[loc][:, :num_bins_window],
        cmap="viridis",
        name="spectrogram",
    )
    spec_fig1[loc].toolbar = False
    spec_fig1[loc].axes.y.tick_format = lambda v, min_v, max_v: f"{v} Hz"

for loc in location_order[2:]:
    spec_fig2[loc].add_image(
        data=specs[loc][:, :num_bins_window],
        cmap="viridis",
        name="spectrogram",
    )
    spec_fig2[loc].toolbar = False
    spec_fig2[loc].axes.y.tick_format = lambda v, min_v, max_v: f"{v} Hz"

# ---
n_ticks = 6
# freq_ticks = np.linspace(0, init_window1.shape[0] - 1, n_ticks).astype(int)
# freq_labels = [f"{int(freq):d}" for freq in f_spec1[::-1][freq_ticks]]

## ask kushal - make yaxis labels for freq

# - ask kushal: de-couple zoom in of videos
# - larger boxes but with no wasted parts

video_widget1 = fpl.ImageWidget(
    [movies[loc] for loc in location_order[:2]],
    histogram_widget=False,
    rgb=False,
    figure_kwargs={"size": (2000, 500), "shape": (1, 2), "controller_ids": None}
)

video_widget2 = fpl.ImageWidget(
    [movies[loc] for loc in location_order[2:]],
    histogram_widget=False,
    rgb=False,
    figure_kwargs={"size": (2000, 500), "shape": (1, 2), "controller_ids": None}
)

for i, iw in enumerate([video_widget1, video_widget2]):
    for subplot in iw.figure:
        subplot.toolbar = False
        subplot.axes.visible = False

    iw.add_event_handler(get_spect_window, "current_index")
    iw.show()

for iw in [video_widget1, video_widget2]:
    for subplot in iw.figure:
        subplot.camera.zoom = 1.25

def update_iw_index(index):
    video_widget1.current_index = index

def garbage_monkey_patch():
    return

video_widget1.figure.guis["bottom"].size = 0
video_widget1.figure.guis["bottom"].update = garbage_monkey_patch
video_widget2.add_event_handler(update_iw_index, "current_index")

spec_fig1.show(maintain_aspect=False)
spec_fig2.show(maintain_aspect=False)


glfw.set_window_pos(spec_fig1.canvas._window, 0, 0)
glfw.set_window_pos(video_widget1.figure.canvas._window, 0, 200)

glfw.set_window_pos(spec_fig2.canvas._window, 0, 700)
glfw.set_window_pos(video_widget2.figure.canvas._window, 0, 900)
fpl.loop.run()
