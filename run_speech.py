## Spectral Analysis Project

# %%

# Clear all user-defined variables
from IPython import get_ipython

ip = get_ipython()
ip.run_line_magic("load_ext", "autoreload")
ip.run_line_magic("autoreload", "2")

import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import time 
plt.ion()
from myutils import (
    calc_pressure_matching,
    calc_smooth_pressure_matching,
    create_rectangular_perimeter_speaker_array,
    create_uniform_rectangular_mic_grid,
    get_energy_map_db,
    get_energy_map_db_sara,
    get_zone_indices,
    plot_audio_analysis,
    plot_pressure_map,
    save_as_wav,
    simulate_listening_points,
    simulate_listening_points_sara,
    play_audio_directly,
    save_combined_wav,
    evaluate_zone_smoothness,
    evaluate_acoustic_contrast,
    get_or_compute_H,
    create_signal, 
    plot_signal,
    clean_wav_data, 
    resample_signal
)

# import myutils

# %% Parameters

nbr_bounces = 3 # max_order: max number of reflections in the room
air_absorption = True # air_absorbtion: if air is absorbed or not
fs = 16_000   # Sampling freq. of the played audio
room_dim = [5.0, 5.0, 5.0]
num_speakers_per_row = 10
# mic_spacing = 0.5
mic_spacing = 0.1
nfft = 512*2  # for the fourier transfrom
f_axis = np.fft.rfftfreq(nfft, d=1/fs)

# %% Importing An Audio File

# from myutils import import_signal

# filepath = 'wav_files/why_were_you_away.wav'
# start_sec = 0.200
# duration = 20e-3

# t_axis, audio_time, audio_fft = import_signal(filepath, fs, nfft, start_sec, duration)

# plot_signal(t_axis, audio_time, f_axis, audio_fft, fs, nfft)

# %% Create signal

# Selecting params for signal
audio_freq = np.array([400, 500, 600])
audio_amp = np.array([1, 1, 1])
audio_phase = np.array([np.pi, np.pi/2, 0])
duration = 2.0  # seconds
c = pra.constants.get('c')  # Speed of sound

t_axis, audio_time, audio_fft = create_signal(duration, fs, nfft, audio_freq, audio_amp, audio_phase)

plot_signal(t_axis, audio_time, f_axis, audio_fft, fs, nfft)

# %% Create room

room = pra.ShoeBox(
    room_dim, fs=fs, max_order=nbr_bounces, air_absorption=air_absorption
)
all_speakers = create_rectangular_perimeter_speaker_array(
    room_dim, num_speakers_per_row
)
num_speakers = all_speakers.shape[1]
for loc in all_speakers.T:
    room.add_source(loc)
    
mics_locs, X, Y = create_uniform_rectangular_mic_grid(room_dim, spacing=mic_spacing)
mic_array = pra.MicrophoneArray(mics_locs, room.fs)
num_mics = mic_array.R.shape[1]
room.add_microphone_array(mic_array)

room.plot()

# %% Compute or Load Room Impulse Response and H_full

# Create a dictionary of all variables that affect the acoustics
cache_params = {
    "nbr_bounces": nbr_bounces,
    "air_absorption": air_absorption,
    "fs": fs,
    "room_dim": room_dim,
    "nfft": nfft,
    "speaker_locs": np.round(all_speakers, 4).tolist(),
    "mic_locs": np.round(mics_locs, 4).tolist()
}

print('='*50)
start = time.time()
H_full = get_or_compute_H(room, nfft, cache_params)
end = time.time()
print(f'H_full ready in {end - start:.2f} seconds')
print('='*50)


# %% Define Zones

radius = 0.5
bright_center = np.array([1.5, 3.0, 2.5])
dark_center = np.array([3.5, 3.0, 2.5])
bright_indices, dark_indices = get_zone_indices(
    mics_locs, bright_center, dark_center, radius
)
print('='*50)
print(f"Mics in Bright Zone: {len(bright_indices)}")
print(f"Mics in Dark Zone: {len(dark_indices)}")
print('='*50)


# %% Calculate and visualize results

# 1. Standard Pressure Matching
p_full, g_full = calc_pressure_matching(room, nfft, H_full, bright_indices, dark_indices)

# 2. Smooth Pressure Matching 
print('='*50)
print("--- Standard PM ---")
p_full_smooth, g_full_smooth = calc_smooth_pressure_matching(room, nfft, H_full, bright_indices, dark_indices)

pressure_map = get_energy_map_db_sara(p_full, audio_fft, X.shape)

# Visualize
plot_pressure_map(
    pressure_map, X, Y, all_speakers, 
    bright_center, dark_center, radius, 
    title=f"Pressure Matching (Standard)"
)

# %% Listen to results

# Run the simulation for the two specific points
bright_norm, dark_norm = simulate_listening_points_sara(
    room_dim, fs, all_speakers, g_full, audio_fft, bright_center, dark_center, nfft
)

# Save the results
save_as_wav("pm_bright_zone_center.wav", bright_norm, fs)
save_as_wav("pm_dark_zone_center.wav", dark_norm, fs)

print("Audio files saved!")
play_audio_directly(bright_norm, dark_norm, fs)

# Save as a single sequential file
save_combined_wav("pm_combined_zones.wav", bright_norm, dark_norm, fs, pause_duration=1.0)


# %% View wav files

plot_audio_analysis(
    "pm_bright_zone_center.wav", 
    "pm_dark_zone_center.wav", time_zoom=(0, 0.15)
)


# %% Only for Sara's Run All button to work

# This has to be in the end of the file for Sara's figures to not close down
plt.ioff()
plt.show()
# %%
