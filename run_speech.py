## Spectral Analysis Project

# %%
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import time 
plt.ion()
from myutils import (
    calc_pressure_matching,
    calc_smooth_pressure_matching,
    create_rectangular_perimeter_speaker_array,
    create_uniform_rectangular_mic_grid,
    get_energy_map_db,
    get_zone_indices,
    plot_audio_analysis,
    plot_pressure_map,
    save_as_wav,
    simulate_listening_points,
    play_audio_directly,
    save_combined_wav,
    evaluate_zone_smoothness,
    evaluate_acoustic_contrast,
    get_or_compute_H
)

# %% Parameters
nbr_bounces = 3 # max_order: max number of reflections in the room
air_absorption = True # air_absorbtion: if air is absorbed or not
fs = 16000   # Sampling freq. of the played audio
room_dim = [5.0, 5.0, 5.0]
num_speakers_per_row = 10
mic_spacing = 0.1
nfft = 512*2  # for the fourier transfrom
f_axis = np.fft.rfftfreq(nfft, d=1/fs)

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




# %% Create signal

# Selecting params for signal
audio_freq = np.array([400, 500, 600])
audio_amp = np.array([1, 1, 1])
audio_phase = np.array([np.pi, np.pi/2, 0])
duration = 2.0  # seconds
c = pra.constants.get('c')  # Speed of sound

# Creating signal as sum of frequencies
t_axis = np.arange(0, duration, 1/fs)   
audio_time = np.zeros(len(t_axis))
for idx, freq in enumerate(audio_freq):
    audio_time += audio_amp[idx] * np.sin(2 * np.pi * freq * t_axis + audio_phase[idx])
audio_fft = np.fft.rfft(audio_time, n=nfft) 




#%% TRY REAL SIGNAL


import scipy.io.wavfile as wav
from myutils import clean_wav_data, resample_signal

# 1. Load the wav file
filepath = 'wav_files/why_were_you_away.wav'
fs_file, wav_data = wav.read(filepath)
data = clean_wav_data(wav_data) # extract signal and normalize
audio_time_full = resample_signal(data, fs_file, fs)


# 2. Select section
start_sec = 0.250
duration = 20e-3
num_samples = int(duration * fs)
start_index = round(start_sec * fs)
end_index = start_index + num_samples
audio_time = audio_time_full[start_index : end_index]

# 3. Compute the FFT
audio_fft = np.fft.rfft(audio_time, n=nfft)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Time domain (plotting the first 10ms for detail)
t_plot = np.arange(len(audio_time)) / fs * 1000  # time in ms
ax1.plot(t_plot, audio_time, color='tab:blue')
ax1.set_xlabel("Time [ms]")
ax1.set_ylabel("Magnitude")
ax1.set_title(f"Time Domain: Audio Signal ({duration*1000:.0f} ms slice)")
ax1.grid(True)

# Frequency domain
ax2.plot(f_axis, np.log(np.abs(audio_fft)/(nfft/2)), color='tab:orange')
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("log(Magnitude)")
ax2.set_title("Frequency Domain: FFT of Audio Signal")
ax2.grid(True)
# ax2.set_ylim(0, np.log(0.03))

plt.tight_layout()
plt.show()


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

std_reg, ripple_reg = evaluate_zone_smoothness(p_full, audio_freq, audio_amp, fs, nfft, bright_indices)
contrast_reg = evaluate_acoustic_contrast(p_full, audio_freq, audio_amp, fs, nfft, bright_indices, dark_indices)

print('='*50)

print("--- Standard PM ---")
print(f"Smoothness (STD): {std_reg:.2f} dB")
print(f"Acoustic Contrast: {contrast_reg:.2f} dB\n")

# 2. Smooth Pressure Matching 
p_full_smooth, g_full_smooth = calc_smooth_pressure_matching(room, nfft, H_full, bright_indices, dark_indices)

std_smooth, ripple_smooth = evaluate_zone_smoothness(p_full_smooth, audio_freq, audio_amp, fs, nfft, bright_indices)
contrast_smooth = evaluate_acoustic_contrast(p_full_smooth, audio_freq, audio_amp, fs, nfft, bright_indices, dark_indices)

print("--- Smooth PM ---")
print(f"Smoothness (STD): {std_smooth:.2f} dB")
print(f"Acoustic Contrast: {contrast_smooth:.2f} dB")
print('='*50)


# Calculate the pressure map using extracted data
pressure_map = get_energy_map_db(
    p_full, audio_freq, audio_amp, room.fs, nfft, X.shape
)

pressure_map_smooth = get_energy_map_db(
    p_full_smooth, audio_freq, audio_amp, room.fs, nfft, X.shape
)

# Visualize
plot_pressure_map(
    pressure_map, X, Y, all_speakers, 
    bright_center, dark_center, radius, 
    title=f"Pressure Matching (Standard): {audio_freq} Hz"
)

plot_pressure_map(
    pressure_map_smooth, X, Y, all_speakers, 
    bright_center, dark_center, radius, 
    title=f"Pressure Matching (Smooth): {audio_freq} Hz"
)

# %% Listen to results

# Run the simulation for the two specific points
bright_norm, dark_norm = simulate_listening_points(
    room_dim, fs, all_speakers, g_full, 
    audio_freq, audio_amp, nfft, 
    bright_center, dark_center
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
    "pm_dark_zone_center.wav",
    freq_range=(300, 700),  # Zoomed into our 400, 500, 600 Hz signals
    time_zoom=(0.5, 0.55)
)


# %% Only for Sara's Run All button to work

# This has to be in the end of the file for Sara's figures to not close down
plt.ioff()
plt.show()