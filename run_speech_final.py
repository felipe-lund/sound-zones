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
    create_pure_signal,
    create_rectangular_perimeter_speaker_array,
    create_uniform_rectangular_mic_grid,
    get_energy_map_db,
    get_energy_map_db_sara,
    get_zone_indices,
    import_signal,
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
    plot_signal,
    plot_signal_log,
    clean_wav_data, 
    resample_signal,
    window_signal
)

# import myutils

# %% Parameters

# Audio 
fs       = 16_000       # Sampling freq. of the played audio

# Room characteristics
nbr_bounces = 3 # max_order: max number of reflections in the room
air_absorption = True # air_absorbtion: if air is absorbed or not

# Room setup
room_dim = [5.0, 5.0, 5.0]
num_speakers_per_row = 10
mic_spacing = 0.1

# Printing
print('='*50)
print(
    f"Room Properties:\n"
    f"\tNumber of bounces: {nbr_bounces}\n"
    f"\tAir Absorption:    {air_absorption}\n"
    f"\tRoom Dimentions:   {room_dim}"
)
print('='*50)

# %% Importing of Creating an audio of x [ms]

# ------------------------
# Import Audio
import_audio = False
filepath = 'wav_files/why_were_you_away.wav' # used if import_audio = True
start_sec = 0.200       # second to start the audio processing
duration  = 40e-3       # seconds
overlap = 0.5

# NFFT needs to be equal to or larger than fs*duration
nfft = 2**int(np.ceil(np.log2(fs * duration))+2) # The next power of 2
nfft = 1024
f_axis = np.fft.rfftfreq(nfft, d=1/fs)

# Params
plot = False
n_chunks = 1
print('='*50)
print(
    f"Audio Properties:\n"
    f"\tImported Audio:    {import_audio}\n"
    f"\tSampling freq:     {fs} [Hz]\n"
    f"\tNFFT:              {nfft}\n"
    f"\tDuration:          {duration} [s]"
)
print('='*50)


time_signals = []
f_signals = []
for i in range(n_chunks):
    print(f'Processing chunk {i+1}...')

    if import_audio:
        
        start_sec += duration*overlap # adjust for overlap
    
        # Show raw signal
        t_axis, audio_time, audio_fft = import_signal(filepath, fs, nfft, start_sec, duration)
        if plot:
            plot_signal_log(t_axis, audio_time, f_axis, audio_fft, fs, nfft)
        
        # Show windowed signal
        t_axis, audio_time, audio_fft = window_signal(audio_time, fs, nfft)
        if plot:
            plot_signal_log(t_axis, audio_time, f_axis, audio_fft, fs, nfft)

        
    else:
    
        # ------------------------
        # Create audio 
        #audio_freq = np.array([406.25]) # 406.25 gives us perfect peak
        audio_freq = np.array([400, 500, 600])
        audio_amp = np.array([1,1,1])
        audio_phase = np.array([0,0,0])
        # audio_freq = np.array([400])
        # audio_amp = np.array([1])
        # audio_phase = np.array([0])
        zero_padd_sec = 0    # seconds, FOR NOW, MUST BE ZERO!!
        mag_clipp = 10e-8
        # ------------------------
    
        # Create signal
        t_axis, audio_time, audio_fft = create_pure_signal(duration, fs, nfft, audio_freq, audio_amp, audio_phase, zero_padd_sec=zero_padd_sec)
        if plot:
            plot_signal_log(t_axis, audio_time, f_axis, audio_fft, fs, nfft, mag_clipp)
        
        # Show windowed signal
        t_axis, audio_time, audio_fft = window_signal(audio_time, fs, nfft)
        if plot:
            plot_signal_log(t_axis, audio_time, f_axis, audio_fft, fs, nfft)

    
    
    
    # Windowed signal
    mag_clipp = 10e-8
    # t_axis, audio_time, audio_fft = window_signal(audio_time, fs, nfft)
    
    # Save information
    time_signals.append(audio_time)
    f_signals.append(audio_fft)
    


# %% Hanning Window

n = len(audio_time) # The individual length of each chunc
n_overlap = int(n/2)

audio_ones = np.ones(n)

t_axis_1, audio_time_1, audio_fft_2 = window_signal(audio_ones, fs, nfft)
audio_time_1_padded = np.pad(audio_time_1, (0, n_overlap), 'constant')
t_axis_2, audio_time_2, audio_fft_2 = window_signal(audio_ones, fs, nfft)
audio_time_2_padded = np.pad(audio_time_2, (n_overlap, 0), 'constant')

t_axis_full = np.arange(nfft + n_overlap) / fs
audio_time_sum = audio_time_1_padded + audio_time_2_padded

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))  # adjust size if needed

# --- First subplot: Time domain ---
ax1.plot(t_axis_full*1_000, audio_time_1_padded, color='tab:blue')
ax1.set_xlabel("Time [ms]")
ax1.set_ylabel("Magnitude")
ax1.set_title("Time Domain: Audio Signal")
ax1.grid(True)

# --- First subplot: Time domain ---
ax2.plot(t_axis_full*1_000, audio_time_2_padded, color='tab:blue')
ax2.set_xlabel("Time [ms]")
ax2.set_ylabel("Magnitude")
ax2.set_title("Time Domain: Audio Signal")
ax2.grid(True)

# --- First subplot: Time domain ---
ax3.plot(t_axis_full*1_000, audio_time_sum, color='tab:blue')
ax3.set_xlabel("Time [ms]")
ax3.set_ylabel("Magnitude")
ax3.set_title("Time Domain: Audio Signal")
ax3.grid(True)

# Adjust layout so titles and labels don't overlap
plt.tight_layout()
plt.show()


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

from myutils import compute_anechoic_H

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
# H_full = compute_anechoic_H(mics_locs, all_speakers, fs, nfft)
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


# %% Plotta layout för mikrofoner och högtalare (2D-vy)

plt.figure(figsize=(8, 8))

# 1. Plotta mikrofonerna (grid-punkterna)
# Vi använder X och Y som kommer från din 'create_uniform_rectangular_mic_grid'
plt.scatter(X, Y, c='lightgray', s=10, label='Microphone', alpha=0.5)

# 2. Plotta högtalarna
# all_speakers[0,:] är x-koordinater, all_speakers[1,:] är y-koordinater
plt.scatter(all_speakers[0, :], all_speakers[1, :], 
            c='blue', marker='s', s=50, label='Speaker')

# 3. Markera zonerna (Bright och Dark) för tydlighet
bright_circle = plt.Circle((bright_center[0], bright_center[1]), radius, 
                           color='yellow', alpha=0.3, label='Bright Zone')
dark_circle = plt.Circle((dark_center[0], dark_center[1]), radius, 
                         color='red', alpha=0.1, label='Dark Zone')

ax = plt.gca()
ax.add_patch(bright_circle)
ax.add_patch(dark_circle)

# Inställningar för grafen
plt.title(f"Layout: {num_speakers} Speakers & {num_mics} Microphones")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
# framealpha=1 makes the square non-transparent.
# facecolor='white' makes the background white.
plt.legend(loc='upper right', framealpha=1, facecolor='white', edgecolor='black')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal') # Viktigt för att skalan ska vara kvadratisk
plt.xlim([0, room_dim[0]])
plt.ylim([0, room_dim[1]])

plt.show()

# %% Calculate and visualize results

# 1. Standard Pressure Matching
p_full, g_full = calc_pressure_matching(room, nfft, H_full, bright_indices, dark_indices)

# 2. Smooth Pressure Matching 
print('='*50)
print("--- Standard PM ---")
p_full_smooth, g_full_smooth = calc_smooth_pressure_matching(room, nfft, H_full, bright_indices, dark_indices)

pressure_map = get_energy_map_db_sara(p_full, f_signals[0], X.shape)

# Visualize
plot_pressure_map(
    pressure_map, X, Y, all_speakers, 
    bright_center, dark_center, radius, 
    title=""
)

# %% Listen to results

from myutils import (simulate_listening_points_sara_stitched, 
                     calculate_broadband_contrast)

# Run the simulation for the two specific points
bright_norm, dark_norm = simulate_listening_points_sara_stitched(
room_dim, fs, all_speakers, g_full, bright_center, dark_center, 
nfft, nbr_bounces, n_chunks, overlap=overlap, time_signals=time_signals, f_signals=f_signals, duration=duration)

# Calculate the actual achieved contrast
achieved_contrast = calculate_broadband_contrast(bright_norm, dark_norm)
print('='*50)
print(f"Final Broadband Acoustic Contrast: {achieved_contrast:.2f} dB")
print('='*50)

# contrast = evaluate_acoustic_contrast(p_full, audio_freq, audio_amp, fs, nfft, bright_indices, dark_indices)
# print('='*50)
# print(f"Acoustic Contrast: {contrast:.2f} dB")
# print('='*50)


# Save the results
save_as_wav("pm_bright_zone_center.wav", bright_norm, fs)
save_as_wav("pm_dark_zone_center.wav", dark_norm, fs)

# Save as a single sequential file
save_combined_wav("pm_combined_zones.wav", bright_norm, dark_norm, fs, pause_duration=1.0)


# %% View wav files

from myutils import calculate_sliding_contrast


plot_audio_analysis(
    "pm_bright_zone_center.wav", 
    "pm_dark_zone_center.wav", 
    time_zoom=(0.00, 0.060), 
    fs = fs, 
    freq_range=(0, 8000)
)

#%% Analyze chunked contrast

from myutils import plot_chunked_contrast

# This replaces the sliding window logic and plots both contrast and raw energy
t_chunks, contrast_chunks, energy_chunks = plot_chunked_contrast(
    bright_norm, 
    dark_norm, 
    fs, 
    chunk_sec=0.04 # 40ms non-overlapping chunks
)




#%%

# 1. Calculate the sliding contrast
# window_sec=0.04 (40ms) is a good default for speech analysis
t_contrast, contrast_vals = calculate_sliding_contrast(bright_norm, dark_norm, fs, window_sec=0.04, overlap=overlap)

# 2. Plotting
plt.figure(figsize=(12, 4))
plt.plot(t_contrast, contrast_vals, color='purple', linewidth=1.5)
plt.axhline(y=np.mean(contrast_vals), color='black', linestyle='--', label=f'Avg: {np.mean(contrast_vals):.1f} dB')

plt.title("Time-Varying Acoustic Contrast")
plt.xlabel("Time [s]")
plt.ylabel("Contrast [dB]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% Only for Sara's Run All button to work

# This has to be in the end of the file for Sara's figures to not close down
plt.ioff()
plt.show()
# %%
