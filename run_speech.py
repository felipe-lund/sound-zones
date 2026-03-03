import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from myutils import (
    calc_pressure_matching,
    create_rectangular_perimeter_speaker_array,
    create_uniform_rectangular_mic_grid,
    get_energy_map_db,
    get_zone_indices,
    plot_audio_analysis,
    plot_pressure_map,
    save_as_wav,
    simulate_listening_points,
)


# Parameteres
nbr_bounces = 3 # max_order: max number of reflections in the room
air_absorption = True # air_absorbtion: if air is absorbed or not
fs = 16000   # Sampling freq. of the played audio
room_dim = [5.0, 5.0, 5.0]
num_speakers_per_row = 10
mic_spacing = 0.5
nfft = 512*2  # for the fourier transfrom
f_axis = np.fft.rfftfreq(nfft, d=1/fs)


# %%  Create room
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


# %% Compute Room impulse Response

print('Computing RIR...')
room.compute_rir()

# %% A 3D matrix to save the fft of each RIR
print('Computing H...')
H_full = np.zeros((num_mics, num_speakers, nfft//2 + 1), dtype=complex)    # All zeros in the beginning
for m in range(num_mics):
    for s in range(num_speakers):
        H_full[m, s, :] = np.fft.rfft(room.rir[m][s], n=nfft)

#%%  Create signal

# Selecting params for signal
audio_freq = np.array([400, 500, 600])
audio_amp = np.array([1, 1, 1])
audio_phase = np.array([np.pi, np.pi/2, 0])
duration = 2.0  # seconds
c = pra.constants.get('c')  # Speed of sound

# Creating signal as some of frequencies
t_axis = np.arange(0, duration, 1/fs)    
audio_time = np.zeros(len(t_axis))
for idx, freq in enumerate(audio_freq):
    if len(audio_amp) == 1:
        # all the parts of the signal have the same frequency
        audio_time += audio_amp * np.sin(2 * np.pi * freq * t_axis  + audio_phase)
    else:
        audio_time += audio_amp[idx] * np.sin(2 * np.pi * freq * t_axis  + audio_phase[idx])

# The frequencies of the signal (division of nfft/2 for the magnitude to be correct)
audio_fft = np.fft.rfft(audio_time, n=nfft) 

plt.plot(t_axis[:int(0.01 * fs)]*1_000, audio_time[:int(0.01 * fs)])
plt.xlabel("Time [ms]")
plt.ylabel("Magnitude")
plt.title("The Audio Signal")
plt.grid(True)
plt.show()

plt.plot(f_axis, np.abs(audio_fft)/(nfft/2))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT of audio signal")
plt.grid(True)
plt.show()


# %% Define Zones
radius = 0.5
bright_center = np.array([1.5, 3.0, 2.5])
dark_center = np.array([3.5, 3.0, 2.5])
bright_indices, dark_indices = get_zone_indices(
    mics_locs, bright_center, dark_center, radius
)

print(f"Mics in Bright Zone: {len(bright_indices)}")
print(f"Mics in Dark Zone: {len(dark_indices)}")


#%% Calculate and visualize results

# Calc p and g vectors
p_full, g_full = calc_pressure_matching(room, nfft, H_full, bright_indices, dark_indices)

# Calculate the pressure map using extracted data
pressure_map = get_energy_map_db(
    p_full, audio_freq, audio_amp, room.fs, nfft, X.shape
)

# Visualize
plot_pressure_map(
    pressure_map, X, Y, all_speakers, 
    bright_center, dark_center, radius, 
    title=f"Pressure Matching: {audio_freq} Hz"
)


#%% Listen to results

# Run the simulation for the two specific points
bright_norm, dark_norm = simulate_listening_points(
    room_dim, fs, all_speakers, g_full, 
    audio_freq, audio_amp, nfft, 
    bright_center, dark_center
)

# Save the results
save_as_wav("pm_bright_zone_center.wav", bright_norm, fs)
save_as_wav("pm_dark_zone_center.wav", dark_norm, fs)

print("Audio files saved! Go listen to them.")


#%% View wav files

plot_audio_analysis(
    "pm_bright_zone_center.wav", 
    "pm_dark_zone_center.wav",
    freq_range=(300, 700),  # Zoomed into our 400, 500, 600 Hz signals
    time_zoom=(0.5, 0.55)
)
