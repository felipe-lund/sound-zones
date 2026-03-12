

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

# %%

def create_signal(duration, fs, audio_freq, audio_amp, audio_phase, zero_padd_sec = 0):
    
    # Creating signal as sum of frequencies
    t_axis = np.arange(0, duration, 1/fs)   
    audio_time = np.zeros(len(t_axis))
    
    nfft = len(audio_time) # Avoids spectral leakage

    for idx, freq in enumerate(audio_freq):
        audio_time += audio_amp[idx] * np.sin(2 * np.pi * freq * t_axis + audio_phase[idx])
    audio_fft = np.fft.rfft(audio_time, n=nfft) 

    # if zero_padd_sec != 0:
    #     audio_time = np.pad(audio_time, (0, round(zero_padd_sec *fs)), 'constant')
    #     t_axis = np.arange(0, duration + zero_padd_sec, 1/fs)  
    #     audio_fft = np.fft.rfft(audio_time)  # This doesn't work to use yet...

    # audio_fft = np.fft.rfft(audio_time, n=int(nfft/2)) 

    return t_axis, audio_time, audio_fft

def plot_signal(t_axis, audio_time, f_axis, audio_fft, fs, nfft, mag_clipp=1e-6):
    # Create a figure with 2 subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # adjust size if needed

    # --- First subplot: Time domain ---
    ax1.plot(t_axis*1_000, audio_time, color='tab:blue')
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Time Domain: Audio Signal")
    ax1.grid(True)

    # --- Second subplot: Frequency domain ---
    # Inside your function:
    magnitude = np.abs(audio_fft) / (nfft / 2)
    # Set a floor so values don't drop to 10e-16
    magnitude_clipped = np.maximum(magnitude, mag_clipp) 

    ax2.semilogy(f_axis, magnitude_clipped, color='tab:orange')
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Frequency Domain: FFT of Audio Signal")
    ax2.grid(True)

    # Adjust layout so titles and labels don't overlap
    plt.tight_layout()
    plt.show()

# %% Parameters

nbr_bounces = 0 # max_order: max number of reflections in the room
air_absorption = True # air_absorbtion: if air is absorbed or not
fs = 16_000   # Sampling freq. of the played audio
room_dim = [5.0, 5.0, 5.0]
num_speakers_per_row = 10
# mic_spacing = 0.5
mic_spacing = 0.1

print('='*50)
print(
    f"Room Properties:\n"
    f"\tNumber of bounces: {nbr_bounces}\n"
    f"\tAir Absorption:    {air_absorption}\n"
    f"\tRoom Dimentions:   {room_dim}"
)
print('='*50)

# %% Importing an audio

import_audio = False

duration = 40e-3        # seconds

audio_freq = np.array([400])

# Använd perfect_freq i create_signal

audio_amp = np.array([1])
audio_phase = np.array([0])
zero_padd_sec = 0    # seconds, FOR NOW, MUST BE ZERO!!
mag_clipp = 10e-4

t_axis, audio_time, audio_fft = create_signal(duration, fs, audio_freq, audio_amp, audio_phase, zero_padd_sec=zero_padd_sec)

# ------------------------------------------------------
# IMPORTANT
# To avoid spectral leakage, it is essential to avoid spectral leakage 
# at the beginning and at the end of a signal. The below definition of
# NFFT makes the signal of 40 ms have an integer amount of periods, 
# which makes the spectral leakage be zero and a perfect peak appears.
nfft = len(audio_time)
# ------------------------------------------------------

f_axis = np.fft.rfftfreq(nfft, d=1/fs)

# f_axis = np.fft.rfftfreq((len(audio_fft) - 1 )*2, d=1/fs)
plot_signal(t_axis, audio_time, f_axis, audio_fft, fs, nfft, mag_clipp)


print('='*50)
print(
    f"Audio Properties:\n"
    f"\tImported Audio:    {import_audio}\n"
    f"\tSampling freq:     {fs} [Hz]\n"
    f"\tNFFT:              {nfft}\n"
    f"\tDuration:          {duration} [s]"
)
print('='*50)

    

# %%
