import os
import json
import hashlib
import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wav
import sounddevice as sd
import time
from matplotlib import pyplot as plt




def create_rectangular_perimeter_speaker_array(room_dim, num_per_side, margin=0.5, height=2.5):
    """
    Creates a rectangular ring of speakers along the four walls.
    """
    x_min, y_min = margin, margin
    x_max, y_max = room_dim[0] - margin, room_dim[1] - margin
    
    # Linear spacing for the speakers along the segments
    # We use a slight inset so they don't overlap at corners
    pos_space = np.linspace(x_min + 0.5, x_max - 0.5, num_per_side)
    
    # Bottom wall (Front)
    front = np.vstack((pos_space, np.full(num_per_side, y_min), np.full(num_per_side, height)))
    # Right wall
    right = np.vstack((np.full(num_per_side, x_max), pos_space, np.full(num_per_side, height)))
    # Top wall (Back)
    back  = np.vstack((pos_space, np.full(num_per_side, y_max), np.full(num_per_side, height)))
    # Left wall
    left  = np.vstack((np.full(num_per_side, x_min), pos_space, np.full(num_per_side, height)))
    
    return np.hstack((front, right, back, left))



def create_uniform_rectangular_mic_grid(room_dim, spacing=0.5, margin=0.6, height=2.5):
    """
    Creates a flat grid of microphones across the room area.
    """
    x_range = np.arange(margin, room_dim[0] - margin, spacing)
    y_range = np.arange(margin, room_dim[1] - margin, spacing)
    X, Y = np.meshgrid(x_range, y_range)
    
    mics_locs = np.vstack((X.flatten(), Y.flatten(), np.full(X.size, height)))
    return mics_locs, X, Y


def get_zone_indices(mics_locs, bright_center, dark_center, radius):
    """
    Finds which microphone indices fall inside circular zones.
    """
    bright_indices = []
    dark_indices = []

    # Iterate through every microphone column in the matrix
    # num_mics is the number of columns (axis 1)
    num_mics = mics_locs.shape[1]

    for i in range(num_mics):
        # Extract the [x, y, z] position of the current mic
        current_mic_pos = mics_locs[:, i]

        # 1. Check if mic is in the Bright Zone
        dist_b = np.linalg.norm(current_mic_pos - bright_center)
        if dist_b <= radius:
            bright_indices.append(i)

        # 2. Check if mic is in the Dark Zone
        dist_d = np.linalg.norm(current_mic_pos - dark_center)
        if dist_d <= radius:
            dark_indices.append(i)

    # Convert lists to numpy arrays to stay compatible with your existing math
    return np.array(bright_indices), np.array(dark_indices)

def calc_pressure_matching(room, nfft, H_full, bright_indices, dark_indices):
    # 1. Extract physical dimensions and counts
    num_mics = room.mic_array.R.shape[1]      # Get count from the mic array object
    num_speakers = len(room.sources)          # Get count from the sources list
    fs = room.fs                              # Get sampling frequency
    
    # 2. Extract coordinates for plane wave math
    mics_locs = room.mic_array.R              # The 3xN coordinate matrix
    
    # 3. Handle physics constants and axes
    c = pra.constants.get('c')                # Speed of sound
    f_axis = np.fft.rfftfreq(nfft, d=1/fs)    # Re-calculate the frequency axis
    
    p_full = np.zeros((num_mics,     nfft//2 + 1), dtype=complex)
    g_full = np.zeros((num_speakers, nfft//2 + 1), dtype=complex)
    for idx, f in enumerate(f_axis):

        # Selecting H
        H = H_full[:, :, idx]
        Hb = H[bright_indices, :]
        Hd = H[dark_indices, :]

        # Bright zone: Plane wave traveling from left to right
        k = 2 * np.pi * f / c   
        theta = 0.0 # Angle of incidence (0 radians = left to right)
        # theta = np.pi / 2
        kx = k * np.cos(theta)
        ky = k * np.sin(theta)

        # Get x and y coordinates of ONLY the bright zone mics
        x_bright = mics_locs[0, bright_indices]
        y_bright = mics_locs[1, bright_indices]

        # The plane wave equation for the bright zone
        p_des = np.exp(-1j * (kx * x_bright + ky * y_bright))

        # Lambda params
        lambda_1 = 1.0   # Weighting for dark zone energy minimization (simplest formulation sets this to 1)
        lambda_2 = 1e-2  # Robustness constraint on array effort (Lagrange multiplier 2)

        # Calculate the components of the equation
        Hb_H_Hb = Hb.conj().T @ Hb
        Hd_H_Hd = Hd.conj().T @ Hd
        I = np.eye(num_speakers)

        # Solve for g
        R_matrix = Hb_H_Hb + lambda_1 * Hd_H_Hd + lambda_2 * I
        right_side = Hb.conj().T @ p_des
        g = np.linalg.inv(R_matrix) @ right_side

        g_full[:, idx] = g
        p_full[:, idx] = H @ g

    return p_full, g_full


def calc_smooth_pressure_matching(room, nfft, H_full, bright_indices, dark_indices):
    # 1. Extract physical dimensions and counts
    num_mics = room.mic_array.R.shape[1]      
    num_speakers = len(room.sources)          
    fs = room.fs                              
    
    # 2. Extract coordinates for plane wave math
    mics_locs = room.mic_array.R              
    
    # 3. Handle physics constants and axes
    c = pra.constants.get('c')                
    f_axis = np.fft.rfftfreq(nfft, d=1/fs)    
    
    # -------------------------------------------------------------
    # Build the Spatial Smoothness (Laplacian) Matrix
    # -------------------------------------------------------------
    # Get the 3D coordinates of just the bright zone mics
    b_coords = mics_locs[:, bright_indices]
    N_b = b_coords.shape[1]
    
    # Calculate pairwise distances between all bright zone mics
    diffs = b_coords[:, :, np.newaxis] - b_coords[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=0)
    
    # Dynamically find the mic spacing (the smallest non-zero distance)
    if N_b > 1:
        mic_spacing = np.min(dists[dists > 0])
        # Find immediate neighbors (distance roughly equal to mic_spacing)
        adjacency = (dists > 0) & (dists < mic_spacing * 1.1)
    else:
        adjacency = np.zeros((1, 1), dtype=bool)
        
    # Build the Laplacian: Diagonal is number of neighbors, -1 for actual neighbors
    degrees = np.sum(adjacency, axis=1)
    L_lap = np.diag(degrees) - adjacency.astype(float)
    # -------------------------------------------------------------

    p_full = np.zeros((num_mics, nfft//2 + 1), dtype=complex)
    g_full = np.zeros((num_speakers, nfft//2 + 1), dtype=complex)
    
    for idx, f in enumerate(f_axis):
        H = H_full[:, :, idx]
        Hb = H[bright_indices, :]
        Hd = H[dark_indices, :]

        k = 2 * np.pi * f / c   
        theta = 0.0 
        kx = k * np.cos(theta)
        ky = k * np.sin(theta)

        x_bright = mics_locs[0, bright_indices]
        y_bright = mics_locs[1, bright_indices]

        p_des = np.exp(-1j * (kx * x_bright + ky * y_bright))

        # Lambda params (Tuning Knobs)
        lambda_1 = 1.0       # Dark zone penalty
        lambda_2 = 1e-2      # Speaker effort penalty
        lambda_smooth = 1e-2 # Spatial smoothness penalty
        
        Hb_H_Hb = Hb.conj().T @ Hb
        Hd_H_Hd = Hd.conj().T @ Hd
        I = np.eye(num_speakers)

        # NEW: Inject the Laplacian penalty into the R matrix
        Hb_H_L_Hb = Hb.conj().T @ L_lap @ Hb
        
        R_matrix = Hb_H_Hb + lambda_1 * Hd_H_Hd + lambda_2 * I + lambda_smooth * Hb_H_L_Hb
        
        right_side = Hb.conj().T @ p_des
        g = np.linalg.inv(R_matrix) @ right_side

        g_full[:, idx] = g
        p_full[:, idx] = H @ g

    return p_full, g_full


def get_energy_map_db(p_full, audio_freq, audio_amp, fs, nfft, grid_shape):
    """
    Combines pressure fields from multiple frequencies into a normalized dB map.
    """
    num_mics = p_full.shape[0]
    energy_tot = np.zeros(num_mics)

    # Accumulate energy (magnitude squared) for target frequencies
    for i, target_freq in enumerate(audio_freq):
        freq_bin = int(np.round((target_freq / fs) * nfft))
        p_freq = p_full[:, freq_bin]
        energy_tot += np.abs(audio_amp[i] * p_freq)**2

    # RMS and dB conversion
    p_total_rms = np.sqrt(energy_tot)
    p_dB = 20 * np.log10(p_total_rms + 1e-12)
    
    # Normalize: Loudest point becomes 0 dB
    p_dB -= np.max(p_dB) 

    return p_dB.reshape(grid_shape)

def get_energy_map_db_sara(p_full, audio_fft, grid_shape):
    """
    Combines pressure fields from multiple frequencies into a normalized dB map.
    """
    num_mics = p_full.shape[0]          # M (total number of mics)
    energy_tot = np.zeros(num_mics)     # We want to calculate the energy each mic hears (starting with zero)

    # Accumulate energy (magnitude squared) for target frequencies
    for i in range(len(audio_fft)):

        # p_freq: the amplitude and phase change that every mic experiences for the studied frequency f
        p_freq = p_full[:, i]  
        # energy_freq: the energy each mic experiences by this frequency f is 
        # the amplitude of the original sound at that freq times the change of the amplitude
        energy_freq = np.abs(audio_fft[i] * p_freq)**2
        # The total energy of each mic is the sum of the energy of all frequencies for that mic
        energy_tot += energy_freq

    # RMS and dB conversion
    p_total_rms = np.sqrt(energy_tot)
    p_dB = 20 * np.log10(p_total_rms + 1e-12)
    
    # Normalize: Loudest point becomes 0 dB
    p_dB -= np.max(p_dB) 

    return p_dB.reshape(grid_shape)

def plot_pressure_map(pressure_map, X, Y, all_speakers, bright_center, dark_center, radius, title):
    """
    Renders the acoustic heatmap with zone and speaker overlays.
    """
    plt.figure(figsize=(10, 6))
    
    min_db = -80
    max_db = 0
    levels = np.linspace(min_db, max_db, 51)
    
    # Plot heatmap
    cont = plt.contourf(X, Y, pressure_map, levels=levels, cmap='inferno')
    plt.colorbar(cont, label="Relative Sound Pressure (dB)")

    # Plot Speakers
    plt.scatter(all_speakers[0, :], all_speakers[1, :], 
                color='white', edgecolors='black', marker='^', s=50, label='Speakers')

    # Add Zone Circles
    bright_circle = plt.Circle(bright_center[:2], radius, color='red', fill=False, 
                               linestyle='--', linewidth=2, label='Bright Zone')
    dark_circle = plt.Circle(dark_center[:2], radius, color='blue', fill=False, 
                             linestyle='--', linewidth=2, label='Dark Zone')
    
    plt.gca().add_patch(bright_circle)
    plt.gca().add_patch(dark_circle)

    plt.title(title)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend(loc='upper right')
    plt.show()


def simulate_listening_points(room_dim, fs, all_speakers, g_full, audio_freq, audio_amp, nfft, bright_center, dark_center, duration=2.0):
    """
    Simulates the audio received at the centers of the bright and dark zones.
    """
    t = np.arange(0, duration, 1/fs)
    num_speakers = all_speakers.shape[1]
    
    # 1. Create the listening room (ShoeBox with same properties)
    listening_room = pra.ShoeBox(room_dim, fs=fs, max_order=3, air_absorption=True)

    # 2. Synthesize signals for each speaker
    for i in range(num_speakers):
        total_signal = np.zeros(len(t))
        for idx, freq in enumerate(audio_freq):
            # Extract weights from g_full
            freq_bin = int(np.round((freq / fs) * nfft))
            freq_g = g_full[:, freq_bin]
            
            mag = np.abs(freq_g[i])
            phase = np.angle(freq_g[i])
            amp = audio_amp[idx]
            
            # Sum individual sine waves
            total_signal += amp * mag * np.sin(2 * np.pi * freq * t + phase)
        
        listening_room.add_source(all_speakers[:, i], signal=total_signal)

    # 3. Add microphones at zone centers
    # Using the intuitive list approach
    mics_coords = np.array([
        [bright_center[0], bright_center[1], bright_center[2]],
        [dark_center[0], dark_center[1], dark_center[2]]
    ]).T
    
    listening_room.add_microphone_array(pra.MicrophoneArray(mics_coords, fs))

    # 4. Run simulation
    print("Simulating audio for playback...")
    listening_room.simulate()

    # 5. Extract and Normalize
    bright_audio = listening_room.mic_array.signals[0, :]
    dark_audio = listening_room.mic_array.signals[1, :]
    
    # Preserve relative volume difference
    max_val = max(np.max(np.abs(bright_audio)), np.max(np.abs(dark_audio)))
    if max_val == 0: max_val = 1 # Avoid division by zero
    
    return bright_audio / max_val, dark_audio / max_val

def simulate_listening_points_sara(room_dim, fs, all_speakers, g_full, audio_fft, bright_center, dark_center, nfft, nbr_bounces):
    """
    Simulates the audio received at the centers of the bright and dark zones.
    """
    num_speakers = all_speakers.shape[1]
    
    # 1. Create the listening room (ShoeBox with same properties)
    listening_room = pra.ShoeBox(room_dim, fs=fs, max_order=nbr_bounces, air_absorption=True)

    # 2. Synthesize signals for each speaker
    for i in range(num_speakers):

        # Every freq. of the original audio is multiplied by its corresponding filter for that speaker
        speaker_audio_freq_domain = audio_fft * g_full[i, :]
        # signal_speaker_i = np.zeros(len(t_axis))

        speaker_audio_time_domain = np.fft.irfft(speaker_audio_freq_domain, n=nfft)
        
        # speaker_audio_time_domain has the length nfft
        # print(f"Lenght of the speaker sound {len(speaker_audio_time_domain)}")

        listening_room.add_source(all_speakers[:, i], signal=speaker_audio_time_domain)

    # 3. Add microphones at zone centers
    # Using the intuitive list approach
    mics_coords = np.array([
        [bright_center[0], bright_center[1], bright_center[2]],
        [dark_center[0], dark_center[1], dark_center[2]]
    ]).T
    
    listening_room.add_microphone_array(pra.MicrophoneArray(mics_coords, fs))

    # 4. Run simulation
    print("Simulating audio for playback...")
    listening_room.simulate()

    # 5. Extract and Normalize
    bright_audio = listening_room.mic_array.signals[0, :]
    dark_audio = listening_room.mic_array.signals[1, :]
    
    # Preserve relative volume difference
    max_val = max(np.max(np.abs(bright_audio)), np.max(np.abs(dark_audio)))
    if max_val == 0: max_val = 1 # Avoid division by zero
    
    return bright_audio / max_val, dark_audio / max_val


def simulate_listening_points_sara_stitched(room_dim, fs, all_speakers, g_full, bright_center, dark_center, nfft, nbr_bounces, n_chunks, overlap: float, time_signals: list, f_signals: list, duration: float):
    """
    Simulates the audio received at the centers of the bright and dark zones.
    """
    num_speakers = all_speakers.shape[1]
    
    # 1. Calculate precise lengths and step sizes
    samples_per_chunk = int(round(fs * duration))
    
    # Define how many samples we step forward for each chunk
    step_size = int(round(samples_per_chunk * overlap)) 
    
    # Total length MUST accommodate the start index of the last chunk PLUS the full nfft tail
    total_length = (n_chunks - 1) * step_size + nfft
    
    # 2. Create the listening room (ShoeBox with same properties)
    listening_room = pra.ShoeBox(room_dim, fs=fs, max_order=nbr_bounces, air_absorption=True)

    # 3. Synthesize signals for each speaker
    for i in range(num_speakers):
        
        # Pre-allocate array for the entire stitched signal
        full_speaker_audio_time_domain = np.zeros(total_length)
        
        for j, audio_fft in enumerate(f_signals):
            
            # Apply the spatial filter in frequency domain
            speaker_audio_freq_domain = audio_fft * g_full[i, :]
    
            # Transform back to time domain with FULL nfft length
            speaker_audio_time_domain = np.fft.irfft(speaker_audio_freq_domain, n=nfft)
            
            # Calculate start and end indices using the step size and nfft
            start_idx = j * step_size
            end_idx = start_idx + nfft
            
            # Add the FULL chunk to the total signal (Overlap-Add)
            full_speaker_audio_time_domain[start_idx:end_idx] += speaker_audio_time_domain

        # Add the completed, stitched signal to the room for this speaker
        listening_room.add_source(all_speakers[:, i], signal=full_speaker_audio_time_domain)

    # 4. Add microphones at zone centers
    mics_coords = np.array([
        [bright_center[0], bright_center[1], bright_center[2]],
        [dark_center[0], dark_center[1], dark_center[2]]
    ]).T
    
    listening_room.add_microphone_array(pra.MicrophoneArray(mics_coords, fs))

    # 5. Run simulation
    print("Simulating audio for playback...")
    listening_room.simulate()

    # 6. Extract and Normalize
    bright_audio = listening_room.mic_array.signals[0, :]
    dark_audio = listening_room.mic_array.signals[1, :]
    
    # Preserve relative volume difference
    max_val = max(np.max(np.abs(bright_audio)), np.max(np.abs(dark_audio)))
    if max_val == 0: 
        max_val = 1 # Avoid division by zero
    
    return bright_audio / max_val, dark_audio / max_val


def simulate_listening_points_sara_stitched2(room_dim, fs, all_speakers, g_full, bright_center, dark_center, nfft, nbr_bounces, n_chunks, overlap: float, time_signals: list, f_signals: list, duration: float):
    """
    Simulates the audio received at the centers of the bright and dark zones using proper Overlap-Add.
    """
    num_speakers = all_speakers.shape[1]
    
    # 1. Calculate precise lengths and step sizes based on the UNPADDED chunk
    samples_per_chunk = int(round(fs * duration))
    
    # Corrected step size: Hop forward by the non-overlapping portion
    step_size = int(round(samples_per_chunk * (1.0 - overlap))) 
    
    # Total length MUST accommodate the start index of the last chunk PLUS the full nfft tail
    total_length = (n_chunks - 1) * step_size + nfft
    
    # 2. Create the listening room (ShoeBox with same properties)
    listening_room = pra.ShoeBox(room_dim, fs=fs, max_order=nbr_bounces, air_absorption=True)

    # 3. Synthesize signals for each speaker
    for i in range(num_speakers):
        
        # Pre-allocate array for the entire stitched signal
        full_speaker_audio_time_domain = np.zeros(total_length)
        
        for j, audio_fft in enumerate(f_signals):
            
            # Apply the spatial filter in frequency domain
            speaker_audio_freq_domain = audio_fft * g_full[i, :]
    
            # Transform back to time domain with FULL nfft length
            speaker_audio_time_domain = np.fft.irfft(speaker_audio_freq_domain, n=nfft)
            
            # Calculate start and end indices using the STEP SIZE (not nfft)
            start_idx = j * step_size
            end_idx = start_idx + nfft
            
            # Add the FULL chunk to the total signal (Overlap-Add)
            # This correctly overlaps the zero-padding and filter tails into the next frame!
            full_speaker_audio_time_domain[start_idx:end_idx] += speaker_audio_time_domain

        # Add the completed, stitched signal to the room for this speaker
        listening_room.add_source(all_speakers[:, i], signal=full_speaker_audio_time_domain)

    # 4. Add microphones at zone centers
    mics_coords = np.array([
        [bright_center[0], bright_center[1], bright_center[2]],
        [dark_center[0], dark_center[1], dark_center[2]]
    ]).T
    
    listening_room.add_microphone_array(pra.MicrophoneArray(mics_coords, fs))

    # 5. Run simulation
    print("Simulating audio for playback...")
    listening_room.simulate()

    # 6. Extract and Normalize
    bright_audio = listening_room.mic_array.signals[0, :]
    dark_audio = listening_room.mic_array.signals[1, :]
    
    # Preserve relative volume difference
    max_val = max(np.max(np.abs(bright_audio)), np.max(np.abs(dark_audio)))
    if max_val == 0: 
        max_val = 1 # Avoid division by zero
    
    return bright_audio / max_val, dark_audio / max_val

def save_as_wav(filename, signal, fs):
    """Helper to convert float signal to 16-bit PCM and save."""
    scaled = np.int16(signal * 32767)
    wav.write(filename, fs, scaled)



def plot_audio_analysis(bright_wav_path, dark_wav_path, freq_range=(0, 8000), time_zoom=(0.1, 0.15), fs = 16000):
    """
    Plots time-domain waveforms and frequency-domain spectra for two wav files.
    """
    # 1. Load the signals
    fs_b, b_data_full = wav.read(bright_wav_path)
    fs_d, d_data_full = wav.read(dark_wav_path)

    idx_start = int(time_zoom[0] * fs)

    if int(time_zoom[1] * fs)  < len(b_data_full):
        idx_end   = int(time_zoom[1] * fs)  
    else:
        idx_end   = len(b_data_full)

    print(f"Sampling frequency:      {fs_b}")
    print(f"Tot length of the sound: {len(b_data_full)}")

    b_data_cut = b_data_full[idx_start:idx_end]
    d_data_cut = d_data_full[idx_start:idx_end]

    # Lägg till nollor före (idx_start) och efter (resten upp till originalet)
    b_data_padded = np.pad(b_data_cut, (idx_start, len(b_data_full) - idx_end), 'constant')
    d_data_padded = np.pad(d_data_cut, (idx_start, len(d_data_full) - idx_end), 'constant')

    # Standardize time and frequency axes
    t_axis_full = np.arange(len(b_data_full)) / fs_b
    freq_axis = np.fft.rfftfreq(len(b_data_cut), 1/fs_b)

    # 2. Perform FFT (Magnitude Spectrum)
    b_fft_mag = np.abs(np.fft.rfft(b_data_cut))
    d_fft_mag = np.abs(np.fft.rfft(d_data_cut))

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Time Domain Plot
    ax1.plot(t_axis_full, b_data_full, label='Bright Zone', color='red', alpha=0.7)
    ax1.plot(t_axis_full, d_data_full, label='Dark Zone', color='blue', alpha=0.7)
    ax1.set_title("Time Domain: Bright vs Dark Zone Waveforms")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(time_zoom) 
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Frequency Domain Plot
    ax2.semilogy(freq_axis, b_fft_mag, label='Bright Zone', color='red')
    ax2.semilogy(freq_axis, d_fft_mag, label='Dark Zone', color='blue')
    ax2.set_title("Frequency Domain: Magnitude Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(freq_range) 
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    

def play_audio_directly(bright_audio, dark_audio, fs):
    """
    Plays the raw numpy arrays directly through the system's audio hardware,
    preserving exact relative amplitudes.
    """
    print("🔊 Playing Bright Zone...")
    sd.play(bright_audio, samplerate=fs)
    sd.wait()  # Block execution until the audio finishes playing
    
    print('Sleeping for 1 second...')
    time.sleep(1)

    print("🔉 Playing Dark Zone...")
    sd.play(dark_audio, samplerate=fs)
    sd.wait()
    
    print("✅ Playback complete.")
    
    
    
def save_combined_wav(filename, bright_audio, dark_audio, fs, pause_duration=1.0):
    """
    Stitches the Bright Zone and Dark Zone audio together with a pause in between,
    ensuring media players normalize them as a single continuous track.
    """
    # Create an array of zeros for the silence
    silence = np.zeros(int(pause_duration * fs))
    
    # Concatenate the arrays: Bright -> Silence -> Dark
    combined_signal = np.concatenate((bright_audio, silence, dark_audio))
    
    # Use your existing save_as_wav helper (which handles the 16-bit PCM scaling)
    save_as_wav(filename, combined_signal, fs)
    
    print(f"Combined audio saved to {filename}! (Listen for the volume drop after the pause)")
    
    
def slow_compute_H(num_mics, num_speakers, room, nfft):
    print('Computing H...')
    H_full = np.zeros((num_mics, num_speakers, nfft//2 + 1), dtype=complex)    # All zeros in the beginning
    for m in range(num_mics):
        for s in range(num_speakers):
            H_full[m, s, :] = np.fft.rfft(room.rir[m][s], n=nfft)
    return H_full
    
    
    
def quick_compute_H(num_mics, num_speakers, room, nfft):

    # 1. Find the maximum length of any RIR
    max_rir_len = max(len(rir) for mic_list in room.rir for rir in mic_list)

    # 2. Pad all RIRs to max_rir_len and create the 3D array
    # We create a zero array and fill it to avoid "inhomogeneous shape" errors
    rir_array = np.zeros((num_mics, num_speakers, max_rir_len))
    for m in range(num_mics):
        for s in range(num_speakers):
            rir_array[m, s, :len(room.rir[m][s])] = room.rir[m][s]

    # 3. Vectorized FFT (very fast)
    H_full = np.fft.rfft(rir_array, n=nfft, axis=-1)
    
    return H_full
    

def evaluate_zone_smoothness(p_full, audio_freq, audio_amp, fs, nfft, bright_indices):
    """
    Calculates the spatial standard deviation and peak-to-peak ripple 
    of the energy map specifically inside the bright zone.
    """
    energy_tot = np.zeros(len(bright_indices))

    # Accumulate energy for target frequencies (just in the bright zone)
    for i, target_freq in enumerate(audio_freq):
        freq_bin = int(np.round((target_freq / fs) * nfft))
        p_freq = p_full[bright_indices, freq_bin]
        energy_tot += np.abs(audio_amp[i] * p_freq)**2

    # Convert to dB
    p_dB = 20 * np.log10(np.sqrt(energy_tot) + 1e-12)
    
    # 1. Spatial Standard Deviation (The standard metric for uniformity)
    std_db = np.std(p_dB)
    
    # 2. Peak-to-Peak Ripple (Max diff between the loudest and quietest mic)
    ripple_db = np.max(p_dB) - np.min(p_dB)
    
    return std_db, ripple_db
    
    
    
def evaluate_acoustic_contrast(p_full, audio_freq, audio_amp, fs, nfft, bright_indices, dark_indices):
    """
    Calculates the broadband Acoustic Contrast (in dB) between the bright and dark zones.
    A higher value means better isolation (the dark zone is quieter).
    """
    energy_b = 0.0
    energy_d = 0.0

    # Accumulate the mean squared pressure for the target frequencies
    for i, target_freq in enumerate(audio_freq):
        freq_bin = int(np.round((target_freq / fs) * nfft))
        
        # Extract pressures for this frequency bin
        p_b = p_full[bright_indices, freq_bin]
        p_d = p_full[dark_indices, freq_bin]
        
        # Calculate mean energy, weighted by the amplitude of the signal
        energy_b += np.mean(np.abs(audio_amp[i] * p_b)**2)
        energy_d += np.mean(np.abs(audio_amp[i] * p_d)**2)

    # Calculate contrast in dB (with 1e-12 to avoid division by zero)
    contrast_dB = 10 * np.log10((energy_b + 1e-12) / (energy_d + 1e-12))
    
    return contrast_dB



def get_or_compute_H(room, nfft, params_dict, cache_dir="cached_rir"):
    """
    Checks if H_full exists for the given parameters. If so, loads it. 
    If not, computes the RIR, calculates H_full, and caches it.
    """
    # 1. Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # 2. Create a unique, deterministic hash from the parameters
    # sort_keys=True ensures the dictionary is always ordered the same way
    params_json = json.dumps(params_dict, sort_keys=True)
    param_hash = hashlib.md5(params_json.encode('utf-8')).hexdigest()
    
    filename = os.path.join(cache_dir, f"H_full_{param_hash}.npy")
    
    # 3. Check if cached file exists
    if os.path.exists(filename):
        print(f"Loading cached H_full from {filename}...")
        return np.load(filename)
    
    # 4. If not, compute it from scratch
    print("No cache found. Computing RIR from scratch (this may take a while)...")
    room.compute_rir()
    
    # Extract dimensions
    num_mics = room.mic_array.R.shape[1]
    num_speakers = len(room.sources)
    
    print('Computing H...')
    # (Using your optimized quick_compute_H logic inline here)
    max_rir_len = max(len(rir) for mic_list in room.rir for rir in mic_list)
    rir_array = np.zeros((num_mics, num_speakers, max_rir_len))
    
    for m in range(num_mics):
        for s in range(num_speakers):
            rir_array[m, s, :len(room.rir[m][s])] = room.rir[m][s]

    H_full = np.fft.rfft(rir_array, n=nfft, axis=-1)
    
    # 5. Save the computed H_full to the cache
    print(f"Saving computed H_full to {filename}...")
    np.save(filename, H_full)
    
    return H_full
    


from scipy.signal import resample


def clean_wav_data(data):
    if data.ndim > 1:
        data = data[:, 0]  # Take one channel if stereo
    data = data.astype(np.float32) / 32768.0 
    return data



def resample_signal(data, fs_file, fs):
    if fs_file != fs:
        print(f'Resampling audio from {fs_file} Hz to {fs} Hz...')
        num_samples_resampled = int(len(data) * fs / fs_file)
        audio_time_full = resample(data, num_samples_resampled)
    else:
        audio_time_full = data    
    return audio_time_full

def create_pure_signal(duration, fs, nfft, audio_freq, audio_amp, audio_phase, zero_padd_sec = 0):
    
    # Creating signal as sum of frequencies
    t_axis = np.arange(0, duration, 1/fs)   
    audio_time = np.zeros(len(t_axis))
    for idx, freq in enumerate(audio_freq):
        audio_time += audio_amp[idx] * np.sin(2 * np.pi * freq * t_axis + audio_phase[idx])
    audio_fft = np.fft.rfft(audio_time, n=nfft) 

    # if zero_padd_sec != 0:
    #     audio_time = np.pad(audio_time, (0, round(zero_padd_sec *fs)), 'constant')
    #     t_axis = np.arange(0, duration + zero_padd_sec, 1/fs)  
    #     audio_fft = np.fft.rfft(audio_time)  # This doesn't work to use yet...

    # audio_fft = np.fft.rfft(audio_time, n=int(nfft/2)) 

    return t_axis, audio_time, audio_fft

def window_signal(audio_time, fs, nfft):

    # The window has the same size as the signal
    window = np.hanning(len(audio_time)) 
    audio_time = audio_time * window

    audio_fft = np.fft.rfft(audio_time, n=nfft)

    t_axis_full = np.arange(nfft) / fs
    audio_time_padded = np.pad(audio_time, (0, nfft - len(audio_time)), 'constant')
    
    return t_axis_full, audio_time_padded, audio_fft

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
    
    
def plot_signal_log(t_axis, audio_time, f_axis, audio_fft, fs, nfft, mag_clipp=1e-6):
    # Create a figure with 2 subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # adjust size if needed

    # --- First subplot: Time domain ---
    ax1.plot(t_axis*1_000, audio_time, color='tab:blue')
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Time Domain: Audio Signal")
    ax1.grid(True)

   # --- Second subplot: Frequency domain ---
    magnitude = np.abs(audio_fft) / (nfft / 2)
    magnitude_clipped = np.maximum(magnitude, mag_clipp) 

    # 1. Rita datan först
    ax2.semilogy(f_axis, magnitude_clipped, color='tab:orange')

    # 2. Sätt skalan till log base 2
    ax2.set_xscale('log', base=2)

    # 3. Definiera dina ticks
    octave_ticks = [125, 250, 500, 1000, 2000, 4000, 8000]
    ax2.set_xticks(octave_ticks)

    # 4. Tvinga Matplotlib att skriva ut siffror (ScalarFormatter)
    # Detta måste ligga EFTER set_xscale och set_xticks
    from matplotlib.ticker import ScalarFormatter
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    
    # 5. Sätt gränser och labels
    ax2.set_xlim([62.5, fs/2])
    ax2.set_xlabel("Frequency [Hz] (Log2 Scale)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Frequency Domain: FFT of Audio Signal")
    
    # 6. Grid (viktigt för log-skalor)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)

    # Adjust layout so titles and labels don't overlap
    plt.tight_layout()
    plt.show()


def import_signal(filepath, fs, nfft, start_sec, duration):

    filepath = 'wav_files/why_were_you_away.wav'
    fs_file, wav_data = wav.read(filepath)
    print(f'fs_file: {fs_file}')
    data = clean_wav_data(wav_data) # extract signal and normalize
    audio_time_full = resample_signal(data, fs_file, fs)

    # 2. Select section
    t_axis = np.arange(0, duration, 1/fs)   
    num_samples = int(duration * fs)
    start_index = round(start_sec * fs)
    end_index = start_index + num_samples
    audio_time = audio_time_full[start_index : end_index]

    # 3. Compute the FFT
    audio_fft = np.fft.rfft(audio_time, n=nfft)

    return t_axis, audio_time, audio_fft


def calculate_broadband_contrast(bright_signal, dark_signal):
    """
    Calculates the broadband acoustic contrast in dB between two time-domain signals.
    
    Parameters:
        bright_signal (np.ndarray): The audio signal from the bright zone.
        dark_signal (np.ndarray): The audio signal from the dark zone.
        
    Returns:
        float: The contrast in decibels (dB).
    """
    # 1. Calculate the Mean Square (Energy) of both signals
    # We use a small epsilon (1e-12) to prevent division by zero
    energy_bright = np.mean(np.square(bright_signal))
    energy_dark = np.mean(np.square(dark_signal))
    
    # 2. Calculate the ratio and convert to decibels
    # Formula: 10 * log10(Energy_Bright / Energy_Dark)
    contrast_db = 10 * np.log10((energy_bright + 1e-12) / (energy_dark + 1e-12))
    
    return contrast_db


def calculate_sliding_contrast(bright_signal, dark_signal, fs, window_sec=0.04, overlap=0.5):
    """
    Calculates how the dB contrast evolves over time using a sliding window.
    
    Parameters:
        bright_signal (np.ndarray): The bright zone audio.
        dark_signal (np.ndarray): The dark zone audio.
        fs (int): Sampling frequency.
        window_sec (float): Duration of each analysis window in seconds.
        overlap (float): Fraction of overlap between windows (0 to 1).
        
    Returns:
        t_axis (np.ndarray): Time points for the center of each window.
        contrast_series (np.ndarray): dB contrast values over time.
    """
    hop_size = int(round(fs * window_sec * (1 - overlap)))
    window_size = int(round(fs * window_sec))
    
    contrast_series = []
    t_axis = []
    
    # Iterate through the signal in steps
    for start in range(0, len(bright_signal) - window_size, hop_size):
        end = start + window_size
        
        # Extract chunks
        b_chunk = bright_signal[start:end]
        d_chunk = dark_signal[start:end]
        
        # Calculate local energy
        energy_b = np.mean(np.square(b_chunk))
        energy_d = np.mean(np.square(d_chunk))
        
        # Calculate local contrast in dB
        db = 10 * np.log10((energy_b + 1e-12) / (energy_d + 1e-12))
        
        contrast_series.append(db)
        t_axis.append((start + window_size / 2) / fs)
        
    return np.array(t_axis), np.array(contrast_series)


def compute_anechoic_H(mics_locs, speaker_locs, fs, nfft):
    """
    Analytically computes the Transfer Function matrix H for an anechoic (free-field) 
    environment using the free-space Green's function.
    
    H[m, s, f] = (1 / dist) * exp(-j * k * dist)
    """
    num_mics = mics_locs.shape[1]
    num_speakers = speaker_locs.shape[1]
    num_bins = nfft // 2 + 1
    
    c = pra.constants.get('c')
    f_axis = np.fft.rfftfreq(nfft, d=1/fs)
    
    # 1. Pre-calculate all pairwise distances between mics and speakers
    # Resulting shape: (num_mics, num_speakers)
    diffs = mics_locs[:, :, np.newaxis] - speaker_locs[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=0)
    
    # 2. Initialize H_full
    H_full = np.zeros((num_mics, num_speakers, num_bins), dtype=complex)
    
    # 3. Vectorized computation over frequencies
    for idx, f in enumerate(f_axis):
        if f == 0:
            # Handle DC component: pure 1/r attenuation, no phase shift
            H_full[:, :, idx] = 1.0 / dists
        else:
            k = 2 * np.pi * f / c
            # Free-space Green's function: (1/r) * e^(-j * k * r)
            H_full[:, :, idx] = (1.0 / dists) * np.exp(-1j * k * dists)
            
    return H_full


def plot_chunked_contrast(bright_signal, dark_signal, fs, chunk_sec=0.04):
    """
    Calculates and plots the acoustic contrast and raw energy for exact, 
    non-overlapping chunks of the audio signals.
    """
    # 1. Calculate chunk size in samples
    chunk_size = int(round(fs * chunk_sec))
    num_chunks = len(bright_signal) // chunk_size
    
    contrast_vals = []
    energy_vals = []
    t_axis = []
    
    # 2. Iterate through strictly non-overlapping chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        
        b_chunk = bright_signal[start:end]
        d_chunk = dark_signal[start:end]
        
        # Calculate Mean Square (Energy)
        energy_b = np.mean(np.square(b_chunk))
        energy_d = np.mean(np.square(d_chunk))
        
        # Calculate Contrast in dB
        db = 10 * np.log10((energy_b + 1e-12) / (energy_d + 1e-12))
        
        contrast_vals.append(db)
        # Convert bright energy to dB for easier visualization against contrast
        energy_vals.append(10 * np.log10(energy_b + 1e-12)) 
        
        # Time axis represents the center of the chunk
        t_axis.append((start + chunk_size / 2) / fs)

    # 3. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot the Contrast (Left Y-Axis)
    color1 = 'black'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Contrast [dB]', color=color1)
    ax1.plot(t_axis, contrast_vals, color=color1, marker='o', linestyle='-', linewidth=2, label='Acoustic Contrast')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Add an average line for contrast
    avg_contrast = np.mean(contrast_vals)
    ax1.axhline(y=avg_contrast, color=color1, linestyle='--', alpha=0.5, label=f'Avg Contrast: {avg_contrast:.1f} dB')

    # Plot the Bright Zone Energy (Right Y-Axis)
    ax2 = ax1.twinx()  
    color2 = 'black'
    ax2.set_ylabel('Bright Zone Energy [dB]', color=color2)
    ax2.plot(t_axis, energy_vals, color=color2, marker='x', linestyle=':', linewidth=1.5, label='Bright Energy')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and grids
    plt.title(f"Acoustic Contrast Analysis ({chunk_sec*1000:.0f} ms chunks)")
    ax1.grid(True, alpha=0.3)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()

    return np.array(t_axis), np.array(contrast_vals), np.array(energy_vals)