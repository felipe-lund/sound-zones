from matplotlib import pyplot as plt
import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wav
import sounddevice as sd
import time



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
        lambda_smooth = 0.9  # Spatial smoothness penalty
        
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

def plot_pressure_map(pressure_map, X, Y, all_speakers, bright_center, dark_center, radius, title):
    """
    Renders the acoustic heatmap with zone and speaker overlays.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot heatmap
    cont = plt.contourf(X, Y, pressure_map, levels=50, cmap='inferno')
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

def save_as_wav(filename, signal, fs):
    """Helper to convert float signal to 16-bit PCM and save."""
    scaled = np.int16(signal * 32767)
    wav.write(filename, fs, scaled)



def plot_audio_analysis(bright_wav_path, dark_wav_path, freq_range=(0, 1500), time_zoom=(0.1, 0.15)):
    """
    Plots time-domain waveforms and frequency-domain spectra for two wav files.
    """
    # 1. Load the signals
    fs_b, b_data = wav.read(bright_wav_path)
    fs_d, d_data = wav.read(dark_wav_path)

    # Standardize time and frequency axes
    t = np.arange(len(b_data)) / fs_b
    freq_axis = np.fft.rfftfreq(len(b_data), 1/fs_b)

    # 2. Perform FFT (Magnitude Spectrum)
    b_fft_mag = np.abs(np.fft.rfft(b_data))
    d_fft_mag = np.abs(np.fft.rfft(d_data))

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Time Domain Plot
    ax1.plot(t, b_data, label='Bright Zone', color='red', alpha=0.7)
    ax1.plot(t, d_data, label='Dark Zone', color='blue', alpha=0.7)
    ax1.set_title("Time Domain: Bright vs Dark Zone Waveforms")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(time_zoom) 
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Frequency Domain Plot
    ax2.plot(freq_axis, b_fft_mag, label='Bright Zone', color='red')
    ax2.plot(freq_axis, d_fft_mag, label='Dark Zone', color='blue')
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