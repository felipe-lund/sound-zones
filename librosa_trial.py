
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# 1. Beräkna CQT
cqt_result = librosa.cqt(audio_time, sr=fs, 
                         fmin=librosa.note_to_hz('C2'), 
                         n_bins=80, 
                         bins_per_octave=12)

# 2. Ta magnituden (medelvärde över tid)
magnitude_cqt_1d = np.abs(cqt_result).mean(axis=1)

# 3. Hämta frekvensaxeln
f_axis_cqt = librosa.cqt_frequencies(n_bins=80, 
                                     fmin=librosa.note_to_hz('C2'), 
                                     bins_per_octave=12)

# 4. Plotta
plt.figure(figsize=(10, 4))
plt.plot(f_axis_cqt, magnitude_cqt_1d, color='tab:orange', linewidth=2)

# --- Inställningar för skalan ---
plt.xscale('log', base=2)

# Definiera dina ticks (siffrorna på axeln)
octave_ticks = [125, 250, 500, 1000, 2000, 4000, 8000]
plt.xticks(octave_ticks)

# Tvinga fram vanliga siffror istället för potenser
plt.gca().xaxis.set_major_formatter(ScalarFormatter())

# Labels och snygghet
plt.xlim([62.5, fs/2])
plt.xlabel("Frequency [Hz] (Log2 Scale)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain: CQT Analysis")
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.show()