

# Sound Zones

## Introduction

This project was done by Sara Elmgart and Felipe Abedrapo as a part of the course Stationary and Non-stationary Spectral Analysis at Lund Technical University (LTH) in the spring of 2026. Supervisors were Andreas Jakobsson and David Sundström. 

The main purpose of the project was to investigate and simulate sound zones in a room with given room impulse responses. The audio used was the famous "why were you away a year, Roy?" and the difference between the simulated bright and dark zone can be heard in the wav-file `pm_combined_zones.wav`.

## Running the Code

The main programs to run are saved in the run_speech-files and the functions created are saved in the myutils-files. For example run_speech_final.py can be run as a whole, or cell by cell in the inveractive mode of Python. We found it easiest is to run it in the interactive mode since this makes it possible to scroll between the different figures and would advise the user to do so as well.

## Main Variables to Tweak

 The sampling frequency `fs`, the number of bounces `nbr_bounces` and `air_absorption` are the first variables appearing. To create an anechoic room, `nbr_bounces` was set to zero. Both the mic spacing and the number of speakers per wall can be changed. Changing the mic spacing means the program might later have to recalculate the room impulse responses and thereby the matrix `H`. Since this is an heavy and time consuming calculation, the room impulse responses of some standard rooms (for example `nbr_bounces = [0, 3]` and `mic_spacing = [0.1, 0.5]`) we used in our project are saved in the cached_rir folder. To use an imported audio, `import_audio` needs to be set to `True` and the `filepath` needs to be specified. The `duration` of each chunk of the file to process can be set. Note that `fs * duration < NFFT` needs to be true. If no import of an audio is desired, one can set `import_audio` to `False` and create a signal of sums of pure sine waves by defining `audio_freq`, `audio_amp` and `audio_phase`. The radius of the bright and the dark zone can easily be changed by the variable `radius`.

