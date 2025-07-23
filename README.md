# Detection for objective tinnitus
This is a device used to detect objective tinnitus in the ear canal using a inear microphone.
## Hardwares
Core: [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)

Touch Screen: [Raspberry Pi Official 7" LCD Touch Display 2 (SC1635)](https://www.raspberrypi.com/products/touch-display-2/)

Sound Card: [Pi Sound](https://blokas.io/pisound/)

Microphone: [RØDE smartLav+](https://rode.com/en/products/smartlav-plus)

Adapters: [SC3, 3.5m TRRS to TRS Adaptor](https://rode.com/en/products/sc3); [AV Adapter, 3.5mm to 1/4](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChsSEwiAj4-F19KOAxWxpGYCHeLsKiQYACICCAEQGhoCc20&co=1&ase=2&gclid=Cj0KCQjwkILEBhDeARIsAL--pjwNm1FiJPDzKa9fMI-j_-p9ddxbYZ-5g3uSl7_SHG9M49bfU8sTmjEaAmcCEALw_wcB&ohost=www.google.comcid=CAESVeD2XnBelKPjYt5ecPFCVxeILIl3LmNixfIlJWfDKJTCmltC4eD1GcUrxFUE8_TawnRX1jr0jJsh3dbwZ156sKJR0OTPHmSeGGqYniwUDoaGZ18CNiA&category=acrcp_v1_45&sig=AOD64_0r7GQa_UCVvws-wRXK3miZs0S86A&ctype=5&q=&nis=4&ved=2ahUKEwjpiYWF19KOAxX6SGwGHRVZG5YQ9aACKAB6BAgJEBo&adurl=)

Heat sinks: [Raspberry Pi 5 Heatsink](https://www.raspberrypi.com/products/active-cooler/)

Charger: [Raspberry Pi 5 charger](https://www.raspberrypi.com/products/27w-power-supply/)

### Extra Parts (Optional)
1. 30mm M2.5 Standoffs
2. 10mm M2.5 Standoffs
3. 2x40 Stackable female pin header


## Setup
Front Screen display
![Front Screen display](Images\Front_screen_display.jpeg)

Bottom View
![Bottom View](Images\Bottom_View.jpeg)

Top View
![Top View](Images\Top_View.jpeg)

Back View
![Back View](Images\Back_View.jpeg)

### Notes on the hardware setup
i had used 3 30mm + 10mm M2.5 standoffs to elvate the Pi Sound so that the DSI cable is not squished im between the heat sink and Pi Sound. The 2x40 stackable pin header was used to extend the pin's length so that the Raspberry Pi 5 GPIO pins can reach the Pi Sound.

## Objective Tinnitus Recorder

This application is a PyQt5-based graphical user interface (GUI) for recording audio, visualizing it as a waveform and a Mel spectrogram, and performing real-time analysis to detect and classify sounds as pulsatile or non-pulsatile. It is designed to find a compatible audio device automatically, record for a fixed duration, and then present the analysis results.

## Requirements

To run this application, you will need to have Python installed, along with the following libraries:

- **PyQt5:** For the graphical user interface.
- **PyAudio:** To capture audio from the microphone.
- **NumPy:** For numerical operations on audio data.
- **SciPy:** Used for signal processing, specifically for finding peaks.
- **librosa:** For advanced audio analysis and Mel spectrogram generation.
- **soundfile:** For saving the recorded audio to a `.wav` file.
- **Matplotlib:** For plotting the waveform and spectrogram.

## Configurations

You can customize the application's behavior by modifying the constants defined at the top of the audio_with_spectogram.py file.
***
### Audio Recording Settings
These settings control the input device's configuration and the recording process.

- **TARGET_SAMPLE_RATE**: The sample rate in Hz for recording. The application will search for a device that supports this rate. Default: 192000.

- **TARGET_CHANNELS**: The number of audio channels. Default: 1 (Mono).

- **TARGET_FORMAT**: The data format for the audio stream. pyaudio.paInt16 represents 16-bit audio.

- **CHUNK_SIZE**: The number of audio frames per buffer. A smaller size may reduce latency but increase CPU usage. Default: 4096.

- **DEFAULT_OUTPUT_DIR**: The name of the folder where recordings will be saved by default. Default: "OBJTIN Recording".

- **FIXED_RECORDING_DURATION_SECONDS**: The duration of each recording in seconds. Default: 30.

***
### Spectrogram Parameters
These settings control the appearance and detail of the Log-Mel spectrogram.

- **SPEC_N_FFT**: The length of the Fast Fourier Transform (FFT) window. A larger value increases frequency resolution but decreases time resolution. Default: 8192.

- **SPEC_HOP_LENGTH**: The number of audio samples between adjacent FFT windows. A smaller value increases the overlap and the time resolution of the spectrogram. Default: 2048.

- **SPEC_N_MELS**: The number of Mel bands to generate, which determines the vertical resolution of the spectrogram. Default: 128.

- **SPEC_WINDOW**: The window function to apply before the FFT. Options include 'hamming', 'hann', 'blackman', etc. Default: 'hamming'.

***
### Sound Detection and Analysis Parameters
These values tune the algorithm that classifies the audio.

- **WINDOW_DURATION_MS**: The duration (in milliseconds) of each chunk used to calculate the RMS of the signal. This affects the time resolution of the peak detection. Default: 50.

- **RELATIVE_THRESHOLD**: A ratio (std_dev / mean_rms) used to determine if a significant sound is present. A higher value makes the detection less sensitive. Default: 0.5.

- **DISTANCE**: The minimum required distance (in number of RMS samples) between detected peaks. This helps prevent detecting multiple peaks for a single sound event. Default: 7.

- **PULSATILE_BPM_MIN**: The minimum beats-per-minute (BPM) to be classified as "Pulsatile". Default: 40.

- **PULSATILE_BPM_MAX**: The maximum beats-per-minute (BPM) to be classified as "Pulsatile". Default: 180.