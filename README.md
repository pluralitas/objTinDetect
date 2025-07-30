# Detection for objective tinnitus
This is a device used to detect objective tinnitus in the ear canal using a inear microphone.
## Hardwares
- Core: [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)
- Touch Screen: [Raspberry Pi Official 7" LCD Touch Display 2 (SC1635)](https://www.raspberrypi.com/products/touch-display-2/)
- Sound Card: [Pi Sound](https://blokas.io/pisound/)
- Microphone: [RØDE smartLav+](https://rode.com/en/products/smartlav-plus)
- Adapters: [SC3, 3.5m TRRS to TRS Adaptor](https://rode.com/en/products/sc3); [AV Adapter, 3.5mm to 1/4](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChsSEwiAj4-F19KOAxWxpGYCHeLsKiQYACICCAEQGhoCc20&co=1&ase=2&gclid=Cj0KCQjwkILEBhDeARIsAL--pjwNm1FiJPDzKa9fMI-j_-p9ddxbYZ-5g3uSl7_SHG9M49bfU8sTmjEaAmcCEALw_wcB&ohost=www.google.comcid=CAESVeD2XnBelKPjYt5ecPFCVxeILIl3LmNixfIlJWfDKJTCmltC4eD1GcUrxFUE8_TawnRX1jr0jJsh3dbwZ156sKJR0OTPHmSeGGqYniwUDoaGZ18CNiA&category=acrcp_v1_45&sig=AOD64_0r7GQa_UCVvws-wRXK3miZs0S86A&ctype=5&q=&nis=4&ved=2ahUKEwjpiYWF19KOAxX6SGwGHRVZG5YQ9aACKAB6BAgJEBo&adurl=)
- Heat sinks: [Raspberry Pi 5 Heatsink](https://www.raspberrypi.com/products/active-cooler/)
- Charger: [Raspberry Pi 5 charger](https://www.raspberrypi.com/products/27w-power-supply/)

### Extra Parts (Optional)
1. 30mm M2.5 Standoffs
2. 10mm M2.5 Standoffs
3. 2x40 Stackable female pin header


## Setup
Front Screen display
![Front View](Images\Front_screen_display.jpeg)

Bottom View
![Bottom View](Images\Bottom_View.jpeg)

Top View
![Top View](Images\Top_View.jpeg)

Back View
![Back View](Images\Back_View.jpeg)

### Notes on the hardware setup
i had used 3 30mm + 10mm M2.5 standoffs to elvate the Pi Sound so that the DSI cable is not squished im between the heat sink and Pi Sound. The 2x40 stackable pin header was used to extend the pin's length so that the Raspberry Pi 5 GPIO pins can reach the Pi Sound.

## Objective Tinnitus Recorder python file (`audio_with_spectrogram.py`)

This application is a PyQt5-based graphical user interface (GUI) for recording audio, visualizing it as a waveform and a Mel spectrogram, and performing real-time analysis to detect and classify sounds as pulsatile or non-pulsatile. It is designed to find a compatible audio device automatically, record for a fixed duration, and then present the analysis results.

### Requirements

To run this application, you will need to have Python 3.9.21 installed, along with the following libraries:

- **PyQt5:** For the graphical user interface.
- **PyAudio:** To capture audio from the microphone.
- **NumPy:** For numerical operations on audio data.
- **SciPy:** Used for signal processing, specifically for finding peaks.
- **librosa:** For advanced audio analysis and Mel spectrogram generation.
- **soundfile:** For saving the recorded audio to a `.wav` file.
- **Matplotlib:** For plotting the waveform and spectrogram.

### Configurations

You can customize the application's behavior by modifying the constants defined at the top of the audio_with_spectogram.py file.
***
#### Audio Recording Settings
These settings control the input device's configuration and the recording process.

- **TARGET_SAMPLE_RATE**: The sample rate in Hz for recording. The application will search for a device that supports this rate.

- **TARGET_CHANNELS**: The number of audio channels.

- **TARGET_FORMAT**: The data format for the audio stream. pyaudio.paInt16 represents 16-bit audio.

- **CHUNK_SIZE**: The number of audio frames per buffer. A smaller size may reduce latency but increase CPU usage.

- **DEFAULT_OUTPUT_DIR**: The name of the folder where recordings will be saved by default.

- **FIXED_RECORDING_DURATION_SECONDS**: The duration of each recording in seconds.

***
#### Spectrogram Parameters
These settings control the appearance and detail of the Log-Mel spectrogram.

- **SPEC_N_FFT**: The length of the Fast Fourier Transform (FFT) window. A larger value increases frequency resolution but decreases time resolution.

- **SPEC_HOP_LENGTH**: The number of audio samples between adjacent FFT windows. A smaller value increases the overlap and the time resolution of the spectrogram.

- **SPEC_N_MELS**: The number of Mel bands to generate, which determines the vertical resolution of the spectrogram.

- **SPEC_WINDOW**: The window function to apply before the FFT. Options include 'hamming', 'hann', 'blackman', etc.

***
#### Sound Detection and Analysis Parameters
These values tune the algorithm that classifies the audio.

- **WINDOW_DURATION_MS**: The duration (in milliseconds) of each chunk used to calculate the RMS of the signal. This affects the time resolution of the peak detection.

- **RELATIVE_THRESHOLD**: A ratio (std_dev / mean_rms) used to determine if a significant sound is present. A higher value makes the detection less sensitive.

- **DISTANCE**: The minimum required distance (in number of RMS samples) between detected peaks. This helps prevent detecting multiple peaks for a single sound event.

- **PULSATILE_BPM_MIN**: The minimum beats-per-minute (BPM) to be classified as "Pulsatile".

- **PULSATILE_BPM_MAX**: The maximum beats-per-minute (BPM) to be classified as "Pulsatile".

***
### Class description

#### `AudioController`

The AudioController class serves as the backend for all low-level audio operations. It abstracts the complexities of interacting with the pyaudio library and handles audio device management and data processing.

- Device Management
    - Initialize pyaudio instance.
    - Searches the system for a compatible audio input device that matches the configuration needed.
    - Returns a True if the suitable input device is found.
    - Stores the parameters of the input device.

- Audio Processing
    - Saves raw recorded audio frames into a `.wav`  file.
    - Converts raw byte frames into a Numpy array (`y`).
    - Generates the Log-Mel Spectrogram (`S_mel_db`).

- Stopping
    - Uses a `close()` to terminate pyaudio when closing application.
***
#### `AudioWorker`

This class is a QThread designed to perform the audio recording in the background. By offloading the recording task to a separate thread, it ensures the main application window remains responsive and does not freeze during the recording process.

- Background Recording
    - The `run` method creates a new pyaudio stream to continuously read audio data in chunks `CHUNK_SIZE` until the recording is stop automatically (`FIXED_RECORDING_DURATION_SECONDS`) or manually (Stop Button).

- UI Update
    - Uses `pyqtSignal` to update on the `MainWindow`.
    - `progress_updated`: Emits the elapsed and remaining time to update the progress bar.
    - `status_updated`: Emits status messages to be displayed in the UI.
    - `recording_finished`: Emits the list of recorded audio frames back to the main window upon successful completion.
    - `recording_error`: Emits an error message if an exception occurs during the recording process.

- Stopping
    - Uses a `stop()` to allos the main thread to interrupt the recording loop early when the user uses the Stop Button.
***
#### `ClickableLabel`
A simple class that extends `Qlabel` to add a click functionality.
***
#### `SoundAnalyser`
This class is the core algorithm for analyzing the recorded audio to determine if it contains sounds and then classify it into pulsatile or non-pulsatile.

- Audio Analysis
    - Takes in the raw audio waveform (`y`) and sample rate (`sr`)
    - Normalises the audio signal to [-1,1]

- Sound Detection
    - Computes the rms of the graph to get the power of the signal
    - Determines if a significant signals is present by comparing the standard deviation of the RMS values to their mean (`relative_ratio` > `RELATIVE_THRESHOLD`).
    - Returns a true value if there is sound detected or else a false value and "No Sound" text to be displayed

- Pulsatile Classification
    - If sound is detected, it crops the waveform to remove the noise floor. Afterwards it performs peak detection on the cropped rms waveform(`cropped_rms`) to find the distinct peaks.
    - Calculates the intervals between these peaks to determine an average Beats per Minute (BPM)
    - Classfies the sound as "Pulsatile" or "Non-Pulsatile" based on whether the calculated BPM falls within the `PULSATILE_BPM_MIN` and `PULSATILE_BPM_MAX thresholds`.
***
#### `MainWindow`
The MainWindow class controls the UI of the application. It inherits from QMainWindow and is responsible for creating the user interface, managing the application's state, and coordinating the interactions between the user and the backend classes (`AudioController`, `AudioWorker`, `SoundAnalyzer`).

- UI Management
    - Builds and Organise all the UI elements (Button, labels, progress bar and layouts)
    - Uses `QstackedWidget` to manange the 3 primary application pages (idle, Recording and Analysis)
    - Embeds Matplotlib `FigureCanvas` widgets to display the waveform and spectrogram plots

- State and Logic Controls
    - Initialise and hold instances of `Audiocontroller` and `SoundAnalyser` 
    - Manages the application's flow: Checking device status --> start recording --> handling record completion --> resetting for a new recording or closing the application.

- Event Handlling
    - `handle_start_stop()`: Starts or stops the `AudioWorker` thread.
    - `handle_save_as()`: Opens a file dialog and saves the recorded audio by calling `audio_controller.save_audio_to_file()`.
    - `handle_finish_reset()`: Resets the UI and application state back to idle.
    - `handle_recording_completion()`: A slot that receives the recorded data from `AudioWorker`, triggers the analysis and plotting, and updates the UI to the analysis page.

- UI Updates
    - Connects to signals from the AudioWorker to update the progress bar and status messages in real-time.
    - Calls `update_analysis_plots()` to render the waveform and spectrogram visualizations after a recording is complete.
    - Manages the visibility and content of the result label ("Pulsatile", "Non-Pulsatile", "No Sound", "Error").

- Closing event
    - Handles the application's `closeEvent` to ensure the `AudioWorker` thread is stopped and the `AudioController`'s resources are released properly upon exit.
***
## Verification of the heart sound audio
### ECG and Audio Recording Comparison (`ECG_vs_Audio_Recording_Test.ipynb`)
`ECG_vs_Audio_Recording_Test.ipynb` provides a Python script designed to analyze and compare Electrocardiogram (ECG) signals with corresponding heart sound audio recordings. The tool is built to run in a Jupyter Notebook and can process multiple pairs of files in a single execution.

For each pair of ECG and audio files, the script performs the following actions:
- Loads the signal data from .csv (for ECG) and .wav (for audio) files.
- Crops the signals to a user-defined time window.
- Resamples both signals to a consistent target sample rate for accurate comparison.
- Detects R-peaks in the ECG signal and sound peaks in the audio signal using customizable parameters.
- Calculates the average Beats Per Minute (BPM) from both signals.
- Performs a paired t-test on the beat-to-beat intervals to determine if there is a statistically significant difference between the two measurement methods.
- Generates and displays a set of three plots for each file pair: the ECG signal with detected peaks, the audio signal with detected peaks, and a boxplot comparing the distribution of beat-to-beat intervals.
***
### Requirements
To run this application, you will need to have Python 3.9.21 installed, along with the following libraries:

- **numpy**
- **pandas**
- **matplotlib**
- **scipy**
***
### Code description
1. Cell 1: Imports
    - This cell imports all the necessary Python libraries. No changes are needed here.
2. Cell 2: Configuration
    - This is the only cell you need to edit.
    - Define all the files you want to process in the file_processing_jobs list.
    - Each "job" is a Python dictionary containing all the settings for one pair of ECG and audio files. You can add as many jobs as you need by copying the structure.
    - Key settings include file paths, signal sample rates, cropping times, and peak detection parameters (peak_height, peak_spacing_ms, peak_width_ms).
3. Cell 3: Processing Loop
    - This cell contains the main logic. It loops through each job defined in Cell 2, performs the analysis, and prints the results and plots.
***
### Execution
1. Open the .ipynb file in Jupyter Notebook or JupyterLab.
2. Populate Cell 2 with the details of all the files you wish to analyze.
3. Run the cells in order from top to bottom.
***
### Output
For each job defined in the configuration cell, the script will output:
- Console Logs: Real-time progress updates in the notebook, including file loading status, processing steps, detected peak counts, BPM values, and the results of the t-test.
- Visual Plots: A multi-panel plot will be displayed directly below the processing cell for each job, allowing for easy visual inspection of the results.
***
### Results Summary
The numerical outputs from this script (such as mean BPM, t-statistic, and p-values) are printed to the console during execution.

A comprehensive summary of the key findings and final results for all processed recordings has been compiled and is available in the accompanying Excel file:
[`Comparison analysis of ECG and Audio waveforms.xlsx`]

## Classification ALgorithm test
`Audio_Classification_Test.ipynb` provides a Python script designed to analyze audio recordings (.wav files) to perform two key tasks:
1. Sound vs. Noise Classification: It classifies an audio segment as either a meaningful "Sound" (e.g., pulsatile tinnitus) or ambient "Noise" based on the signal's amplitude variance.
2. BPM Detection: For signals classified as "Sound," it identifies rhythmic peaks and calculates the corresponding Beats Per Minute (BPM).
***

### Analysis methodology
The script processes each audio file using the following steps:

1. Loading and Pre-processing: The audio file is loaded, converted to mono, and its amplitude is normalized to a standard range of [-1, 1].
2. Cropping: A specific segment of the audio file is selected for analysis based on user-defined start and end times.
3. Chunking and RMS Calculation: The cropped audio is divided into small, consecutive, non-overlapping chunks (e.g., 50 ms). The Root Mean Square (RMS) value, which represents the effective power, is calculated for each chunk.
4. Sound vs. Noise Classification: The script uses the standard deviation of the RMS values across all chunks. A signal with high variance (a large standard deviation) is typically pulsatile or contains distinct events and is classified as "Sound". A signal with low variance is more uniform and is classified as "Noise".
5. BPM Detection:
    - A dynamic threshold is calculated (Mean RMS + 1 Standard Deviation).
    - RMS values below this threshold are set to zero, isolating the significant peaks.
    - The script identifies the peaks in this thresholded signal.
    - The average time interval between these peaks is calculated to estimate the final BPM.
***
### Requirements
To run this application, you will need to have Python 3.9.21 installed, along with the following libraries:

- **numpy**
- **matplotlib**
- **scipy**
***
### Code Description
1. Cell 1: Imports
    - This cell imports the necessary libraries. No changes are needed here.
2. Cell 2: Configuration
    - This is the only cell you need to edit.
    - All files to be analyzed are defined in the `audio_analysis_jobs` list.
    - To add a new file, simply copy an existing job (the block from { to }) and modify its parameters:
        - audio_path: Set the path to your new .wav file.
        - crop_start_s / crop_end_s: Define the time window you want to analyze.
        - bpm_peak_distance: Adjust this to match the expected heart rate. A larger number works better for slower BPMs.
3. Cell 3: Processing Loop
    - Run this cell after setting up your configuration. It will automatically loop through each job, perform the full analysis, and display the outputs.

***
### Output
For each audio file defined in the configuration, the script will produce:

- A Multi-Panel Plot:
    - Top Panel: Shows the average absolute waveform, giving a general sense of the signal's structure.
    - Middle Panel: Displays the calculated RMS values for each chunk, used for the "Sound vs. Noise" classification.
    - Bottom Panel: Shows the thresholded RMS signal with the detected peaks clearly marked, along with the final estimated BPM.
- A Console Summary: Printed below the plot, this text report provides the final classification results and the calculated BPM value for easy reference.