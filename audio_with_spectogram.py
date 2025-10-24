import sys
import os
import time
import numpy as np
from scipy import signal 
from scipy.signal import find_peaks
from scipy.io import wavfile
import matplotlib
matplotlib.use('Qt5Agg') # Use Qt5Agg for Matplotlib backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure # Import Figure for embedding
import pyaudio
import wave
import librosa # For Mel spectrogram
import librosa.display # For displaying Mel spectrogram axis correctly

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QStackedWidget, QFrame,
    QFileDialog, QMessageBox, QSpinBox, QSizePolicy 
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal

# --- Configuration ---
TARGET_SAMPLE_RATE = 48000 #48000
TARGET_CHANNELS = 1 
TARGET_FORMAT = pyaudio.paInt24 #pyaudio.paInt16 , pyaudio.paInt24
CHUNK_SIZE = 2**15 # Size of each audio chunk to read from the stream
DEFAULT_OUTPUT_DIR = "OBJTIN Recording" # Folder name for saving recordings
FIXED_RECORDING_DURATION_SECONDS = 30 # Duration of each recording in seconds
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

#select format based on TARGET_FORMAT
PCM_FORMAT = {pyaudio.paInt24: "PCM_24", pyaudio.paInt16: "PCM_16", pyaudio.paInt32: "PCM_32"}[TARGET_FORMAT]
WIDTH_SAMPLE = {pyaudio.paInt24: 3, pyaudio.paInt16: 2, pyaudio.paInt32: 4}[TARGET_FORMAT] #24bits = 3bytes

# --- Spectrogram Parameters ---
SPEC_N_FFT = 4096 #8192         
SPEC_HOP_LENGTH = 1024 #2048    
SPEC_N_MELS = 128         
SPEC_WINDOW = 'hamming'   

# --- Sound Detection Parameters ---
WINDOW_DURATION_MS = 50 # Duration of each window in milliseconds
RELATIVE_THRESHOLD = 0.5 # Relative RMS threshold for sound detection
DISTANCE = 7  # Minimum d  istance between peaks in samples (35ms apart) where 1 sample = 0.05sec after cropping RMS waveform
# Pulsatile BPM detection thresholds
PULSATILE_BPM_MIN = 40
PULSATILE_BPM_MAX = 180

# --- Backend Logic Class (AudioController) ---
class AudioController:
    """
    Handles audio input device initialization, recording, and analysis.
    """
    def __init__(self):
        """
        Initializes the audio controller and attempts to find a suitable audio input device.
        """
        self.device_params = None
        self.pyaudio_instance = None
        self.initialize_pyaudio()
        if self.pyaudio_instance:
            self._find_and_verify_workable_device() # Changed from Pisound specific

# --- Device Initialization and Configuration ---
    def initialize_pyaudio(self):
        """
        Initializes the PyAudio instance for audio input/output.
        """
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
        except Exception as e:
            print(f"Failed to initialize PyAudio: {e}")
            self.pyaudio_instance = None

    def _find_and_verify_workable_device(self):
        """
        Searches for an audio input device that meets the target configuration.
        If no suitable device is found, sets `self.device_params` to None.
        """
        if not self.pyaudio_instance:
            print("PyAudio not initialized. Cannot find devices.")
            self.device_params = None
            return

        print(f"Searching for an input device supporting {TARGET_SAMPLE_RATE}Hz...")
        
        # Check the default input device first
        try:
            default_info = self.pyaudio_instance.get_default_input_device_info()
            device_index = default_info['index']
            device_name = default_info['name']
            print(f"Checking default input device: {device_name} (Index: {device_index})")
            if default_info.get('maxInputChannels', 0) >= TARGET_CHANNELS and self.pyaudio_instance.is_format_supported(TARGET_SAMPLE_RATE, input_device=device_index,input_channels=TARGET_CHANNELS, input_format=TARGET_FORMAT):
                self.device_params = {
                    'index': device_index, 
                    'name': device_name, 
                    'rate': TARGET_SAMPLE_RATE,
                    'channels': TARGET_CHANNELS, 
                    'format': TARGET_FORMAT,
                    'max_input_channels': default_info.get('maxInputChannels')
                }
                print(f"Default device '{device_name}' supports the target configuration.")
                return
            else:
                print(f"Default device '{device_name}' does not support the required configuration.")
        except Exception as e:
            print(f"Could not get or check default input device: {e}")

        # If default device fails, iterate through all devices
        print("Checking all available devices...")
        num_devices = self.pyaudio_instance.get_device_count()
        for i in range(num_devices):
            try:
                info = self.pyaudio_instance.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) >= TARGET_CHANNELS:
                    if self.pyaudio_instance.is_format_supported(
                            TARGET_SAMPLE_RATE, input_device=i,
                            input_channels=TARGET_CHANNELS, input_format=TARGET_FORMAT):
                        self.device_params = {
                            'index': i, 'name': info['name'], 'rate': TARGET_SAMPLE_RATE,
                            'channels': TARGET_CHANNELS, 'format': TARGET_FORMAT,
                            'max_input_channels': info.get('maxInputChannels')
                        }
                        print(f"Found suitable device: '{info['name']}' (Index {i}) supports the target configuration.")
                        return
            except Exception as e:
                continue # Skip any devices that fail the check
        
        print(f"Error: No suitable audio input device found supporting {TARGET_SAMPLE_RATE}Hz.")
        self.device_params = None

# --- Audio Handling ---
    def is_ready(self):
        """
        Returns True if a suitable audio device is found and initialized.
        """
        return self.device_params is not None

    def save_audio_to_file(self, frames, filepath): 
        """
        Saves the recorded audio frames to a file in WAV format.
        """
        if not frames or not self.device_params: return False
        print(f"\nSaving audio to: {filepath}")
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1) #mono
                wf.setsampwidth(WIDTH_SAMPLE)
                wf.setframerate(self.device_params['rate'])
                wf.writeframes(b''.join(frames))

            print("Audio saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False

# --- Audio Analysis ---
    def compute_audio_analysis_data(self, frames):
        """
        Computes the Log-Mel spectrogram and other audio analysis data from the recorded frames.
        """
        if not self.device_params or not frames:
            return None, None, None
        
        try:
            print("\nComputing analysis data from recorded frames...")

            if WIDTH_SAMPLE == 4 :
                y = np.frombuffer(b''.join(frames), dtype=np.int32).astype(np.float32) #32bit
            elif WIDTH_SAMPLE == 3:
                #Join list of bytes
                buffer = b''.join(frames)
                nframes = len(buffer) // WIDTH_SAMPLE

                #truncate data that isnt in multiple of frame_size
                buffer = buffer[:nframes*WIDTH_SAMPLE]

                #convert each byte to an 8-bit integer and reshape for flattening by 3bytes
                u8 = np.frombuffer(buffer,dtype=np.uint8).reshape(-1,3)

                #flattening to int32
                int32 = (u8[..., 0].astype(np.int32) | (u8[..., 1].astype(np.int32) << 8) | (u8[..., 2].astype(np.int32) << 16))
                int32 -= (int32 & 0x800000) << 1 

                # Extend signed bit from original to 32-bit signed:
                # If value >= 2^23 (3bytes), subtract 2^24 to get negative range
                y = int32.astype(np.float32) / (1 << 23)

            else:
                y = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) #16bit

            sr = self.device_params['rate']
            
            S_mel = [0]
            S_mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=SPEC_N_FFT, hop_length=SPEC_HOP_LENGTH,
                n_mels=SPEC_N_MELS, window=SPEC_WINDOW
            )
            S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
            print("Analysis data computed successfully.")
            return y, sr, S_mel_db
        except Exception as e:
            print(f"Error computing analysis data: {e}")
            return None, None, None


    def close(self):
        """
        Terminates the PyAudio instance and releases resources.
        """
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            print("PyAudio instance terminated.")


# --- PyQt5 Worker Thread for Recording ---
class AudioWorker(QThread):
    """
    Worker thread responsible for recording audio in the background
    and updating the UI with progress, status, and the recorded frames.
    """
    progress_updated = pyqtSignal(int, int)  # Signal for progress bar update (elapsed time, remaining time)
    status_updated = pyqtSignal(str)  # Signal for updating the status label
    recording_finished = pyqtSignal(list)  # Signal when recording is finished (with audio frames)
    recording_error = pyqtSignal(str)  # Signal when an error occurs during recording

    def __init__(self, device_params, duration):
        """
        Initializes the audio worker with device parameters and recording duration.
        """
        super().__init__()
        self.device_params = device_params
        self.total_duration = duration 
        self._is_running = True
        self.start_time = 0

    def run(self):
        """
        Starts the recording process and updates the UI during the recording.
        """
        p_record = pyaudio.PyAudio() # Create a new PyAudio instance for recording
        stream = None
        frames = [] # List to store the recorded audio frames
        self.status_updated.emit(f"Opening stream on {self.device_params['name']}...")
        try:
            stream = p_record.open(format=self.device_params['format'],
                                   channels=self.device_params['channels'],
                                   rate=self.device_params['rate'],
                                   input=True,
                                   frames_per_buffer=CHUNK_SIZE,
                                   input_device_index=self.device_params['index'])
            self.status_updated.emit(f"Recording for {self.total_duration}s...")
            self.start_time = time.time() # Record the start time
            # Loop to record audio while the worker is running
            while self._is_running:
                elapsed_seconds = time.time() - self.start_time # Calculate elapsed time
                if elapsed_seconds >= self.total_duration: break 
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)

                # Calculate remaining time and update progress bar
                remaining_seconds = max(0, int(self.total_duration - elapsed_seconds))
                progress_value_for_bar = int(elapsed_seconds) 
                self.progress_updated.emit(progress_value_for_bar, remaining_seconds)
            
            if self._is_running: 
                 self.status_updated.emit("Recording finished.")
                 self.progress_updated.emit(self.total_duration, 0) 
        except Exception as e:
            self.recording_error.emit(f"Error during recording: {e}")
            frames = [] 
        finally:
            if stream: stream.stop_stream(); stream.close()
            p_record.terminate()
        self.recording_finished.emit(frames) 

    def stop(self):
        """
        Stops the recording process by setting the running flag to False.
        """
        self._is_running = False

class AudioPlayer(QThread):
    """
    Player thread responsible for playing audio in the background.
    """
    status_updated = pyqtSignal(str)  # Signal for updating the status label

    def __init__(self, device_params, frames):
        super().__init__()
        self.play_frames = frames #b''.join(frames)
        self.device_params = device_params
        self._is_running = True
    
    def run(self):
        p_play = pyaudio.PyAudio() # Create a new PyAudio instance for recording
        self.status_updated.emit(f"Playing stream on {self.device_params['name']}...")
        try:
            stream = p_play.open(format=self.device_params['format'],
                                   channels=self.device_params['channels'],
                                   rate=self.device_params['rate'],
                                   frames_per_buffer=CHUNK_SIZE,
                                   output=True
                                   )
            for i in range(0,len(self.play_frames)):
                stream.write(self.play_frames[i])

            if self._is_running:
                self.status_updated.emit("Finish playing.")
        except Exception as e:
            self.status_updated.emit(f"Error during playback: {e}")
        finally:
            if stream: stream.stop_stream(); stream.close()
            p_play.terminate()

    def stop(self):
        """
        Stops the playback by setting the running flag to False.
        """
        self._is_running = False

# --- Clickable Label for PyQt5 ---
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()

# --- Algorithm Class for Sound Analysis ---
class SoundAnalyzer:
    """
    Provides methods for analyzing audio data to detect sound events
    and classify them as pulsatile or non-pulsatile.
    """
    def __init__(self):
        # No specific initialization needed for these static-like methods
        pass

    def analyze_audio(self, y, sample_rate):
        """
        Analyze audio to detect sound presence and classify it as pulsatile or non-pulsatile.
        Returns a tuple: (sound_detected, pulsatile_result).
        """
        if y is None or len(y) == 0:
            return False, "Error"  # Return "Error" if no audio data is available

        # Normalize the waveform
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))  # Normalize the audio signal to [-1, 1]

        # === Absolute waveform ===
        abs_waveform = np.abs(y)

        # === Chunking ===
        chunk_size = int(sample_rate * (WINDOW_DURATION_MS / 1000))
        num_chunks = len(abs_waveform) // chunk_size
        usable_waveform = abs_waveform[:num_chunks * chunk_size]
        chunks = usable_waveform.reshape(num_chunks, chunk_size)

        # === RMS computation ===
        rms_values = np.sqrt(np.mean(chunks**2, axis=1))
        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)

        # === Sound Detection (Relative RMS Threshold) ===
        relative_ratio = std_rms / mean_rms
        sound_detected = relative_ratio > RELATIVE_THRESHOLD  # If ratio exceeds threshold, sound is detected

        # If sound is not detected, return early with "No Sound"
        if not sound_detected:
            return False, "No Sound"  # No sound detected, return "No Sound"

        # === Threshold RMS to create cropped RMS (set values below threshold to 0) ===
        rms_threshold = mean_rms + std_rms
        cropped_rms = np.copy(rms_values)
        cropped_rms[cropped_rms < rms_threshold] = 0

        # === Peak Detection on Cropped RMS ===
        peaks, _ = find_peaks(cropped_rms, distance=DISTANCE)

        # Map the peak indices to the time axis
        time_axis = np.arange(len(cropped_rms)) * (WINDOW_DURATION_MS / 1000)  # Time in seconds (50 ms chunks)
        peak_times = time_axis[peaks]  # Retrieve time for each marked peak

        # Calculate the intervals between consecutive peaks (in seconds)
        peak_intervals = np.diff(peak_times)

        # Calculate the average peak interval
        average_peak_interval = np.mean(peak_intervals)

        # Calculate BPM (60 seconds in a minute)
        bpm = 60 / average_peak_interval

        # === Pulsatile vs Non-Pulsatile Classification ===
        pulsatile_result = "Pulsatile" if PULSATILE_BPM_MIN <= bpm <= PULSATILE_BPM_MAX else "Non-Pulsatile"

        return True, pulsatile_result  # Return sound detection status and pulsatile classification

class MainWindow(QMainWindow):
    """
    The main window of the application, responsible for the UI layout, managing audio recordings,
    and interacting with the AudioController and AudioWorker.
    """
    PAGE_IDLE = 0
    PAGE_RECORDING = 1
    PAGE_ANALYSIS = 2

    def __init__(self):
        """
        Initializes the main window, sets up UI components, and starts the application logic.
        """
        super().__init__()
        # Initialize the audio controller, sound analyzer, and worker thread
        self.audio_controller = AudioController()
        self.sound_analyzer = SoundAnalyzer()
        self.worker_thread = None
        self.player_thread = None
        self.recorded_frames = None
        self.current_audio_filepath = None
        self.initUI()  # Initialize the UI components
        self.check_audio_device_status()  # Check the audio device status when the window starts

# --- UI Initialization and Setup ---
    def initUI(self):
        """
        Initializes the user interface (UI), setting up widgets, layouts, and controls.
        """
        self.setWindowTitle(f"OBJECTIVE TINNITUS RECORDER")
        self.setGeometry(100, 100, 900, 700)

        # Set up the stacked widget (for switching between different pages of the UI)
        self.stacked_widget = QStackedWidget()
        self.idle_widget = QWidget()
        self.setup_idle_ui()
        self.recording_widget = QWidget()
        self.setup_recording_ui()
        self.analysis_page = QWidget()
        self.setup_analysis_page()

        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.idle_widget)
        self.stacked_widget.addWidget(self.recording_widget)
        self.stacked_widget.addWidget(self.analysis_page)

        # Setup control frame with buttons for starting, saving, and resetting recording
        self.control_frame = QFrame()
        self.control_frame.setFrameShape(QFrame.StyledPanel)
        self.control_frame.setFixedHeight(80)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10,5,10,5)
        button_height = 55
        button_width = 190

        # --- Control Buttons ---
        self.start_stop_button = QPushButton("START")
        self.start_stop_button.clicked.connect(self.handle_start_stop)
        self.start_stop_button.setFixedHeight(button_height); self.start_stop_button.setMinimumWidth(button_width)
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)

        # --- Control Buttons ---
        self.open_button = QPushButton("OPEN")
        self.open_button.clicked.connect(self.handle_open_file)
        self.open_button.setFixedHeight(button_height); self.open_button.setMinimumWidth(button_width)
        self.open_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)

        # Save As button
        self.save_as_button = QPushButton("SAVE")
        self.save_as_button.clicked.connect(self.handle_save_as)
        self.save_as_button.setFixedHeight(button_height); self.save_as_button.setMinimumWidth(button_width)
        self.save_as_button.setEnabled(False)
        self.save_as_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
        """)

        # Finish/Reset button
        self.finish_reset_button = QPushButton("RESET")
        self.finish_reset_button.clicked.connect(self.handle_finish_reset)
        self.finish_reset_button.setFixedHeight(button_height); self.finish_reset_button.setMinimumWidth(button_width)
        self.finish_reset_button.setEnabled(False)
        self.finish_reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ffca28;
                color: black;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)

        # --- Play Buttons ---
        self.play_button = QPushButton("PLAY")
        self.play_button.clicked.connect(self.handle_play_audio)
        self.play_button.setFixedHeight(button_height); self.play_button.setMinimumWidth(button_width)
        self.play_button.setEnabled(False)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #5827c4;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #7b58c7;
            }
        """)

        # --- Layout for Control Frame ---
        control_layout.addStretch(1)
        control_layout.addWidget(self.start_stop_button)
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.save_as_button)
        control_layout.addWidget(self.finish_reset_button)
        control_layout.addWidget(self.play_button)
        # control_layout.addWidget(self.true_exit_button)
        control_layout.addStretch(1)
        self.control_frame.setLayout(control_layout)

        # --- Stacked Widget Layout ---`
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(self.control_frame)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Floating "X" Exit Button at Top-Right ---
        self.exit_icon_button = QPushButton("X", self)
        self.exit_icon_button.setFixedSize(40, 40)
        self.exit_icon_button.setStyleSheet("""
            QPushButton {
                background-color: #343a40;
                color: white;
                border: none;
                font-size: 16px;
                font-weight: bold;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #23272b;
            }
        """)
        self.exit_icon_button.clicked.connect(self.close)
        self.exit_icon_button.raise_()


        self.status_label = QLabel("Status: Initializing...")
        self.statusBar().addWidget(self.status_label)
        self.data_stats_label = QLabel("Live: 0.00 MB @ 0.00 MB/s")
        self.statusBar().addPermanentWidget(self.data_stats_label)

    def setup_idle_ui(self):
        """
        Sets up the widgets for the idle page (the first page the user sees).
        """
        layout = QVBoxLayout(self.idle_widget)
        self.device_status_label_idle = QLabel("Checking for a suitable audio device...")
        self.device_status_label_idle.setAlignment(Qt.AlignCenter)
        self.device_status_label_idle.setStyleSheet("font-size: 14px; color: #555;")
        layout.addWidget(self.device_status_label_idle)

        fixed_duration_label = QLabel(f"Recording for {FIXED_RECORDING_DURATION_SECONDS}s @ {TARGET_SAMPLE_RATE}Hz")
        fixed_duration_label.setAlignment(Qt.AlignCenter)
        fixed_duration_label.setStyleSheet("font-size: 16px; margin-bottom: 30px;")
        layout.addWidget(fixed_duration_label)

        start_instruction_label = QLabel("Click START to begin recording.")
        start_instruction_label.setAlignment(Qt.AlignCenter)
        start_instruction_label.setStyleSheet("font-size: 18px; color: #333; font-weight: bold;")
        layout.addWidget(start_instruction_label)
        layout.addStretch()

    def setup_recording_ui(self):
        """
        Sets up the widgets for the recording page.
        """
        layout = QVBoxLayout(self.recording_widget)
        self.recording_progress_bar = QProgressBar()
        self.recording_progress_bar.setFixedHeight(25)
        self.recording_progress_bar.setRange(0, FIXED_RECORDING_DURATION_SECONDS)
        self.recording_progress_bar.setValue(0)
        self.recording_progress_bar.setFormat(f"{FIXED_RECORDING_DURATION_SECONDS}s remaining")
        self.recording_progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.recording_progress_bar)

        self.live_waveform_placeholder = QLabel(f"Recording...") #\n(Live waveform placeholder)")
        self.live_waveform_placeholder.setAlignment(Qt.AlignCenter)
        self.live_waveform_placeholder.setStyleSheet("background-color: #1A1A1A; border: 1px solid #333; font-size: 16px; color: #A0A0A0;")
        self.live_waveform_placeholder.setMinimumHeight(350)
        self.live_waveform_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.live_waveform_placeholder)

    def setup_analysis_page(self):
        """
        Sets up the widgets for the analysis results page.
        """
        layout = QVBoxLayout(self.analysis_page)  # Use vertical layout to stack elements one on top of the other
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(0)  # Remove space between frames

        # Frame for the first waveform
        self.frame_waveform = QFrame(self.analysis_page)
        self.frame_waveform.setFrameShape(QFrame.StyledPanel)
        self.frame_waveform.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expansion
        waveform_layout = QVBoxLayout(self.frame_waveform)
        self.analysis_canvas_1 = FigureCanvas(Figure(figsize=(8, 6), dpi=100))
        waveform_layout.addWidget(self.analysis_canvas_1)
        layout.addWidget(self.frame_waveform)  # Add the waveform frame to the vertical layout

        # Frame for the results display box
        self.result_frame = QFrame(self.analysis_page)
        self.result_frame.setFrameShape(QFrame.StyledPanel)
        self.result_frame.setFixedHeight(100)  # Set a fixed height for the result display box (can adjust if needed)
        result_layout = QVBoxLayout(self.result_frame)
        self.result_label = ClickableLabel(self.result_frame)
        self.result_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 160);
            color: white;
            font-size: 32px;
            font-weight: bold;
            padding: 20px;
            border-radius: 12px;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.hide()  # Initially hidden
        result_layout.addWidget(self.result_label)
        layout.addWidget(self.result_frame)  # Add the result display box to the vertical layout

        # Frame for the second spectrogram
        self.frame_spectrogram = QFrame(self.analysis_page)
        self.frame_spectrogram.setFrameShape(QFrame.StyledPanel)
        self.frame_spectrogram.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expansion
        spectrogram_layout = QVBoxLayout(self.frame_spectrogram)
        self.analysis_canvas_2 = FigureCanvas(Figure(figsize=(8, 6), dpi=100))
        spectrogram_layout.addWidget(self.analysis_canvas_2)
        layout.addWidget(self.frame_spectrogram)  # Add the spectrogram frame to the vertical layout

# --- Event Handlers ---
    def handle_start_stop(self):
        """
        Handles clicks on the "START"/"STOP" button. This is a central part of the application's logic.
        """
        # If the worker thread is currently running, the button acts as a "STOP" button.
        if self.worker_thread and self.worker_thread.isRunning():
            self.update_status_bar_text("Stopping recording early...")
            self.worker_thread.stop()
            self.start_stop_button.setEnabled(False)
        else:
            # If no recording is running, the button acts as a "START" button.
            # First, double-check that an audio device is ready.
            if not self.audio_controller.is_ready():
                self.check_audio_device_status()
                if not self.audio_controller.is_ready():
                    return

            # Prepare for a new recording
            self.recorded_frames = []  # ✅ Needed for live MB tracking
            self.current_audio_filepath = None
            self.stacked_widget.setCurrentIndex(self.PAGE_RECORDING)
            self.recording_progress_bar.setValue(0)
            self.recording_progress_bar.setFormat(f"{FIXED_RECORDING_DURATION_SECONDS}s remaining")

            # Update button text and style to "STOP RECORDING".
            self.start_stop_button.setText("STOP RECORDING")
            self.start_stop_button.setStyleSheet(
                "font-weight: bold; font-size: 16px; background-color: #DC3545; color: white; border-radius: 6px;"
            )
            self.save_as_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.finish_reset_button.setEnabled(False)
            self.open_button.setEnabled(False)

            # Start the background recording thread
            self.worker_thread = AudioWorker(self.audio_controller.device_params, FIXED_RECORDING_DURATION_SECONDS)
            # Connect signals from the worker thread to handler methods (slots) in this MainWindow class.
            self.worker_thread.progress_updated.connect(self.update_recording_progress)
            self.worker_thread.status_updated.connect(self.update_status_bar_text)
            self.worker_thread.recording_finished.connect(self.handle_recording_completion)
            self.worker_thread.recording_error.connect(self.on_recording_error_and_reset)
            self.worker_thread.finished.connect(self.on_worker_thread_actually_finished)
            self.worker_thread.start()

    def handle_open_file(self):
        """
        Handles clicks on the "Open" button. This loads an audio file for analysis.
        """
        # Open folder to let user select a WAV file
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            DEFAULT_OUTPUT_DIR,
            "WAV files (*.wav)"
        )

        frame_size = self.audio_controller.device_params['channels'] * WIDTH_SAMPLE
        frames_per_buffer = max(1,CHUNK_SIZE // frame_size)
        bytes_per_chunk = frames_per_buffer * frame_size
        formatted_frames = []
        # Check if filepath exists
        if filepath:
            try:
                # Read the selected WAV file
                with wave.open(filepath,'rb') as rd:

                    # Reset and asks user to select a correct file if the sample rate or sample width(16bit, 24bit) does not match the current settings
                    if rd.getnchannels() != self.audio_controller.device_params['channels'] or rd.getsampwidth() != WIDTH_SAMPLE or rd.getframerate() != self.audio_controller.device_params['rate']:
                        self.handle_finish_reset()
                        self.update_status_bar_text("Sample bitwidth mismatch detected. Please select a correct audio file recorded using this device to analyze.")
                        QMessageBox.information(self, "Wrong audio file", "Sample bitwidth mismatch detected. Please select a correct audio file recorded using this device to analyze.")
                        QApplication.processEvents()
                        return

                    # Read the data into frames of size frames_per_buffer just like in live recording 
                    while True:
                        data=rd.readframes(frames_per_buffer)
                        if not data:
                            break
                        formatted_frames.append(data)

                self.update_status_bar_text("Recording loaded.")
                QApplication.processEvents()
                
                # Analyze the audio from the loaded file
                self.handle_recording_completion(formatted_frames)

                # Update status bar
                self.update_status_bar_text("Recording analyzed.")
                QApplication.processEvents()

            except Exception as e:
                print(f"Error loading WAV file: {e}")
                self.update_status_bar_text("Failed to load the recording.")
                QApplication.processEvents()
        else:
            self.update_status_bar_text("File selection cancelled.")
            QApplication.processEvents()
        return

    def handle_play_audio(self):
        """
        Handles clicks on the "PLAY" button.
        """
        #Check if there is data is frames
        if not self.recorded_frames:
                QMessageBox.warning(self, "No Data", "Unable to play audio")
                return
        
        if len(b''.join(self.recorded_frames)) % CHUNK_SIZE != 0:
            self.update_status_bar_text("Audio data length is not a multiple of {WIDTH_SAMPLE}.")
            QMessageBox.information(self, "Sample bitrate mismatch detected.", "Audio file does not match bitrate. Please select a correct audio file recorded using this device to analyze.")
            QApplication.processEvents()

        # If the player thread is currently running, the button acts as a "STOP" button.
        if self.player_thread and self.player_thread.isRunning():
            self.update_status_bar_text("Stop playback early....")
            self.player_thread.stop()
            self.player_thread.terminate()
            self.on_player_thread_actually_finished()
        else:
            # Update button text and style to "STOP PLAYING".
            self.play_button.setText("STOP PLAYING")
            self.play_button.setStyleSheet(
                "font-weight: bold; font-size: 16px; background-color: #DC3545; color: white; border-radius: 6px;"
            )
            #Disable other buttons while playing
            self.start_stop_button.setEnabled(False)
            self.open_button.setEnabled(False)
            self.save_as_button.setEnabled(False)
            self.finish_reset_button.setEnabled(False)

            # Start the background playing thread
            self.player_thread = AudioPlayer(self.audio_controller.device_params, self.recorded_frames)
            # Connect signals from the player thread to handler methods (slots) in this MainWindow class.
            self.player_thread.status_updated.connect(self.update_status_bar_text)
            self.player_thread.finished.connect(self.on_player_thread_actually_finished)
            self.player_thread.start()

    def handle_save_as(self):
        """
        Handles clicks on the "SAVE" button.
        """
        if not self.recorded_frames:
            QMessageBox.warning(self, "No Data", "No audio data to save.")
            return

        # Generate a default filename based on the current time and device info.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dev_name = self.audio_controller.device_params.get('name', 'AudioDevice').replace(" ", "_")[:15]
        rate = self.audio_controller.device_params.get('rate', TARGET_SAMPLE_RATE)
        ch = self.audio_controller.device_params.get('channels', TARGET_CHANNELS)
        default_filename = f"rec_{dev_name}_{rate}Hz_{ch}ch_{timestamp}.wav"

        # Open a standard "Save File" dialog.
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Audio As",
            os.path.join(DEFAULT_OUTPUT_DIR, default_filename), "WAV files (*.wav)")

        if filepath:
            self.current_audio_filepath = filepath
            self.update_status_bar_text(f"Saving to {os.path.basename(filepath)}...")
            QApplication.processEvents()

            if self.audio_controller.save_audio_to_file(self.recorded_frames, filepath):
                self.update_status_bar_text(f"Audio successfully saved to {os.path.basename(filepath)}")
                QMessageBox.information(self, "Save Successful", f"Audio saved to:\n{filepath}")
            else:
                self.update_status_bar_text("Save failed.")
                QMessageBox.critical(self, "Save Error", "Failed to save audio file.")
        else:
            self.update_status_bar_text("Save As cancelled.")

    def handle_finish_reset(self):
        """
        Handles clicks on the "RESET" button.
        """
        self.update_status_bar_text("Resetting to idle state...")
        self.reset_ui_to_idle_state_internal()
        self.check_audio_device_status()

    def handle_recording_completion(self, frames):
        """
        This method is called when the AudioWorker thread finishes successfully or when an audio file is successfully loaded.
        This function will start the analysis of the live recording or the loaded audio file.
        """
        self.recorded_frames = frames

        # Reset the start/stop button to its "START" state.
        self.start_stop_button.setText("START")
        self.start_stop_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4CAF50; color: white; border-radius: 6px;")
        self.start_stop_button.setEnabled(self.audio_controller.is_ready())

        # If for some reason no frames were captured, show a warning and reset.
        if not self.recorded_frames:
            self.update_status_bar_text("No audio data captured. Ready for new recording.")
            QMessageBox.warning(self, "No Data", "No audio data was captured.")
            self.stacked_widget.setCurrentIndex(self.PAGE_IDLE)
            self.finish_reset_button.setEnabled(True)
            return

        self.update_status_bar_text("Generating plots...")
        QApplication.processEvents()

        # Ask the audio controller to compute the waveform and spectrogram data.
        y, sr, S_mel_db = self.audio_controller.compute_audio_analysis_data(self.recorded_frames)

        if y is not None:
            self.update_analysis_plots(y, sr, S_mel_db)
            self.run_sound_check() #Start analyzing audio for presence of sound and if so, update "Pulsatile" or "Non-Pulsatile" result
            self.update_status_bar_text("Plot displayed. Ready to save.")
            self.save_as_button.setEnabled(True)
            self.open_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.finish_reset_button.setEnabled(True)
            self.stacked_widget.setCurrentIndex(self.PAGE_ANALYSIS)
        else:
            self.update_status_bar_text("Plot generation failed.")
            QMessageBox.warning(self, "Plot Error", "Failed to compute analysis data from recording.")
            self.reset_ui_to_idle_state_internal()

    def on_recording_error_and_reset(self, error_message):
        """
        This method is called if the AudioWorker thread emits a recording_error signal.
        """
        self.update_status_bar_text(f"Recording Error!")
        QMessageBox.critical(self, "Recording Error", error_message)
        self.reset_ui_to_idle_state_internal()
        self.finish_reset_button.setEnabled(True)

    def on_worker_thread_actually_finished(self):
        """
        This method is connected to the QThread's 'finished' signal.
        """
        if not (self.worker_thread and self.worker_thread.isRunning()):
            self.start_stop_button.setText("START")
            self.start_stop_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4CAF50; color: white; border-radius: 6px;")
            self.start_stop_button.setEnabled(self.audio_controller.is_ready())
            if self.worker_thread:
                self.worker_thread = None
    
    def on_player_thread_actually_finished(self):
        """
        This method is connected to QThread AudioPlayer when playback is finished
        """
        # Reset the start/stop button to its "PLAY" state and enable other buttons.
        if not (self.player_thread and self.player_thread.isRunning()):
            self.play_button.setText("PLAY")
            self.play_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #5827c4; color: white; border-radius: 6px;")
            self.start_stop_button.setEnabled(True)
            self.open_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.save_as_button.setEnabled(True)
            self.finish_reset_button.setEnabled(True)

    def resizeEvent(self, event):
        """
        This is a built-in Qt event that is automatically called whenever the main window is resized.
        """
        super().resizeEvent(event)
        # Update the layout of the two waveforms and the result box
        self.update_overlay_positions()
        self.analysis_canvas_1.updateGeometry()
        self.analysis_canvas_2.updateGeometry()

    def closeEvent(self, event):
        """
        This is a built-in Qt event that is called when the user tries to close the window.
        """
        self.update_status_bar_text("Exiting application...")
        QApplication.processEvents()
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(1000)
        if self.audio_controller:
            self.audio_controller.close()
        print("Application closed.")
        super().closeEvent(event)

# --- Audio and Analysis Methods ---
    def check_audio_device_status(self):
        """
        Updates the status of the audio device and notifies the user if a valid device is found.
        """
        self.update_status_bar_text("Checking for a suitable audio device...")
        QApplication.processEvents()

        # Ensure audio controller is initialized
        if not self.audio_controller.is_ready():
            self.audio_controller = AudioController()

        if self.audio_controller.is_ready():
            dev_name = self.audio_controller.device_params['name']
            dev_rate = self.audio_controller.device_params['rate']
            msg = f"Device Ready: {dev_name} @ {dev_rate}Hz"
            self.update_status_bar_text(msg)
            self.start_stop_button.setEnabled(True)
            if hasattr(self, 'device_status_label_idle'):
                self.device_status_label_idle.setText(msg)
        else:
            msg = f"No input device found supporting {TARGET_SAMPLE_RATE}Hz. Check console."
            self.update_status_bar_text(msg)
            if hasattr(self, 'device_status_label_idle'):
                self.device_status_label_idle.setText(msg + "\nTry re-plugging or check audio settings.")
            QMessageBox.critical(self, "Device Error", msg + "\nThe application might not function correctly.")
            self.start_stop_button.setEnabled(False)

    def run_sound_check(self):
        """
        Check if sound is detected and update result label.
        """
        if not self.recorded_frames:  # Direct check for 'recorded_frames' (no need for hasattr)
            return

        y, sr, _ = self.audio_controller.compute_audio_analysis_data(self.recorded_frames)
        if y is None:
            return

        # Analyze the audio: Detect sound presence and classify as Pulsatile or Non-Pulsatile
        sound_detected, result_text = self.sound_analyzer.analyze_audio(y, sr)

        # If sound is detected, you could log or perform additional checks
        if sound_detected:
            print("Sound detected, proceeding with classification...")
        else:
            print("No sound detected.")

        # Update the result label with the appropriate text (No Sound, Pulsatile, or Non-Pulsatile)
        if self.result_label:
            self.result_label.setText(result_text)  # Display the result (Pulsatile, Non-Pulsatile, or No Sound)
            self.result_label.show()  # Ensure the result label is visible

# --- UI Update and Helper Methods ---
    def update_analysis_plots(self, y, sr, S_mel_db):
        """
        Clears the existing figure and draws new waveform and spectrogram plots on the canvas.
        """

        # Clear both canvas figures
        self.analysis_canvas_1.figure.clear()  # Clears the waveform canvas
        self.analysis_canvas_2.figure.clear()  # Clears the spectrogram canvas

        # --- Plot on the first canvas (Waveform) ---
        ax_waveform = self.analysis_canvas_1.figure.add_subplot(1, 1, 1)  # Only one subplot for the waveform
        librosa.display.waveshow(y, sr=sr, ax=ax_waveform, color='darkcyan')

        # Remove axis, labels, and title for the waveform plot
        ax_waveform.set_xticks([])  # Remove x-axis ticks
        ax_waveform.set_yticks([])  # Remove y-axis ticks
        ax_waveform.spines['top'].set_visible(False)  # Hide top spine
        ax_waveform.spines['right'].set_visible(False)  # Hide right spine
        ax_waveform.spines['left'].set_visible(False)  # Hide left spine
        ax_waveform.spines['bottom'].set_visible(False)  # Hide bottom spine
        ax_waveform.set_title('')  # Remove title

        # Adjust layout
        self.analysis_canvas_1.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # --- Plot on the second canvas (Spectrogram) ---
        ax_spectrogram = self.analysis_canvas_2.figure.add_subplot(1, 1, 1)  # Only one subplot for the spectrogram
        if S_mel_db is not None:
            librosa.display.specshow(S_mel_db, sr=sr, hop_length=SPEC_HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax_spectrogram, fmax=sr/2, cmap='viridis')

        # Remove axis, labels, and title for the spectrogram plot
        ax_spectrogram.set_xticks([])  # Remove x-axis ticks
        ax_spectrogram.set_yticks([])  # Remove y-axis ticks
        ax_spectrogram.set_yticklabels([])  # Remove y-axis labels (Hz)
        ax_spectrogram.set_ylabel('')  # Explicitly remove the "Hz" label on the y-axis
        ax_spectrogram.spines['top'].set_visible(False)  # Hide top spine
        ax_spectrogram.spines['right'].set_visible(False)  # Hide right spine
        ax_spectrogram.spines['left'].set_visible(False)  # Hide left spine
        ax_spectrogram.spines['bottom'].set_visible(False)  # Hide bottom spine
        ax_spectrogram.set_title('')  # Remove title

        # Adjust layout
        self.analysis_canvas_2.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # --- Ensure the result label is shown properly ---
        if self.result_label:
            # The result is already handled in run_sound_check, no need to handle it here
            pass

        # Adjust the result display box layout, making sure it's not covered by the plot
        self.result_frame.setFixedHeight(100)  # Set the height of the result display box

        # Draw both canvases
        self.analysis_canvas_1.draw()
        self.analysis_canvas_2.draw()
        print("Analysis plots updated in the GUI.")

    def update_recording_progress(self, elapsed_seconds, remaining_seconds):
        self.recording_progress_bar.setValue(elapsed_seconds)
        self.recording_progress_bar.setFormat(f"{remaining_seconds}s remaining")

        # Estimate total data recorded so far
        if self.audio_controller and self.audio_controller.device_params:
            sr = self.audio_controller.device_params['rate']
            estimated_samples = elapsed_seconds * sr
            estimated_bytes = estimated_samples * 2  # 16-bit audio = 2 bytes/sample
            total_mb = estimated_bytes / (1024 * 1024)
            mbps = total_mb / elapsed_seconds if elapsed_seconds > 0 else 0
            self.data_stats_label.setText(f"Live: {total_mb:.2f} MB @ {mbps:.2f} MB/s")

    def update_status_bar_text(self, message):
        """
        To help set the text of the main status label.
        """
        self.status_label.setText(f"Status: {message}")

    def update_overlay_positions(self):
        """
        Calculates and updates the positions of floating widgets (result label and exit button)
        """
        from PyQt5.QtCore import QPoint

        # Ensure the canvas is updated with the latest geometry
        self.analysis_canvas_1.updateGeometry()
        self.analysis_canvas_2.updateGeometry()

        # Get the canvas position and size for both canvases
        canvas_pos_1 = self.analysis_canvas_1.mapTo(self, QPoint(0, 0))
        canvas_width_1 = self.analysis_canvas_1.width()
        canvas_height_1 = self.analysis_canvas_1.height()  # Fix: Ensure canvas height is retrieved

        canvas_pos_2 = self.analysis_canvas_2.mapTo(self, QPoint(0, 0))
        canvas_width_2 = self.analysis_canvas_2.width()
        canvas_height_2 = self.analysis_canvas_2.height()  # Fix: Ensure canvas height is retrieved

        # Position the result label (centered between the two waveforms)
        self._position_result_label(canvas_pos_1, canvas_pos_2, canvas_width_1, canvas_width_2, canvas_height_1, canvas_height_2)

        # Position the exit icon (top-right)
        self._position_exit_icon()

    def _position_result_label(self, canvas_pos_1, canvas_pos_2, canvas_width_1, canvas_width_2, canvas_height_1, canvas_height_2):
        """
        Helper to calculate the exact position for the result label.
        """
        if hasattr(self, 'result_label') and self.result_label.isVisible():
            # Calculate the position in the middle between the two waveforms
            center_x = (canvas_pos_1.x() + canvas_pos_2.x() + canvas_width_1 + canvas_width_2) // 2
            center_y = max(canvas_pos_1.y(), canvas_pos_2.y()) + min(canvas_height_1, canvas_height_2) // 2

            self.result_label.move(
                center_x - self.result_label.width() // 2,
                center_y - self.result_label.height() // 2
            )
            self.result_label.raise_()

    def _position_exit_icon(self):
        """
        Helper to calculate the exact position for the exit icon.
        """
        if hasattr(self, 'exit_icon_button'):
            self.exit_icon_button.move(
                self.width() - self.exit_icon_button.width() - 10,
                10
            )

    def reset_ui_to_idle_state_internal(self):
        """
        Resets the entire UI and application state back to the initial idle condition.
        """
        self.status_label.setText("Status: Ready" if self.audio_controller.is_ready() else "Audio device not ready")
        self.start_stop_button.setText("START")
        self.start_stop_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4CAF50; color: white; border-radius: 6px;")
        self.start_stop_button.setEnabled(self.audio_controller.is_ready())
        self.save_as_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.play_button.setText("PLAY")
        self.play_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #5827c4; color: white; border-radius: 6px;")
        self.finish_reset_button.setEnabled(False)
        self.data_stats_label.setText("Live: 0.00 MB @ 0.00 MB/s")
        if hasattr(self, 'recording_progress_bar'): self.recording_progress_bar.setValue(0)
        self.stacked_widget.setCurrentIndex(self.PAGE_IDLE)
        self.recorded_frames = None
        self.current_audio_filepath = None

        if hasattr(self, 'analysis_figure'):
            self.analysis_figure.clear()
            self.analysis_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.showFullScreen()
    sys.exit(app.exec_())
      