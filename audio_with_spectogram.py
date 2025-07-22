import sys
import os
import time
import numpy as np
from scipy import signal 
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Qt5Agg') # Use Qt5Agg for Matplotlib backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure # Import Figure for embedding
import pyaudio
import soundfile as sf
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
from PyQt5.QtCore import QTimer

# --- Configuration ---
# TARGET_DEVICE_NAME_SUBSTRING is removed, the script will find any compatible device.
TARGET_SAMPLE_RATE = 192000 
TARGET_CHANNELS = 1 
TARGET_FORMAT = pyaudio.paInt16 
CHUNK_SIZE = 4096
DEFAULT_OUTPUT_DIR = "gui_recordings_192khz_30s" # More generic name
FIXED_RECORDING_DURATION_SECONDS = 30
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# --- Spectrogram Parameters ---
SPEC_N_FFT = 8192         
SPEC_HOP_LENGTH = 2048    
SPEC_N_MELS = 128         
SPEC_WINDOW = 'hamming'   

# --- Backend Logic Class (AudioController) ---
class AudioController:
    def __init__(self):
        self.device_params = None
        self.pyaudio_instance = None
        self.initialize_pyaudio()
        if self.pyaudio_instance:
            self._find_and_verify_workable_device() # Changed from Pisound specific

# --- Device Initialization and Configuration ---
    def initialize_pyaudio(self):
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
        except Exception as e:
            print(f"Failed to initialize PyAudio: {e}")
            self.pyaudio_instance = None

    def _find_and_verify_workable_device(self): # Renamed and logic changed
        if not self.pyaudio_instance:
            print("PyAudio not initialized. Cannot find devices.")
            self.device_params = None
            return

        print(f"Searching for an input device supporting {TARGET_SAMPLE_RATE}Hz...")
        
        # First, try the default input device
        try:
            default_info = self.pyaudio_instance.get_default_input_device_info()
            device_index = default_info['index']
            device_name = default_info['name']
            print(f"Checking default input device: {device_name} (Index: {device_index})")
            if default_info.get('maxInputChannels', 0) >= TARGET_CHANNELS and \
               self.pyaudio_instance.is_format_supported(TARGET_SAMPLE_RATE, input_device=device_index,
                                                         input_channels=TARGET_CHANNELS, input_format=TARGET_FORMAT):
                self.device_params = {
                    'index': device_index, 'name': device_name, 'rate': TARGET_SAMPLE_RATE,
                    'channels': TARGET_CHANNELS, 'format': TARGET_FORMAT,
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
                # print(f"Could not check device {i} ({info.get('name', 'Unknown')}): {e}") # Can be verbose
                continue
        
        print(f"Error: No suitable audio input device found supporting {TARGET_SAMPLE_RATE}Hz.")
        self.device_params = None

# --- Audio Handling ---
    def is_ready(self):
        return self.device_params is not None

    def save_audio_to_file(self, frames, filepath): 
        if not frames or not self.device_params: return False
        print(f"\nSaving audio to: {filepath}")
        try:
            audio_data_np = np.frombuffer(b''.join(frames), dtype=np.int16)
            sf.write(filepath, audio_data_np, self.device_params['rate'], subtype='PCM_16')
            print("Audio saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False

# --- Audio Analysis ---
    def compute_audio_analysis_data(self, frames):
        """
        Converts frames to a plottable audio array and computes the Log-Mel spectrogram data.
        """
        if not self.device_params or not frames:
            return None, None, None
        
        try:
            print("\nComputing analysis data from recorded frames...")
            y = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
            sr = self.device_params['rate']
            
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
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            print("PyAudio instance terminated.")

# --- PyQt5 Worker Thread for Recording ---
class AudioWorker(QThread):
    progress_updated = pyqtSignal(int, int)
    status_updated = pyqtSignal(str)
    recording_finished = pyqtSignal(list) 
    recording_error = pyqtSignal(str)

    def __init__(self, device_params, duration):
        super().__init__()
        self.device_params = device_params
        self.total_duration = duration 
        self._is_running = True
        self.start_time = 0

    def run(self):
        p_record = pyaudio.PyAudio()
        stream = None
        frames = []
        self.status_updated.emit(f"Opening stream on {self.device_params['name']}...")
        try:
            stream = p_record.open(format=self.device_params['format'],
                                   channels=self.device_params['channels'],
                                   rate=self.device_params['rate'],
                                   input=True,
                                   frames_per_buffer=CHUNK_SIZE,
                                   input_device_index=self.device_params['index'])
            self.status_updated.emit(f"Recording for {self.total_duration}s...")
            self.start_time = time.time()
            while self._is_running:
                elapsed_seconds = time.time() - self.start_time
                if elapsed_seconds >= self.total_duration: break 
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
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
        self._is_running = False

# --- Clickable Label for PyQt5 ---
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()

# --- PyQt5 Main Window ---
class MainWindow(QMainWindow):
    PAGE_IDLE = 0
    PAGE_RECORDING = 1
    PAGE_ANALYSIS = 2 

    def __init__(self):
        super().__init__()
        # Initialize the audio controller, sound analyzer, and worker thread
        self.audio_controller = AudioController()
        self.sound_analyzer = SoundAnalyzer()
        self.worker_thread = None
        self.recorded_frames = None
        self.current_audio_filepath = None
        self.initUI()  # Initialize the UI components
        self.check_audio_device_status()  # Check the audio device status when the window starts


# --- Initialize the UI ---
    def initUI(self):
        self.setWindowTitle(f"OBJECTIVE TINNITUS RECORDER") 
        self.setGeometry(100, 100, 900, 700) 

        self.stacked_widget = QStackedWidget()
        self.idle_widget = QWidget()
        self.setup_idle_ui()
        self.recording_widget = QWidget()
        self.setup_recording_ui()
        self.analysis_page = QWidget() 
        self.setup_analysis_page()    

        self.stacked_widget.addWidget(self.idle_widget)
        self.stacked_widget.addWidget(self.recording_widget)
        self.stacked_widget.addWidget(self.analysis_page)

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

        # --- Layout for Control Frame ---
        control_layout.addStretch(1) 
        control_layout.addWidget(self.start_stop_button)
        control_layout.addWidget(self.save_as_button)
        control_layout.addWidget(self.finish_reset_button) 
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

# --- Setup UI Methods ---
    def setup_idle_ui(self):
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

# --- Setup Recording UI ---
    def setup_recording_ui(self):
        layout = QVBoxLayout(self.recording_widget)
        self.recording_progress_bar = QProgressBar()
        self.recording_progress_bar.setFixedHeight(25)
        self.recording_progress_bar.setRange(0, FIXED_RECORDING_DURATION_SECONDS) 
        self.recording_progress_bar.setValue(0) 
        self.recording_progress_bar.setFormat(f"{FIXED_RECORDING_DURATION_SECONDS}s remaining") 
        self.recording_progress_bar.setAlignment(Qt.AlignCenter) 
        layout.addWidget(self.recording_progress_bar)
        self.live_waveform_placeholder = QLabel(f"Recording...\n(Live waveform placeholder)")
        self.live_waveform_placeholder.setAlignment(Qt.AlignCenter) 
        self.live_waveform_placeholder.setStyleSheet("background-color: #1A1A1A; border: 1px solid #333; font-size: 16px; color: #A0A0A0;") 
        self.live_waveform_placeholder.setMinimumHeight(350) 
        self.live_waveform_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.live_waveform_placeholder)

# --- Analysis Page Setup ---
    def setup_analysis_page(self):
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


# --- Generate Plots ---
    def update_analysis_plots(self, y, sr, S_mel_db):
        """Clears the existing figure and draws new waveform and spectrogram plots on the canvas."""

        # Clear both canvas figures
        self.analysis_canvas_1.figure.clear()  # Clears the waveform canvas
        self.analysis_canvas_2.figure.clear()  # Clears the spectrogram canvas

        # --- Plot on the first canvas (Waveform) ---
        ax_waveform = self.analysis_canvas_1.figure.add_subplot(1, 1, 1)  # Only one subplot for the waveform

        # Plot the waveform on the first canvas
        librosa.display.waveshow(y, sr=sr, ax=ax_waveform, color='darkcyan')

        # Remove axis, labels, and title for the waveform plot
        ax_waveform.set_xticks([])  # Remove x-axis ticks
        ax_waveform.set_yticks([])  # Remove y-axis ticks
        ax_waveform.spines['top'].set_visible(False)  # Hide top spine
        ax_waveform.spines['right'].set_visible(False)  # Hide right spine
        ax_waveform.spines['left'].set_visible(False)  # Hide left spine
        ax_waveform.spines['bottom'].set_visible(False)  # Hide bottom spine
        ax_waveform.set_title('')  # Remove title

        # Remove the extra background around the plot area (fill the canvas)
        self.analysis_canvas_1.figure.patch.set_facecolor('white')  # Set background color of figure

        # Adjust layout to fully fill the canvas and prevent overlapping
        self.analysis_canvas_1.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust to fill bottom of the canvas

        # --- Plot on the second canvas (Spectrogram) ---
        ax_spectrogram = self.analysis_canvas_2.figure.add_subplot(1, 1, 1)  # Only one subplot for the spectrogram

        # Plot the Mel Spectrogram on the second canvas
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

        # Remove the extra background around the plot area (fill the canvas)
        self.analysis_canvas_2.figure.patch.set_facecolor('white')  # Set background color of figure

        # Adjust layout to fully fill the canvas and prevent overlapping
        self.analysis_canvas_2.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust to fill bottom of the canvas

        # --- Ensure the result label is shown properly ---
        if self.result_label:
            result_text = "Sound" if self.detect_sound_presence(y) else "Noise"
            self.result_label.setText(result_text)  # Update the result label text (Sound or Noise)
            self.result_label.adjustSize()  # Adjust the size of the label to fit the text
            self.result_label.show()  # Make sure the result label is visible

            # Print the geometry for debugging
            print(f"Result Label Position: {self.result_label.geometry()}")

        # Adjust the result display box layout, making sure it's not covered by the plot
        self.result_frame.setFixedHeight(100)  # Set the height of the result display box
        self.result_label.adjustSize()  # Adjust the size of the label if necessary

        # Draw both canvases
        self.analysis_canvas_1.draw()
        self.analysis_canvas_2.draw()
        print("Analysis plots updated in the GUI.")

<<<<<<< HEAD
#--- Analysis Algorithm ---
=======

    def detect_sound_presence(self, y):
            if y is None or len(y) == 0:
                return False

            sample_rate = self.audio_controller.device_params['rate']
            y = y.astype(np.float32)
            y /= np.max(np.abs(y))  # Normalize

            # === Absolute waveform ===
            abs_waveform = np.abs(y)

            # === Chunking ===
            CHUNK_DURATION_MS = 50
            chunk_size = int(sample_rate * (CHUNK_DURATION_MS / 1000))
            num_chunks = len(abs_waveform) // chunk_size
            usable_waveform = abs_waveform[:num_chunks * chunk_size]
            chunks = usable_waveform.reshape(num_chunks, chunk_size)

            # === RMS computation ===
            rms_values = np.sqrt(np.mean(chunks**2, axis=1))
            mean_rms = np.mean(rms_values)
            std_rms = np.std(rms_values)
            relative_ratio = std_rms / mean_rms
            print(f"[DEBUG] Relative RMS ratio (std/mean): {relative_ratio:.4f}")

            # === Classification B threshold ===
            RELATIVE_THRESHOLD = 0.5
            return relative_ratio > RELATIVE_THRESHOLD #returns true if sound is detected, false otherwise
        
>>>>>>> parent of 0d2f2d3 (Update audio_with_spectogram.py)
    def run_sound_check(self):
            """Check if sound is detected and update result label."""
            
            if not hasattr(self, 'recorded_frames') or not self.recorded_frames:
                return
            
            y, sr, _ = self.audio_controller.compute_audio_analysis_data(self.recorded_frames)
            if y is None:
                return
            
            has_sound = self.detect_sound_presence(y)
            result_text = "✅ Sound" if has_sound else "❌ Noise"

            # Update the result label with the appropriate text (Sound/Noise)
            if self.result_label:
                self.result_label.setText(result_text)  # Display the result (Sound or Noise)
                self.result_label.adjustSize()  # Adjust the size of the label to fit the text
                self.result_label.show()  # Ensure the result label is visible

<<<<<<< HEAD
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
            self.result_label.adjustSize()  # Adjust the size of the label to fit the text
            self.result_label.show()  # Ensure the result label is visible
    
# --- Finds input device ---
    def check_audio_device_status(self):
        """
        Updates the status of the audio device and notifies the user if a valid device is found.
        """
        self.update_status_bar_text("Checking for a suitable audio device...")
=======
    def check_audio_device_status(self): 
        self.update_status_bar_text("Checking for a suitable audio device...") 
>>>>>>> parent of 0d2f2d3 (Update audio_with_spectogram.py)
        QApplication.processEvents()
        if self.audio_controller: self.audio_controller.close()
        self.audio_controller = AudioController() 
        if self.audio_controller.is_ready() and self.audio_controller.device_params:
            dev_name = self.audio_controller.device_params['name']
            dev_rate = self.audio_controller.device_params['rate']
            msg = f"Device Ready: {dev_name} @ {dev_rate}Hz" 
            self.update_status_bar_text(msg)
            if hasattr(self, 'device_status_label_idle'): self.device_status_label_idle.setText(msg)
            self.start_stop_button.setEnabled(True)
        else:
            msg = f"No input device found supporting {TARGET_SAMPLE_RATE}Hz. Check console." 
            self.update_status_bar_text(msg)
            if hasattr(self, 'device_status_label_idle'): self.device_status_label_idle.setText(msg + "\nTry re-plugging or check audio settings.")
            QMessageBox.critical(self, "Device Error", msg + "\nThe application might not function correctly.")
            self.start_stop_button.setEnabled(False)
        self.reset_ui_to_idle_state_internal()

# --- Event Handler for start stop button ---
    def handle_start_stop(self):
        if self.worker_thread and self.worker_thread.isRunning(): 
            self.update_status_bar_text("Stopping recording early...")
            self.worker_thread.stop()
            self.start_stop_button.setEnabled(False) 
        else: 
            if not self.audio_controller.is_ready():
                self.check_audio_device_status() 
                if not self.audio_controller.is_ready():
                    return

            self.recorded_frames = []  # ✅ Needed for live MB tracking
            self.current_audio_filepath = None
            self.stacked_widget.setCurrentIndex(self.PAGE_RECORDING)
            self.recording_progress_bar.setValue(0)
            self.recording_progress_bar.setFormat(f"{FIXED_RECORDING_DURATION_SECONDS}s remaining")
            
            self.start_stop_button.setText("STOP RECORDING")
            self.start_stop_button.setStyleSheet(
                "font-weight: bold; font-size: 16px; background-color: #DC3545; color: white; border-radius: 5px;"
            )
            self.save_as_button.setEnabled(False)
            self.finish_reset_button.setEnabled(False) 

            self.worker_thread = AudioWorker(self.audio_controller.device_params, FIXED_RECORDING_DURATION_SECONDS)
            self.worker_thread.progress_updated.connect(self.update_recording_progress)
            self.worker_thread.status_updated.connect(self.update_status_bar_text)
            self.worker_thread.recording_finished.connect(self.handle_recording_completion) 
            self.worker_thread.recording_error.connect(self.on_recording_error_and_reset)
            self.worker_thread.finished.connect(self.on_worker_thread_actually_finished)
            self.worker_thread.start()

# --- Update recording data size ---
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
        self.status_label.setText(f"Status: {message}")

# --- Handle recording completion ---
    def handle_recording_completion(self, frames):
        self.recorded_frames = frames
        self.start_stop_button.setText("START")
        self.start_stop_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.start_stop_button.setEnabled(self.audio_controller.is_ready())

        if not self.recorded_frames:
            self.update_status_bar_text("No audio data captured. Ready for new recording.")
            QMessageBox.warning(self, "No Data", "No audio data was captured.")
            self.stacked_widget.setCurrentIndex(self.PAGE_IDLE) 
            self.finish_reset_button.setEnabled(True) 
            return

        self.update_status_bar_text("Recording complete. Generating plots...")
        QApplication.processEvents()

        y, sr, S_mel_db = self.audio_controller.compute_audio_analysis_data(self.recorded_frames)

        if y is not None:
            self.update_analysis_plots(y, sr, S_mel_db)
            self.run_sound_check()
            self.update_status_bar_text("Plot displayed. Ready to save.")
            self.save_as_button.setEnabled(True)
            self.finish_reset_button.setEnabled(True)
            self.stacked_widget.setCurrentIndex(self.PAGE_ANALYSIS)
        else:
            self.update_status_bar_text("Plot generation failed.")
            QMessageBox.warning(self, "Plot Error", "Failed to compute analysis data from recording.")
            self.reset_ui_to_idle_state_internal()

# --- Handle Save As functionality ---
    def handle_save_as(self):
        if not self.recorded_frames:
            QMessageBox.warning(self, "No Data", "No audio data to save.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dev_name = self.audio_controller.device_params.get('name', 'AudioDevice').replace(" ", "_")[:15]
        rate = self.audio_controller.device_params.get('rate', TARGET_SAMPLE_RATE)
        ch = self.audio_controller.device_params.get('channels', TARGET_CHANNELS)
        default_filename = f"rec_{dev_name}_{rate}Hz_{ch}ch_{timestamp}.wav"
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Audio As", 
            os.path.join(DEFAULT_OUTPUT_DIR, default_filename), "WAV files (*.wav)")

        if filepath:
            self.current_audio_filepath = filepath
            self.update_status_bar_text(f"Saving to {os.path.basename(filepath)}...")
            QApplication.processEvents()

            if self.audio_controller.save_audio_to_file(self.recorded_frames, filepath):
                self.update_status_bar_text(f"Audio successfully saved to {os.path.basename(filepath)}")
                QMessageBox.information(self, "Save Successful", f"Audio saved to:\n{filepath}")
                self.save_as_button.setEnabled(False) 
            else:
                self.update_status_bar_text("Save failed.")
                QMessageBox.critical(self, "Save Error", "Failed to save audio file.")
        else:
            self.update_status_bar_text("Save As cancelled.")



    def on_recording_error_and_reset(self, error_message):
        self.update_status_bar_text(f"Recording Error!")
        QMessageBox.critical(self, "Recording Error", error_message)
        self.reset_ui_to_idle_state_internal() 
        self.finish_reset_button.setEnabled(True) 

# --- Resets start stop button ---
    def on_worker_thread_actually_finished(self): 
        if not (self.worker_thread and self.worker_thread.isRunning()): 
            self.start_stop_button.setText("START")
            self.start_stop_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4CAF50; color: white; border-radius: 5px;")
            self.start_stop_button.setEnabled(self.audio_controller.is_ready())
            if self.worker_thread: 
                self.worker_thread = None

# --- Handle Finish/Reset button ---
    def handle_finish_reset(self): 
        self.update_status_bar_text("Resetting to idle state...")
        self.reset_ui_to_idle_state_internal()
        self.check_audio_device_status() 

# --- Reset UI to Idle State ---
    def reset_ui_to_idle_state_internal(self):
        self.status_label.setText("Status: Ready" if self.audio_controller.is_ready() else "Audio device not ready") 
        self.start_stop_button.setText("START")
        self.start_stop_button.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.start_stop_button.setEnabled(self.audio_controller.is_ready())
        self.save_as_button.setEnabled(False) 
        self.finish_reset_button.setEnabled(False) 
        self.data_stats_label.setText("Live: 0.00 MB @ 0.00 MB/s")
        if hasattr(self, 'recording_progress_bar'): self.recording_progress_bar.setValue(0)
        self.stacked_widget.setCurrentIndex(self.PAGE_IDLE)
        self.recorded_frames = None
        self.current_audio_filepath = None
        
        if hasattr(self, 'analysis_figure'):
            self.analysis_figure.clear()
            self.analysis_canvas.draw()

# --- Close Event ---
    def closeEvent(self, event): 
        self.update_status_bar_text("Exiting application...")
        QApplication.processEvents() 
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(1000) 
        if self.audio_controller:
            self.audio_controller.close()
        print("Application closed.")
        super().closeEvent(event)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update the layout of the two waveforms and the result box
        self.update_overlay_positions()
        self.analysis_canvas_1.updateGeometry()
        self.analysis_canvas_2.updateGeometry()


    def update_overlay_positions(self):
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


    # Position the result label (centered between the two waveforms)
    def _position_result_label(self, canvas_pos_1, canvas_pos_2, canvas_width_1, canvas_width_2, canvas_height_1, canvas_height_2):
        if hasattr(self, 'result_label') and self.result_label.isVisible():
            self.result_label.adjustSize()

            # Calculate the position in the middle between the two waveforms
            center_x = (canvas_pos_1.x() + canvas_pos_2.x() + canvas_width_1 + canvas_width_2) // 2
            center_y = max(canvas_pos_1.y(), canvas_pos_2.y()) + min(canvas_height_1, canvas_height_2) // 2

            self.result_label.move(
                center_x - self.result_label.width() // 2,
                center_y - self.result_label.height() // 2
            )
            self.result_label.raise_()


    # Position the exit icon (top-right)
    def _position_exit_icon(self):
        if hasattr(self, 'exit_icon_button'):
            self.exit_icon_button.move(
                self.width() - self.exit_icon_button.width() - 10,
                10
            )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
      