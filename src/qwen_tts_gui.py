import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
import threading
import os

SAMPLE_RATE = 44100
CHANNELS = 1

class QwenTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3 TTS Voice Training & Generation")
        self.root.geometry("800x1000")
        
        # Variables
        self.device_type = tk.StringVar(value="cuda")
        self.recording = False
        self.audio_chunks = []
        self.model = None
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Train Voice Tab
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Train Voice")
        self.setup_train_tab()
        
        # Use Voice Tab
        self.use_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.use_frame, text="Use Voice")
        self.setup_use_tab()
        
    def setup_train_tab(self):
        # Title
        title_label = ttk.Label(self.train_frame, text="Train a New Voice", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Voice name input
        name_frame = ttk.Frame(self.train_frame)
        name_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(name_frame, text="Voice Name:").pack(side=tk.LEFT, padx=5)
        self.voice_name_entry = ttk.Entry(name_frame, width=30)
        self.voice_name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Save location for trained voice
        save_frame = ttk.Frame(self.train_frame)
        save_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(save_frame, text="Save Location:").pack(side=tk.LEFT, padx=5)
        self.train_save_entry = ttk.Entry(save_frame, width=30)
        self.train_save_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(save_frame, text="Browse", command=self.browse_save_location_train).pack(side=tk.LEFT, padx=5)
        
        # Device selection
        device_frame = ttk.LabelFrame(self.train_frame, text="Device Selection", padding=10)
        device_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Radiobutton(device_frame, text="CUDA (GPU)", variable=self.device_type, 
                       value="cuda").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_type, 
                       value="cpu").pack(side=tk.LEFT, padx=10)
        
        # Training method selection
        method_frame = ttk.LabelFrame(self.train_frame, text="Training Method", padding=10)
        method_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.train_method = tk.StringVar(value="file")
        ttk.Radiobutton(method_frame, text="From Audio File + Transcript", 
                       variable=self.train_method, value="file").pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(method_frame, text="Record Audio (Pre-made Script)", 
                       variable=self.train_method, value="record").pack(anchor=tk.W, pady=5)
        
        # File input section
        file_frame = ttk.LabelFrame(self.train_frame, text="File Input", padding=10)
        file_frame.pack(fill=tk.X, padx=20, pady=10)
        
        audio_file_frame = ttk.Frame(file_frame)
        audio_file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(audio_file_frame, text="Audio File:").pack(side=tk.LEFT, padx=5)
        self.audio_file_entry = ttk.Entry(audio_file_frame, width=40)
        self.audio_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(audio_file_frame, text="Browse", command=self.browse_audio_file).pack(side=tk.LEFT, padx=5)
        
        transcript_frame = ttk.Frame(file_frame)
        transcript_frame.pack(fill=tk.X, pady=5)
        ttk.Label(transcript_frame, text="Transcript:").pack(anchor=tk.W, padx=5)
        self.transcript_entry = scrolledtext.ScrolledText(transcript_frame, height=4, width=50)
        self.transcript_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recording section
        record_frame = ttk.LabelFrame(self.train_frame, text="Recording", padding=10)
        record_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.script_text = "On a warm Saturday morning, the quick brown fox jumped over several lazy dogs while distant musicians played jazzy tunes near the quiet park. People checked their phones, argued about numbers, dates, and prices, and casually mentioned names like Alex, Jordan, and Taylor. A cyclist shouted warnings, a train horn echoed, and someone asked, ‘Why does this even matter?’ as rain began falling lightly at exactly 9:47 a.m., changing plans, moods, and expectations all at once."
        ttk.Label(record_frame, text="Script to read:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        script_display = scrolledtext.ScrolledText(record_frame, height=7, width=50, wrap=tk.WORD)
        script_display.insert("1.0", self.script_text)
        script_display.config(state=tk.DISABLED)
        script_display.pack(fill=tk.X, pady=5)
        
        self.record_status_label = ttk.Label(record_frame, text="Ready to record", foreground="green")
        self.record_status_label.pack(pady=5)
        
        self.record_button = ttk.Button(record_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=5)
        
        # Status and progress
        self.train_status_label = ttk.Label(self.train_frame, text="", foreground="blue")
        self.train_status_label.pack(pady=10)
        
        # Train button
        train_button = ttk.Button(self.train_frame, text="Train Voice", command=self.train_voice, 
                                 style="Accent.TButton")
        train_button.pack(pady=20)
        
    def setup_use_tab(self):
        # Title
        title_label = ttk.Label(self.use_frame, text="Generate Speech from a pretrained Voice", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Voice selection
        voice_frame = ttk.LabelFrame(self.use_frame, text="Voice Selection", padding=10)
        voice_frame.pack(fill=tk.X, padx=20, pady=10)
        
        voice_select_frame = ttk.Frame(voice_frame)
        voice_select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(voice_select_frame, text="Voice File:").pack(side=tk.LEFT, padx=5)
        self.voice_file_entry = ttk.Entry(voice_select_frame, width=40)
        self.voice_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(voice_select_frame, text="Browse", command=self.browse_voice_file).pack(side=tk.LEFT, padx=5)
        
        # Device selection
        device_frame = ttk.LabelFrame(self.use_frame, text="Device Selection", padding=10)
        device_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.use_device_type = tk.StringVar(value="cuda")
        ttk.Radiobutton(device_frame, text="CUDA (GPU)", variable=self.use_device_type, 
                       value="cuda").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.use_device_type, 
                       value="cpu").pack(side=tk.LEFT, padx=10)
        
        # Text input
        text_frame = ttk.LabelFrame(self.use_frame, text="Text to Generate", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.text_entry = scrolledtext.ScrolledText(text_frame, height=8, width=50)
        self.text_entry.pack(fill=tk.BOTH, expand=True)
        
        # Save location for output
        save_frame = ttk.Frame(self.use_frame)
        save_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(save_frame, text="Save Location:").pack(side=tk.LEFT, padx=5)
        self.use_save_entry = ttk.Entry(save_frame, width=40)
        self.use_save_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(save_frame, text="Browse", command=self.browse_save_location_use).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.use_status_label = ttk.Label(self.use_frame, text="", foreground="blue")
        self.use_status_label.pack(pady=10)
        
        # Generate button
        generate_button = ttk.Button(self.use_frame, text="Generate Speech", command=self.generate_speech,
                                    style="Accent.TButton")
        generate_button.pack(pady=20)
        
    def browse_audio_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.audio_file_entry.delete(0, tk.END)
            self.audio_file_entry.insert(0, filename)
    
    def browse_voice_file(self):
        filename = filedialog.askopenfilename(
            title="Select Voice File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.voice_file_entry.delete(0, tk.END)
            self.voice_file_entry.insert(0, filename)
    
    def browse_save_location_train(self):
        """Browse for save folder when training a voice"""
        folder = filedialog.askdirectory(
            title="Select Folder to Save Trained Voice"
        )
        if folder:
            self.train_save_entry.delete(0, tk.END)
            self.train_save_entry.insert(0, folder)
    
    def browse_save_location_use(self):
        """Browse for save folder when generating speech"""
        folder = filedialog.askdirectory(
            title="Select Folder to Save Generated Speech"
        )
        if folder:
            self.use_save_entry.delete(0, tk.END)
            self.use_save_entry.insert(0, folder)
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.recording = True
        self.audio_chunks = []
        self.record_status_label.config(text="Recording... Press Stop to finish", foreground="red")
        self.record_button.config(text="Stop Recording")
        
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_chunks.append(indata.copy())
        
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback)
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.audio_chunks:
            audio_data = np.concatenate(self.audio_chunks, axis=0)
            sf.write("mic.wav", audio_data, SAMPLE_RATE)
            self.record_status_label.config(text="Recording saved to mic.wav", foreground="green")
        else:
            self.record_status_label.config(text="No audio recorded", foreground="orange")
        
        self.record_button.config(text="Start Recording")
    
    def get_model_config(self, device_type, show_warning=True):
        """Get model configuration based on device type"""
        # Check if CUDA is requested but not available
        if device_type == "cuda" and not torch.cuda.is_available():
            if show_warning:
                self.root.after(0, lambda: messagebox.showwarning(
                    "CUDA Not Available", 
                    "CUDA is not available. Falling back to CPU mode."))
            device_type = "cpu"
        
        # Return CUDA config if CUDA is available and requested
        if device_type == "cuda" and torch.cuda.is_available():
            # Try flash_attention_2, fallback to sdpa if not available
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
            
            return {
                "device_map": "auto",
                "dtype": torch.bfloat16,
                "attn_implementation": attn_impl,
            }
        else:
            # CPU configuration
            return {
                "device_map": "cpu",
                "dtype": torch.float32,
                "attn_implementation": "eager",
            }
    
    def train_voice(self):
        voice_name = self.voice_name_entry.get().strip()
        if not voice_name:
            messagebox.showerror("Error", "Please enter a voice name")
            return
        
        method = self.train_method.get()
        device_type = self.device_type.get()
        
        # Run training in a separate thread to avoid blocking UI
        thread = threading.Thread(target=self._train_voice_thread, args=(voice_name, method, device_type))
        thread.daemon = True
        thread.start()
    
    def _train_voice_thread(self, voice_name, method, device_type):
        try:
            self.root.after(0, lambda: self.train_status_label.config(
                text="Loading model...", foreground="blue"))
            
            # Load model
            config = self.get_model_config(device_type, show_warning=True)
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                **config
            )
            
            ref_audio = None
            ref_text = None
            
            if method == "file":
                audio_file = self.audio_file_entry.get().strip()
                ref_text = self.transcript_entry.get("1.0", tk.END).strip()
                
                if not audio_file:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Please select an audio file"))
                    return
                
                if not ref_text:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a transcript"))
                    return
                
                if not os.path.exists(audio_file):
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Audio file not found: {audio_file}"))
                    return
                
                ref_audio = audio_file
                
            else:  # recording
                if not os.path.exists("mic.wav"):
                    self.root.after(0, lambda: messagebox.showerror("Error", 
                        "Please record audio first using the recording section"))
                    return
                
                ref_audio = "mic.wav"
                ref_text = self.script_text
            
            self.root.after(0, lambda: self.train_status_label.config(
                text="Creating voice clone prompt...", foreground="blue"))
            
            # Generate voice clone prompt
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
            
            # Determine save location
            save_folder = self.train_save_entry.get().strip()
            if save_folder:
                # Use user-specified folder
                output_file = os.path.join(save_folder, f"{voice_name}.pt")
            else:
                # Default to current directory with voice name
                output_file = f"{voice_name}.pt"
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the prompt
            torch.save(prompt_items, output_file)
            
            self.root.after(0, lambda: self.train_status_label.config(
                text=f"Voice '{voice_name}' trained successfully! Saved to {output_file}", 
                foreground="green"))
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Voice '{voice_name}' has been trained and saved to {output_file}"))
            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            self.root.after(0, lambda: self.train_status_label.config(
                text=error_msg, foreground="red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
    
    def generate_speech(self):
        voice_file = self.voice_file_entry.get().strip()
        text = self.text_entry.get("1.0", tk.END).strip()
        
        if not voice_file:
            messagebox.showerror("Error", "Please select a voice file")
            return
        
        if not text:
            messagebox.showerror("Error", "Please enter text to generate")
            return
        
        if not os.path.exists(voice_file):
            messagebox.showerror("Error", f"Voice file not found: {voice_file}")
            return
        
        # Run generation in a separate thread
        thread = threading.Thread(target=self._generate_speech_thread, args=(voice_file, text))
        thread.daemon = True
        thread.start()
    
    def _generate_speech_thread(self, voice_file, text):
        try:
            self.root.after(0, lambda: self.use_status_label.config(
                text="Loading model...", foreground="blue"))
            
            device_type = self.use_device_type.get()
            config = self.get_model_config(device_type, show_warning=True)
            
            # Load model
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                **config
            )
            
            self.root.after(0, lambda: self.use_status_label.config(
                text="Loading voice...", foreground="blue"))
            
            # Allowlist the class
            torch.serialization.add_safe_globals([VoiceClonePromptItem])
            
            # Load the saved voice clone prompt with proper device mapping
            # Use map_location to handle CPU/GPU properly
            map_location = "cpu" if device_type == "cpu" or not torch.cuda.is_available() else None
            
            # Check if file exists and is readable
            if not os.path.exists(voice_file):
                raise FileNotFoundError(f"Voice file not found: {voice_file}")
            
            if not voice_file.lower().endswith('.pt'):
                raise ValueError(f"Invalid file type. Expected .pt file, got: {voice_file}")
            
            try:
                # Try loading with weights_only parameter (PyTorch 2.0+)
                try:
                    if map_location:
                        prompt_items = torch.load(voice_file, map_location=map_location, weights_only=False)
                    else:
                        prompt_items = torch.load(voice_file, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions that don't support weights_only
                    if map_location:
                        prompt_items = torch.load(voice_file, map_location=map_location)
                    else:
                        prompt_items = torch.load(voice_file)
            except Exception as load_error:
                # Try loading on CPU as final fallback
                try:
                    prompt_items = torch.load(voice_file, map_location="cpu")
                except Exception as e:
                    error_details = str(e)
                    if "pickle" in error_details.lower() or "unpickling" in error_details.lower():
                        raise Exception(f"Failed to load voice file. The file may be corrupted or incompatible. Error: {error_details}")
                    else:
                        raise Exception(f"Failed to load voice file '{voice_file}': {error_details}")
            
            self.root.after(0, lambda: self.use_status_label.config(
                text="Generating speech...", foreground="blue"))
            
            # Generate speech
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="English",
                voice_clone_prompt=prompt_items,
            )
            
            # Determine save location
            save_folder = self.use_save_entry.get().strip()
            voice_name = os.path.splitext(os.path.basename(voice_file))[0]
            
            if save_folder:
                # Use user-specified folder
                output_file = os.path.join(save_folder, f"{voice_name}_output.wav")
            else:
                # Default to current directory with voice name
                output_file = f"{voice_name}_output.wav"
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save output
            sf.write(output_file, wavs[0], sr)
            
            self.root.after(0, lambda: self.use_status_label.config(
                text=f"Speech generated successfully! Saved to {output_file}", 
                foreground="green"))
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Speech has been generated and saved to {output_file}"))
            
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.root.after(0, lambda: self.use_status_label.config(
                text=error_msg, foreground="red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

def main():
    root = tk.Tk()
    app = QwenTTSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
