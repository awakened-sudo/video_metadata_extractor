import streamlit as st
import cv2
import tempfile
import json
import numpy as np
import pandas as pd
from openai import OpenAI, LengthFinishReasonError
from pathlib import Path
import time
from datetime import datetime, timedelta
import base64
from moviepy.editor import VideoFileClip
import os
from dotenv import load_dotenv
import plotly.express as px
from io import BytesIO
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid
from fpdf import FPDF
import matplotlib.pyplot as plt
import logging
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

class EventData(BaseModel):
    eventID: str
    eventImageURL: str = ""
    inpoint: float
    outpoint: float

class CaptionTrack(BaseModel):
    eventData: List[EventData]

class SubtitleEntry(BaseModel):
    inpoint: str
    outpoint: str
    text: str

class Tracks(BaseModel):
    caption: CaptionTrack

class SourceData(BaseModel):
    description: str = ""
    title: Optional[str] = None
    file_id: int = Field(default_factory=lambda: int(time.time()))
    lls_kv_id: int = Field(default_factory=lambda: int(str(int(time.time()))[-8:]))
    thumbnail: str = Field(default_factory=lambda: f"{str(uuid.uuid4())[:8]}.png")
    clip_name: str = Field(default_factory=lambda: f"FIN-{str(int(time.time()))[-2:]}")
    clip_title: str = Field(default_factory=lambda: str(int(time.time())))
    duration: str = "00:00:00:00"
    proxy_uri: str = ""
    relative_path: str = "//"
    tracks: Dict[str, CaptionTrack] = Field(default_factory=dict, alias="_tracks")
    subtitles: Dict[str, List[SubtitleEntry]] = Field(default_factory=dict, alias="_subtitles")

class VideoMetadata(BaseModel):
    index: str = Field(default_factory=lambda: str(int(time.time())))
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    seq_no: int = Field(default_factory=lambda: int(time.time()))
    primary_term: int = 1
    found: bool = True
    source: SourceData

class FrameAnalysis(BaseModel):
    description: str
    objects_detected: List[str]
    scene_type: str

class QueryResponse(BaseModel):
    answer: str
    relevant_timestamps: List[str]
    confidence: float

class ReportGenerator:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        
    def add_title(self, title):
        """Add a title to the PDF with proper styling"""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 24)
        self.pdf.set_text_color(31, 61, 143)  # Dark blue color
        self.pdf.cell(0, 20, title, ln=True, align='C')
        self.pdf.ln(10)  # Add some spacing after title
        
    def add_section_header(self, text):
        """Add a section header with styling"""
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.set_text_color(0, 0, 0)  # Black color
        self.pdf.cell(0, 10, text, ln=True)
        self.pdf.ln(5)
        
    def add_text(self, text):
        """Add normal text with proper styling"""
        self.pdf.set_font('Arial', '', 12)
        self.pdf.set_text_color(51, 51, 51)  # Dark gray for better readability
        self.pdf.multi_cell(0, 8, text)
        self.pdf.ln(5)
        
    def add_plot(self, fig, caption=""):
        """Add a plot with improved readability"""
        # Configure plot styling before saving
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300
        })
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout(pad=2.0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            # Save with high quality settings
            fig.savefig(tmpfile.name, 
                       format="png",
                       bbox_inches="tight",
                       dpi=300,
                       pad_inches=0.5)
            
            # Add to PDF with proper sizing
            self.pdf.add_page()
            self.pdf.image(tmpfile.name, x=10, w=190)
            
            if caption:
                self.pdf.set_font('Arial', 'I', 10)
                self.pdf.set_text_color(102, 102, 102)
                self.pdf.cell(0, 10, caption, ln=True, align='C')
            self.pdf.ln(5)

    def add_dataframe(self, df, max_rows=30):
        self.pdf.set_font('Arial', 'B', 12)
        # Add headers
        for idx, col in enumerate(df.columns):
            self.pdf.cell(190/len(df.columns), 10, str(col), 1)
        self.pdf.ln()
        
        # Add rows
        self.pdf.set_font('Arial', '', 10)
        for i in range(min(len(df), max_rows)):
            for col in df.columns:
                self.pdf.cell(190/len(df.columns), 10, str(df.iloc[i][col]), 1)
            self.pdf.ln()
            
    def get_pdf_download_link(self):
        """Generate download link for PDF"""
        pdf_data = self.pdf.output(dest="S").encode("latin-1")
        b64 = base64.b64encode(pdf_data)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="report.pdf">Download PDF Report</a>'

class ProcessingVisualizer:
    def __init__(self):
        # Create layout containers
        self.create_layout()
        
# Initialize state
        self.current_character_index = 0
        self.animation_speed = 0.03  # Seconds per character

        # Add new containers for specific visualizations
        with self.main_container:
            self.audio_col, self.translation_col = st.columns(2)
            with self.audio_col:
                self.audio_spectrum = st.empty()
                self.audio_waveform = st.empty()
            with self.translation_col:
                self.translation_progress = st.empty()
                self.translation_output = st.empty()
            
            self.metadata_col, self.analytics_col = st.columns(2)
            with self.metadata_col:
                self.metadata_progress = st.empty()
                self.metadata_display = st.empty()
            with self.analytics_col:
                self.analytics_chart = st.empty()
                self.analytics_metrics = st.empty()

    def create_layout(self):
        """Create the main layout for the processing visualization"""
        st.markdown("""
        <style>
            .processing-window {
                border: 2px solid #1e3d8f;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                background-color: #f8f9fa;
            }
            .log-window {
                height: 200px;
                overflow-y: auto;
                font-family: monospace;
                background-color: #1e1e1e;
                color: #00ff00;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
            .prompt-window {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                font-family: monospace;
            }
            .output-window {
                background-color: #f0f2f6;
                border-left: 4px solid #1e3d8f;
                padding: 15px;
                margin: 10px 0;
                font-family: monospace;
            }
            .status-badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 0.8em;
                margin-right: 10px;
            }
            .status-running {
                background-color: #ffd700;
                color: #000000;
            }
            .status-completed {
                background-color: #00ff00;
                color: #000000;
            }
            .status-error {
                background-color: #ff0000;
                color: #ffffff;
            }
            .processing-header {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .animated-text {
                overflow: hidden;
                white-space: pre-wrap;
                animation: typing 3s steps(40, end);
            }
            @keyframes typing {
                from { width: 0 }
                to { width: 100% }
            }
        </style>
        """, unsafe_allow_html=True)
        
        self.main_container = st.container()
        with self.main_container:
            col1, col2 = st.columns([2, 1])
            with col1:
                self.log_container = st.empty()
                self.prompt_container = st.empty()
                self.output_container = st.empty()
            with col2:
                self.image_container = st.empty()
                self.progress_container = st.empty()
                self.status_container = st.empty()

    def update_status(self, status, message):
        """Update the status badge and message"""
        status_classes = {
            'running': 'status-running',
            'completed': 'status-completed',
            'error': 'status-error'
        }
        self.status_container.markdown(f"""
        <div class="processing-header">
            <span class="status-badge {status_classes.get(status, 'status-running')}">
                {status.upper()}
            </span>
            {message}
        </div>
        """, unsafe_allow_html=True)

    def show_frame(self, frame):
        """Display the current frame being processed"""
        self.image_container.image(frame, use_container_width=True)

    def animate_text(self, text, container, text_type="log"):
        """Animate text appearing letter by letter"""
        placeholder = container.empty()
        displayed_text = ""
        
        for char in text:
            displayed_text += char
            if text_type == "log":
                placeholder.markdown(f"""
                <div class="log-window">{displayed_text}</div>
                """, unsafe_allow_html=True)
            elif text_type == "prompt":
                placeholder.markdown(f"""
                <div class="prompt-window">{displayed_text}</div>
                """, unsafe_allow_html=True)
            elif text_type == "output":
                placeholder.markdown(f"""
                <div class="output-window">{displayed_text}</div>
                """, unsafe_allow_html=True)
            time.sleep(self.animation_speed)

    def update_progress(self, progress, total, task_name):
        """Update the progress bar with current progress"""
        progress_pct = progress / total
        self.progress_container.progress(progress_pct)
        self.status_container.markdown(f"Processing {task_name}: {progress}/{total}")

    def log_processing_step(self, step_name, details, status="running"):
        """Log a processing step with details"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {step_name}: {details}"
        self.animate_text(log_message, self.log_container, "log")
        self.update_status(status, step_name)

    def show_prompt(self, prompt_text):
        """Display the prompt being sent to the model"""
        self.animate_text(f"Prompt:\n{prompt_text}", self.prompt_container, "prompt")

    def show_output(self, output_text):
        """Display the model's output"""
        self.animate_text(f"Output:\n{output_text}", self.output_container, "output")

    def clear_all(self):
        """Clear all containers"""
        self.log_container.empty()
        self.prompt_container.empty()
        self.output_container.empty()
        self.image_container.empty()
        self.progress_container.empty()
        self.status_container.empty()

    def visualize_audio_processing(self, frame_data, spectrum_data):
        """Visualize audio processing with spectrum and waveform"""
        try:
            # Create waveform visualization
            if isinstance(frame_data, pd.DataFrame) and not frame_data.empty:
                fig_waveform = px.line(
                    frame_data,
                    x='time',
                    y='amplitude',
                    title="Audio Waveform"
                )
            else:
                # Convert to list if not already
                if isinstance(frame_data, np.ndarray):
                    frame_data = frame_data.tolist()
                
                # Convert to DataFrame with proper sequence types
                df_waveform = pd.DataFrame({
                    'time': list(range(len(frame_data))),
                    'amplitude': frame_data
                })
                fig_waveform = px.line(
                    df_waveform,
                    x='time',
                    y='amplitude',
                    title="Audio Waveform"
                )
            
            self.audio_waveform.plotly_chart(fig_waveform, use_container_width=True)
            
            # Create spectrum visualization
            if isinstance(spectrum_data, pd.DataFrame) and not spectrum_data.empty:
                fig_spectrum = px.line(
                    spectrum_data,
                    x='frequency',
                    y='magnitude',
                    title="Audio Spectrum Analysis"
                )
            else:
                # Convert to list if not already
                if isinstance(spectrum_data, np.ndarray):
                    spectrum_data = spectrum_data.tolist()
                
                # Convert to DataFrame with proper sequence types
                df_spectrum = pd.DataFrame({
                    'frequency': list(range(len(spectrum_data))),
                    'magnitude': spectrum_data
                })
                fig_spectrum = px.line(
                    df_spectrum,
                    x='frequency',
                    y='magnitude',
                    title="Audio Spectrum Analysis"
                )
            
            self.audio_spectrum.plotly_chart(fig_spectrum, use_container_width=True)
            
            self.log_processing_step("Audio Processing", "Audio visualization completed")
            
        except Exception as e:
            logging.error(f"Audio visualization error: {str(e)}", exc_info=True)
            self.log_processing_step("Audio Visualization Error", 
                f"Error visualizing audio data: {str(e)}", 
                    status="error"
                )
            
    def visualize_translation(self, source_text, target_language, progress):
        """Visualize translation progress and results"""
        # Show translation progress
        self.translation_progress.progress(progress)
        
        # Display translation output
        self.translation_output.markdown(f"""
        **Source Text:** {source_text[:100]}...
        **Target Language:** {target_language}
        **Progress:** {progress*100:.1f}%
        """)
        
        self.log_processing_step("Translation", f"Translating to {target_language}")

    def visualize_metadata(self, metadata_dict):
        """Visualize metadata creation and structure"""
        # Show metadata structure
        self.metadata_display.json(metadata_dict)
        
        # Create tree visualization of metadata
        fig = self.create_metadata_tree_viz(metadata_dict)
        self.metadata_progress.plotly_chart(fig, use_container_width=True)
        
        self.log_processing_step("Metadata", "Generating metadata structure")
    
    def visualize_audio_processing(self, frame_data, spectrum_data):
        """Visualize audio processing with spectrum and waveform"""
        try:
            # Handle spectrum visualization
            if isinstance(spectrum_data, pd.DataFrame):
                fig_spectrum = px.line(
                    spectrum_data, 
                    x=spectrum_data.columns[0],
                    y=spectrum_data.columns[1],
                    title="Audio Spectrum Analysis"
                )
            else:
                # Convert array/list to DataFrame with proper sequence
                spectrum_array = np.asarray(spectrum_data)
                df_spectrum = pd.DataFrame({
                    'frequency': list(range(len(spectrum_array))),
                    'magnitude': list(spectrum_array)
                })
                fig_spectrum = px.line(
                    df_spectrum,
                    x='frequency',
                    y='magnitude',
                    title="Audio Spectrum Analysis"
                )
            
            self.audio_spectrum.plotly_chart(fig_spectrum, use_container_width=True)
            
            # Handle waveform visualization
            if isinstance(frame_data, pd.DataFrame):
                fig_waveform = px.line(
                    frame_data,
                    x=frame_data.columns[0],
                    y=frame_data.columns[1],
                    title="Audio Waveform"
                )
            else:
                # Convert array/list to DataFrame with proper sequence
                waveform_array = np.asarray(frame_data)
                df_waveform = pd.DataFrame({
                    'time': list(range(len(waveform_array))),
                    'amplitude': list(waveform_array)
                })
                fig_waveform = px.line(
                    df_waveform,
                    x='time',
                    y='amplitude',
                    title="Audio Waveform"
                )
            
            self.audio_waveform.plotly_chart(fig_waveform, use_container_width=True)
            
            self.log_processing_step("Audio Processing", "Visualizing audio data")
            
        except Exception as e:
            self.log_processing_step("Audio Visualization Error", 
                f"Error visualizing audio data: {str(e)}", 
                status="error"
            )

    def visualize_analytics(self, analytics_data):
        """Visualize analytics generation and metrics"""
        # Create analytics charts
        fig = px.bar(
            analytics_data,
            title="Content Analysis Results"
        )
        self.analytics_chart.plotly_chart(fig, use_container_width=True)
        
        # Display key metrics
        metrics = self.calculate_analytics_metrics(analytics_data)
        self.analytics_metrics.markdown(f"""
        ### Key Metrics
        - Total Duration: {metrics['duration']}
        - Scene Changes: {metrics['scene_changes']}
        - Confidence Score: {metrics['confidence']:.2f}
        """)
        
        self.log_processing_step("Analytics", "Generating content analytics")

    def create_metadata_tree_viz(self, metadata_dict):
        """Create a tree visualization of metadata structure"""
        # Convert metadata to tree structure for visualization
        fig = go.Figure(go.Treemap(
            labels=[str(k) for k in metadata_dict.keys()],
            parents=[''] * len(metadata_dict),
            values=[1] * len(metadata_dict)
        ))
        fig.update_layout(title="Metadata Structure")
        return fig

    def calculate_analytics_metrics(self, analytics_data):
        """Calculate key metrics from analytics data"""
        return {
            'duration': str(timedelta(seconds=int(analytics_data.get('duration', 0)))),
            'scene_changes': analytics_data.get('scene_changes', 0),
            'confidence': analytics_data.get('confidence', 0.0)
        }

class VideoProcessor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.supported_languages = {
            'en-US': 'English',
            'ar-AR': 'Arabic',
            'zh-CN': 'Mandarin',
            'ta-IN': 'Tamil',
            'ms-MY': 'Malay'
        }

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language using OpenAI"""
        try:
            logging.info(f"Starting translation to {target_language}")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate the following text to {target_language}. Maintain the original meaning and tone."
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "text"}
            )
            translated = response.choices[0].message.content.strip()
            logging.debug(f"Translation completed successfully: {text[:50]}... -> {translated[:50]}...")
            return translated
        except Exception as e:
            logging.error(f"Translation error: {str(e)}", exc_info=True)
            return text

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        try:
            logging.info("Starting language detection")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Detect the language of the following text and respond with the language code only"
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "text"}
            )
            detected = response.choices[0].message.content.strip()
            logging.debug(f"Language detected: {detected}")
            return detected
        except Exception as e:
            logging.error(f"Language detection error: {str(e)}", exc_info=True)
            return "en-US"

    def extract_frames(self, video_path, sample_rate=1):
        """Extract frames from video with enhanced metadata"""
        logging.info(f"Starting frame extraction from {video_path}")
        try:
            frames = []
            timestamps = []
            frame_images = []
            frame_numbers = []
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logging.debug(f"Video properties - FPS: {fps}, Total frames: {total_frames}")
            
            frame_interval = int(fps * sample_rate)
            duration = total_frames / fps if fps > 0 else 0
            
            video_metadata = {
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames,
                'frame_interval': frame_interval
            }
            
            if fps <= 0:
                fps = 30  # Default fallback
            
            current_frame = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame % frame_interval == 0:
                    timestamp = current_frame / fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Store frame data
                    frame_images.append(frame_rgb)
                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_frame = base64.b64encode(buffer).decode('utf-8')
                    frames.append(base64_frame)
                    timestamps.append(timestamp)
                    frame_numbers.append(current_frame)
                
                current_frame += 1
            
            cap.release()
            logging.info(f"Frame extraction completed. Extracted {len(frames)} frames")
            return frames, timestamps, frame_images, video_metadata, frame_numbers
        
        except Exception as e:
            logging.error(f"Frame extraction error: {str(e)}", exc_info=True)
            raise

    def analyze_frame(self, frame_base64: str, timestamp: float) -> str:
        """Analyze frame using OpenAI Vision"""
        try:
            logging.debug(f"Analyzing frame at timestamp {timestamp:.2f}")
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video frame analyzer. Describe the scene, detect objects, and categorize the scene type."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe this video frame at timestamp {timestamp:.2f} seconds."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_base64}"
                                }
                            }
                        ]
                    }
                ],
                response_format=FrameAnalysis,
                max_tokens=500
            )
            
            # Get the parsed response
            frame_analysis = completion.choices[0].message.parsed
            
            # Convert to JSON string to maintain compatibility with existing code
            logging.debug(f"Frame analysis completed for timestamp {timestamp:.2f}")
            return json.dumps({
                "description": frame_analysis.description,
                "objects_detected": frame_analysis.objects_detected,
                "scene_type": frame_analysis.scene_type
            })
            
        except Exception as e:
            logging.error(f"Frame analysis error at timestamp {timestamp:.2f}: {str(e)}", exc_info=True)
            if isinstance(e, LengthFinishReasonError):
                # Handle token limit error
                return json.dumps({
                    "description": "Error: Response exceeded token limit",
                    "objects_detected": [],
                    "scene_type": "error"
                })
            else:
                # Handle other errors
                return json.dumps({
                    "description": f"Error analyzing frame: {str(e)}",
                    "objects_detected": [],
                    "scene_type": "error"
                })

    def extract_audio_with_translations(self, video_path: str) -> dict:
        """Extract audio and generate translations with enhanced visualization"""
        video = None
        try:
            logging.info(f"Starting audio extraction from {video_path}")
            self.visualizer.log_processing_step("Audio Processing", "Initializing audio extraction...")
            
            # Extract audio from video
            video = VideoFileClip(video_path)
            if video.audio is None:
                return {"error": "No audio found in video"}
                
            audio_duration = video.duration
            
            try:
                # Convert audio to numpy array with explicit list conversion
                audio_array = list(video.audio.iter_frames())  # Use iter_frames() instead of to_soundarray()
                audio_array = np.array(audio_array)
                
                # Convert stereo to mono if needed
                if len(audio_array.shape) > 1:
                    waveform_data = [float(x) for x in np.mean(audio_array, axis=1)]
                else:
                    waveform_data = [float(x) for x in audio_array]
                
                # Calculate FFT with explicit list conversion
                fft_result = np.fft.rfft(waveform_data)
                fft_data = [float(x) for x in np.abs(fft_result)]
                freqs = [float(x) for x in np.fft.rfftfreq(len(waveform_data), 1/video.audio.fps)]
                
                # Create time points as explicit list
                time_points = [float(x) for x in np.linspace(0, audio_duration, len(waveform_data))]
                
                # Create DataFrames with guaranteed sequences
                waveform_df = pd.DataFrame({
                    'time': time_points,
                    'amplitude': waveform_data
                })
                
                spectrum_df = pd.DataFrame({
                    'frequency': freqs,
                    'magnitude': fft_data
                })
                
                # Ensure DataFrames are not empty
                if waveform_df.empty or spectrum_df.empty:
                    raise ValueError("Generated DataFrames are empty")
                
                # Visualize audio data
                self.visualizer.visualize_audio_processing(waveform_df, spectrum_df)
                
            except Exception as audio_process_error:
                logging.error(f"Error processing audio data: {str(audio_process_error)}")
                self.visualizer.log_processing_step("Audio Processing Error", 
                    f"Error processing audio data: {str(audio_process_error)}", 
                    status="error")
                return {"error": f"Audio processing error: {str(audio_process_error)}"}
            
            # Process audio in chunks
            chunk_duration = 300
            chunk_overlap = 10
            overlap_threshold = 0.5
            num_chunks = int(np.ceil(audio_duration / (chunk_duration - chunk_overlap)))
            
            with tempfile.TemporaryDirectory() as temp_dir:
                chunk_segments = []
                
                for i in range(num_chunks):
                    try:
                        start_time = i * (chunk_duration - chunk_overlap)
                        end_time = min(start_time + chunk_duration, audio_duration)
                        
                        self.visualizer.update_progress(i + 1, num_chunks, "Audio Chunk Processing")
                        self.visualizer.log_processing_step("Audio Chunk", 
                            f"Processing chunk {i+1}/{num_chunks} ({start_time:.1f}s - {end_time:.1f}s)")
                        
                        chunk = video.subclip(start_time, end_time)
                        chunk_path = os.path.join(temp_dir, f'chunk_{i}.mp3')
                        chunk.audio.write_audiofile(chunk_path, verbose=False, logger=None)
                        
                        with open(chunk_path, 'rb') as audio_file:
                            response = self.client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                response_format="verbose_json",
                                timestamp_granularities=["segment"]
                            )
                            
                            for segment in response.segments:
                                chunk_segments.append({
                                    'start': segment.start + start_time,
                                    'end': segment.end + start_time,
                                    'text': segment.text
                                })
                        
                    except Exception as chunk_error:
                        logging.error(f"Error processing chunk {i}: {str(chunk_error)}")
                        self.visualizer.log_processing_step("Chunk Error", 
                            f"Error in chunk {i}: {str(chunk_error)}", 
                            status="error")
                        continue
                
                try:
                    if chunk_segments:
                        # Process transcription
                        return self.process_transcription(chunk_segments)
                    else:
                        return {"error": "No segments processed successfully"}
                        
                except Exception as process_error:
                    logging.error(f"Error processing transcription: {str(process_error)}")
                    return {"error": f"Transcription processing error: {str(process_error)}"}
                    
        except Exception as main_error:
            logging.error(f"Main audio extraction error: {str(main_error)}")
            self.visualizer.log_processing_step("Error", 
                f"Error in audio extraction: {str(main_error)}", 
                status="error")
            return {"error": str(main_error)}
            
        finally:
            # Clean up video object
            if video is not None:
                try:
                    video.close()
                except Exception as close_error:
                    logging.error(f"Error closing video: {str(close_error)}")
    def process_transcription(self, chunk_segments):
        """Process transcription and generate translations with visualization"""
        try:
            self.visualizer.log_processing_step("Translation", "Starting translation processing...")
            subtitles = {}
            
            # Sort and process segments
            chunk_segments.sort(key=lambda x: x['start'])
            full_text = " ".join(seg['text'] for seg in chunk_segments)
            
            # Detect source language
            self.visualizer.show_prompt("Detecting source language...")
            source_language = self.detect_language(full_text)
            self.visualizer.log_processing_step("Language Detection", 
                f"Detected source language: {source_language}",
                status="completed"
            )
            
            # Process source language subtitles
            subtitles[source_language] = []
            for segment in chunk_segments:
                subtitles[source_language].append({
                    "inpoint": str(timedelta(seconds=int(segment['start']))),
                    "outpoint": str(timedelta(seconds=int(segment['end']))),
                    "text": segment['text']
                })
            
            # Generate translations for each language
            total_languages = len(self.supported_languages)
            for idx, (lang_code, lang_name) in enumerate(self.supported_languages.items()):
                if lang_code != source_language:
                    self.visualizer.log_processing_step("Translation", 
                        f"Translating to {lang_name}...")
                    
                    subtitles[lang_code] = []
                    for i, segment in enumerate(chunk_segments):
                        # Visualize translation progress
                        progress = (idx * len(chunk_segments) + i) / (total_languages * len(chunk_segments))
                        self.visualizer.visualize_translation(
                            segment['text'],
                            lang_name,
                            progress
                        )
                        
                        translated_text = self.translate_text(
                            segment['text'],
                            lang_name
                        )
                        
                        subtitles[lang_code].append({
                            "inpoint": str(timedelta(seconds=int(segment['start']))),
                            "outpoint": str(timedelta(seconds=int(segment['end']))),
                            "text": translated_text
                        })
                        
                        self.visualizer.show_output(
                            f"Translation to {lang_name}:\n{translated_text}"
                        )
                    
                    self.visualizer.log_processing_step("Translation", 
                        f"Completed translation to {lang_name}",
                        status="completed"
                    )
            
            return subtitles
            
        except Exception as e:
            self.visualizer.log_processing_step("Error", 
                f"Translation error: {str(e)}", 
                status="error")
            return {}

class MetadataManager:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.metadata_df = None
        self.video_metadata = None
        self.structured_data = None
        
    def create_metadata_df(self, frame_metadata, timestamps, audio_data, video_metadata, frame_numbers):
        """Create structured metadata with enhanced visualization"""
        try:
            self.visualizer.log_processing_step("Metadata", "Starting metadata organization...")
            self.video_metadata = video_metadata
            
            # Parse frame metadata with visualization
            parsed_metadata = []
            for i, metadata_str in enumerate(frame_metadata):
                try:
                    metadata = json.loads(metadata_str)
                    parsed_metadata.append(metadata)
                    
                    # Visualize metadata parsing progress
                    progress = (i + 1) / len(frame_metadata)
                    self.visualizer.update_progress(i + 1, len(frame_metadata), "Metadata Parsing")
                    
                except json.JSONDecodeError:
                    self.visualizer.log_processing_step("Warning", 
                        f"Failed to parse metadata at index {i}",
                        status="error"
                    )
                    parsed_metadata.append({
                        "description": metadata_str,
                        "objects_detected": [],
                        "scene_type": "unknown"
                    })
            
            # Create and visualize DataFrame
            self.metadata_df = pd.DataFrame({
                'frame_number': frame_numbers,
                'timestamp': timestamps,
                'formatted_time': [str(timedelta(seconds=int(t))) for t in timestamps],
                'frame_description': [m['description'] for m in parsed_metadata],
                'objects_detected': [m['objects_detected'] for m in parsed_metadata],
                'scene_type': [m['scene_type'] for m in parsed_metadata]
            })
            
            # Create event data and visualize progress
            self.visualizer.log_processing_step("Events", "Creating event timeline...")
            event_data = []
            for idx, row in self.metadata_df.iterrows():
                event_data.append({
                    "eventID": row['frame_description'],
                    "eventImageURL": "",
                    "inpoint": float(row['timestamp']),
                    "outpoint": float(row['timestamp'])
                })
                
                # Visualize event creation progress
                if idx % 10 == 0:  # Update every 10 events
                    self.visualizer.update_progress(idx + 1, len(self.metadata_df), "Event Creation")
            
            # Create and visualize source data structure
            source_data = {
                "description": video_metadata.get('description', ''),
                "title": None,
                "file_id": int(time.time()),
                "duration": self.format_duration(video_metadata['duration']),
                "tracks": {"caption": {"eventData": event_data}},
                "subtitles": audio_data if isinstance(audio_data, dict) else {}
            }
            
            # Visualize metadata structure
            self.visualizer.visualize_metadata(source_data)
            
            # Create final metadata structure
            self.structured_data = {
                "index": str(int(time.time())),
                "id": str(uuid.uuid4()),
                "version": 1,
                "source": source_data
            }
            
            self.visualizer.log_processing_step("Metadata", 
                "Metadata organization completed",
                status="completed"
            )
            
        except Exception as e:
            self.visualizer.log_processing_step("Error", 
                f"Metadata error: {str(e)}",
                status="error"
            )
            raise

    def get_event_data(self):
        """Safely get event data from structured data"""
        try:
            return self.structured_data['source']['tracks']['caption']['eventData']
        except (KeyError, TypeError):
            return []

    def format_duration(self, seconds):
        """Format duration in HH:MM:SS:FF format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * 24)  # Assuming 24 fps
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

    def get_structured_output(self):
        """Return the complete structured output"""
        return self.structured_data

    def query_metadata(self, query):
        """Enhanced metadata querying with structured output support"""
        if self.metadata_df is None:
            return "No metadata available. Please process a video first."
        
        try:
            # Prepare context from structured data
            metadata_context = []
            for _, row in self.metadata_df.iterrows():
                metadata_context.append({
                    "timestamp": row['formatted_time'],
                    "description": row['frame_description'],
                    "objects": row['objects_detected'],
                    "scene_type": row['scene_type']
                })
            
            # Define the response schema using Pydantic
            class QueryResponse(BaseModel):
                answer: str
                relevant_timestamps: List[str]
                confidence: float
            
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are analyzing video content with structured metadata. Provide detailed answers based on the frame descriptions, detected objects, and scene types."
                    },
                    {
                        "role": "user",
                        "content": f"Based on this video metadata:\n{json.dumps(metadata_context, indent=2)}\n\nQuery: {query}"
                    }
                ],
                response_format=QueryResponse
            )
            
            result = response.choices[0].message.parsed
            return f"""Answer: {result.answer}\n\nRelevant Timestamps: {', '.join(result.relevant_timestamps)}\nConfidence: {result.confidence:.2f}"""
            
        except Exception as e:
            return f"Error querying metadata: {str(e)}"

    def export_metadata(self, format_type="json"):
        """Export metadata in various formats"""
        if format_type == "json":
            # Convert to JSON with proper UTF-8 encoding
            json_str = json.dumps(self.structured_data, 
                                ensure_ascii=False,  # Allow non-ASCII characters
                                indent=2)  # Keep pretty printing
            return json_str.encode('utf-8')  # Return UTF-8 encoded bytes
        elif format_type == "csv":
            return self.metadata_df.to_csv(index=False)
        elif format_type == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                self.metadata_df.to_excel(writer, index=False, sheet_name='Video Metadata')
                
                # Add structured data sheet
                pd.DataFrame([self.structured_data]).to_excel(
                    writer, 
                    index=False, 
                    sheet_name='Structured Output'
                )
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def get_subtitle_languages(self):
        """Get list of available subtitle languages"""
        if self.structured_data and 'source' in self.structured_data:
            subtitles = self.structured_data['source'].get('subtitles', {})
            return list(subtitles.keys())
        return []

    def get_subtitles_for_language(self, language_code):
        """Get subtitles for specific language"""
        if self.structured_data and 'source' in self.structured_data:
            subtitles = self.structured_data['source'].get('subtitles', {})
            return subtitles.get(language_code, [])
        return []
    
def format_timestamp(seconds):
    """Format seconds into HH:MM:SS"""
    return str(timedelta(seconds=int(float(seconds)))).split('.')[0]


def create_video_player(video_file, metadata_manager):
    """Enhanced video player with subtitle support"""
    video_col, timeline_col = st.columns([1, 1])
    
    with video_col:
        video_container = st.empty()
        st.markdown("""
            <style>
            .video-container video {
                max-width: 100%;
                max-height: 400px;
                width: auto;
                margin: 0 auto;
                display: block;
            }
            </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            video_container.video(video_file)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Subtitle language selector
        available_languages = metadata_manager.get_subtitle_languages()
        if available_languages:
            selected_language = st.selectbox(
                "Select Subtitle Language",
                available_languages,
                format_func=lambda x: {
                    'en-US': 'English',
                    'ar-AR': 'Arabic',
                    'zh-CN': 'Mandarin',
                    'ta-IN': 'Tamil',
                    'ms-MY': 'Malay'
                }.get(x, x)
            )
            
            subtitles = metadata_manager.get_subtitles_for_language(selected_language)
            if subtitles:
                st.markdown("### Subtitles")
                # Check if subtitles is a string or list
                if isinstance(subtitles, str):
                    st.markdown(subtitles)
                else:
                    for subtitle in subtitles:
                        if isinstance(subtitle, dict) and 'inpoint' in subtitle and 'outpoint' in subtitle and 'text' in subtitle:
                            st.markdown(f"""
                                <div class='subtitle-entry'>
                                    <small>{subtitle.get('inpoint', '')} - {subtitle.get('outpoint', '')}</small><br/>
                                    {subtitle.get('text', '')}
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Invalid subtitle format")
    
    with timeline_col:
        st.markdown("### 🎬 Timeline")
        with st.container():
            # Get structured events from metadata
            events = metadata_manager.get_event_data()
            
            if events:
                for event in events:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button(f"⏱️ {format_timestamp(event['inpoint'])}", 
                                       key=f"ts_{event['inpoint']}"):
                                video_container.video(video_file, start_time=int(event['inpoint']))
                        with col2:
                            st.markdown(f"<small>{event['eventID'][:100]}...</small>", 
                                      unsafe_allow_html=True)
                        st.markdown("<hr style='margin: 4px 0'>", unsafe_allow_html=True)
            else:
                st.info("No timeline events available.")

def create_metadata_viewer(metadata_manager):
    """Enhanced metadata viewer with structured data"""
    tabs = st.tabs(["Timeline", "Structured Data", "Subtitles"])
    
    with tabs[0]:
        if metadata_manager.metadata_df is not None:
            st.dataframe(
                metadata_manager.metadata_df[[
                    'frame_number',
                    'formatted_time', 
                    'frame_description',
                    'scene_type'
                ]],
                use_container_width=True
            )
        else:
            st.info("No timeline data available")
    
    with tabs[1]:
        if metadata_manager.structured_data:
            st.json(metadata_manager.structured_data)
        else:
            st.info("No structured data available")
    
    with tabs[2]:
        languages = metadata_manager.get_subtitle_languages()
        if languages:
            selected_language = st.selectbox(
                "Select Language",
                languages,
                key="subtitle_viewer"
            )
            
            subtitles = metadata_manager.get_subtitles_for_language(selected_language)
            if subtitles:
                try:
                    # Convert subtitles to a list of dictionaries if it isn't already
                    if isinstance(subtitles, str):
                        st.text(subtitles)
                    else:
                        # Ensure we have a list of dictionaries with consistent keys
                        subtitle_data = []
                        for sub in subtitles:
                            if isinstance(sub, dict):
                                subtitle_data.append({
                                    'inpoint': sub.get('inpoint', ''),
                                    'outpoint': sub.get('outpoint', ''),
                                    'text': sub.get('text', '')
                                })
                        
                        if subtitle_data:
                            df = pd.DataFrame(subtitle_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No valid subtitle data found")
                            
                except Exception as e:
                    st.error(f"Error displaying subtitles: {str(e)}")
            else:
                st.info("No subtitles available for this language")
        else:
            st.info("No subtitle languages available")

def create_analytics_view(metadata_manager):
        """Enhanced analytics view with real-time visualization"""
        visualizer = ProcessingVisualizer()
        
        try:
            visualizer.log_processing_step("Analytics", "Generating video analytics...")
            
            # Calculate and prepare basic metrics as DataFrame
            metrics_data = pd.DataFrame({
                'Metric': ['Duration', 'Total Scenes', 'Available Languages', 'Total Objects'],
                'Value': [
                    metadata_manager.structured_data['source']['duration'],
                    len(metadata_manager.structured_data['source']['tracks']['caption']['eventData']),
                    len(metadata_manager.get_subtitle_languages()),
                    sum(len(objects) for objects in metadata_manager.metadata_df['objects_detected'])
                ]
            })
            
            # Create metrics bar chart
            metrics_fig = px.bar(
                metrics_data,
                x='Metric',
                y='Value',
                title="Content Analysis Overview",
                labels={'Value': 'Count', 'Metric': 'Metric Type'}
            )
            st.plotly_chart(metrics_fig, use_container_width=True)
            
            # Generate and visualize scene analysis
            visualizer.log_processing_step("Analytics", "Analyzing scene distribution...")
            scene_types = metadata_manager.metadata_df['scene_type'].value_counts().reset_index()
            scene_types.columns = ['Scene Type', 'Count']
            
            scene_fig = px.pie(
                scene_types,
                values='Count',
                names='Scene Type',
                title="Scene Type Distribution"
            )
            st.plotly_chart(scene_fig, use_container_width=True)
            
            # Generate and visualize object timeline
            visualizer.log_processing_step("Analytics", "Creating object detection timeline...")
            object_timeline = []
            for idx, row in metadata_manager.metadata_df.iterrows():
                for obj in row['objects_detected']:
                    object_timeline.append({
                        'timestamp': row['formatted_time'],
                        'object': obj
                    })
            
            if object_timeline:
                df_timeline = pd.DataFrame(object_timeline)
                timeline_fig = px.scatter(
                    df_timeline,
                    x='timestamp',
                    y='object',
                    title="Object Appearances Over Time",
                    color='object'
                )
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Scene complexity analysis
            scene_complexity = pd.DataFrame({
                'Time': metadata_manager.metadata_df['formatted_time'],
                'Complexity': metadata_manager.metadata_df.apply(
                    lambda x: len(x['objects_detected']),
                    axis=1
                )
            })
            
            complexity_fig = px.line(
                scene_complexity,
                x='Time',
                y='Complexity',
                title="Scene Complexity Over Time"
            )
            st.plotly_chart(complexity_fig, use_container_width=True)
            
            # Display summary metrics
            st.markdown("### 📊 Summary Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            metrics_display = {
                'Duration': metadata_manager.structured_data['source']['duration'],
                'Total Scenes': len(metadata_manager.structured_data['source']['tracks']['caption']['eventData']),
                'Languages': len(metadata_manager.get_subtitle_languages()),
                'Objects Detected': sum(len(objects) for objects in metadata_manager.metadata_df['objects_detected'])
            }
            
            with col1:
                st.metric("Total Duration", metrics_display['Duration'])
            with col2:
                st.metric("Total Scenes", metrics_display['Total Scenes'])
            with col3:
                st.metric("Languages", metrics_display['Languages'])
            with col4:
                st.metric("Objects Detected", metrics_display['Objects Detected'])
                
            visualizer.log_processing_step("Analytics", 
                "Analytics generation completed",
                status="completed"
            )
            
        except Exception as e:
            logging.error(f"Analytics error: {str(e)}", exc_info=True)
            visualizer.log_processing_step("Error", 
                f"Analytics error: {str(e)}",
                status="error"
            )
            st.error(f"Error generating analytics: {str(e)}")
def create_analytics_report(metadata_manager):
    """Create downloadable PDF report of analytics"""
    st.markdown("### Generate PDF Report")
    
    if st.button("Generate Report"):
        with st.spinner("Generating PDF report..."):
            try:
                report = ReportGenerator()
                
                # Add title and timestamp
                report.add_title("Video Analysis Report")
                report.add_text(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
                
                # Add video statistics section
                report.add_section_header("Video Statistics")
                stats_text = (
                    f"Duration: {metadata_manager.structured_data['source']['duration']}\n"
                    f"Total Scenes: {len(metadata_manager.structured_data['source']['tracks']['caption']['eventData'])}\n"
                    f"Available Languages: {len(metadata_manager.get_subtitle_languages())}"
                )
                report.add_text(stats_text)
                
                # Add scene distribution chart
                report.add_section_header("Scene Type Distribution")
                scene_types = metadata_manager.metadata_df['scene_type'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                scene_types.plot(
                    kind='pie',
                    ax=ax,
                    autopct='%1.1f%%',
                    colors=['#1e3d8f', '#2e5edb', '#4b7bff', '#7698ff', '#a3b8ff']
                )
                ax.set_title('Distribution of Scene Types')
                report.add_plot(fig, "Figure 1: Scene Type Distribution")
                plt.close()
                
                # Add object detection timeline
                report.add_section_header("Object Detection Timeline")
                object_timeline = []
                for _, row in metadata_manager.metadata_df.iterrows():
                    for obj in row['objects_detected']:
                        object_timeline.append({
                            'timestamp': row['formatted_time'],
                            'object': obj
                        })
                
                if object_timeline:
                    df_timeline = pd.DataFrame(object_timeline)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plt.scatter(df_timeline['timestamp'], df_timeline['object'], alpha=0.6)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    ax.set_title("Object Appearances Over Time")
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel("Detected Objects")
                    report.add_plot(fig, "Figure 2: Object Detection Timeline")
                    plt.close()
                
                # Create download link
                st.markdown(
                    report.get_pdf_download_link(),
                    unsafe_allow_html=True
                )
                
                st.success("Report generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

def init_session_state():
    """Initialize session state variables"""
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'current_timestamp' not in st.session_state:
        st.session_state.current_timestamp = 0
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

def main():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #1e3d8f;'>
                🎬 FINAS Demo x BlacX - Enhanced Video Analysis
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # API Key Management
    col1, col2 = st.columns([2, 1])
    with col1:
        api_key = st.text_input(
            "Enter BlacX API Key:",
            value=st.session_state.OPENAI_API_KEY,
            type="password",
            key="api_key_input"
        )
    
    if not api_key:
        st.warning("⚠️ Please enter your BlacX API key to continue.")
        return
    
    st.session_state.OPENAI_API_KEY = api_key
    
    # Initialize processors with API key
    processor = VideoProcessor(api_key)
    metadata_manager = MetadataManager(api_key)
    visualizer = ProcessingVisualizer()
    processor.visualizer = visualizer
    metadata_manager.visualizer = visualizer
    
    # File Upload
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Video (MP4, AVI)", 
            type=['mp4', 'avi'],
            help="Upload a video file for analysis"
        )
    
    if uploaded_file:
        # Create temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Video Processing Section
        st.markdown("### 🎥 Video Processing")
        
        process_col, status_col = st.columns([1, 2])
        
        with process_col:
            if st.button("🔄 Process Video", help="Start video analysis"):
                visualizer.clear_all()
                try:
                    # Extract frames (25% of total progress)
                    visualizer.log_processing_step("Frame Extraction", "Starting video frame extraction...")
                    frames, timestamps, frame_images, video_metadata, frame_numbers = processor.extract_frames(
                        video_path,
                        sample_rate=2
                    )
                    visualizer.log_processing_step("Frame Extraction", 
                        f"Successfully extracted {len(frames)} frames", 
                        status="completed"
                    )
                    
                    # Process frames with structured output (50% of total progress)
                    visualizer.log_processing_step("Frame Analysis", "Starting frame analysis...")
                    frame_metadata = []
                    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                        # Update progress and show current frame
                        visualizer.update_progress(i + 1, len(frames), "Frame Analysis")
                        visualizer.show_frame(frame_images[i])
                        
                        # Show the prompt being sent to the model
                        prompt = f"Analyzing frame {i+1}/{len(frames)} at timestamp {timestamp:.2f}s"
                        visualizer.show_prompt(prompt)
                        
                        # Process frame and show output
                        metadata = processor.analyze_frame(frame, timestamp)
                        visualizer.show_output(json.dumps(json.loads(metadata), indent=2))
                        frame_metadata.append(metadata)
                        
                        # Log progress
                        visualizer.log_processing_step("Frame Analysis", 
                            f"Processed frame {i+1}/{len(frames)}", 
                            status="running"
                        )

                    visualizer.log_processing_step("Frame Analysis", 
                        "Completed frame analysis", 
                        status="completed"
                    )
                    
                    # Process audio and generate translations (15% of total progress)
                    visualizer.log_processing_step("Audio Processing", "Starting audio extraction and translation...")
                    audio_data = processor.extract_audio_with_translations(video_path)
                    
                    # Log each language processing
                    for lang_code in audio_data.keys():
                        visualizer.log_processing_step("Translation", 
                            f"Generated subtitles for {lang_code}", 
                            status="completed"
                        )
                    
                    # Create structured metadata (final 10% of progress)
                    visualizer.log_processing_step("Metadata Organization", "Creating structured metadata...")
                    metadata_manager.create_metadata_df(
                        frame_metadata,
                        timestamps,
                        audio_data,
                        video_metadata,
                        frame_numbers
                    )
                    
                    # Store metadata manager in session state
                    st.session_state.metadata_manager = metadata_manager
                    st.session_state.video_processed = True
                    
                    visualizer.log_processing_step("Processing Complete", 
                        "All video processing tasks completed successfully", 
                        status="completed"
                    )
                    
                except Exception as e:
                    visualizer.log_processing_step("Error", str(e), status="error")
                    st.error(f"❌ Error during processing: {str(e)}")
                finally:
                    os.unlink(video_path)

        # Display processed content if available
        if st.session_state.video_processed and st.session_state.metadata_manager is not None:
            main_tabs = st.tabs(["Video Player", "Analysis", "Metadata", "Export"])
            
            with main_tabs[0]:
                create_video_player(uploaded_file, st.session_state.metadata_manager)
            
            with main_tabs[1]:
                create_analytics_view(st.session_state.metadata_manager)
                
                # Interactive Query Section with visualizer
                st.markdown("### 💬 Query Video Content")
                query = st.text_input(
                    "Ask about the video content:",
                    placeholder="e.g., What objects appear most frequently?"
                )
                
                if query:
                    visualizer.clear_all()
                    visualizer.log_processing_step("Query Analysis", f"Processing query: {query}")
                    visualizer.show_prompt(f"Analyzing: {query}")
                    
                    response = st.session_state.metadata_manager.query_metadata(query)
                    visualizer.show_output(response)
                    
                    visualizer.log_processing_step("Query Analysis", 
                        "Query processed successfully", 
                        status="completed"
                    )
            
            with main_tabs[2]:
                create_metadata_viewer(st.session_state.metadata_manager)
            
            with main_tabs[3]:
                st.markdown("### 💾 Export Options")
                export_format = st.selectbox(
                    "Select Export Format",
                    ["Structured JSON", "Full Analysis", "Subtitles Only", "Scene Analysis"]
                )
                
                if st.button("Export Data"):
                    visualizer.clear_all()
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        visualizer.log_processing_step("Export", f"Preparing {export_format} export...")
                        
                        if export_format == "Structured JSON":
                            data = json.dumps(
                                st.session_state.metadata_manager.structured_data,
                                indent=2,
                                ensure_ascii=False
                            )
                            visualizer.show_output(json.dumps(json.loads(data), indent=2)[:1000] + "...")
                            st.download_button(
                                "Download JSON",
                                data.encode('utf-8'),
                                f"video_analysis_{timestamp}.json",
                                "application/json"
                            )
                            
                        elif export_format == "Full Analysis":
                            full_data = {
                                "structured_data": st.session_state.metadata_manager.structured_data,
                                "frame_analysis": st.session_state.metadata_manager.metadata_df.to_dict('records'),
                                "video_info": st.session_state.metadata_manager.video_metadata
                            }
                            visualizer.show_output(json.dumps(full_data, indent=2)[:1000] + "...")
                            st.download_button(
                                "Download Full Analysis",
                                json.dumps(full_data, indent=2).encode('utf-8'),
                                f"full_analysis_{timestamp}.json",
                                "application/json"
                            )
                            
                        elif export_format == "Scene Analysis":
                            scene_data = {
                                str(idx): {
                                    "timestamp": row["timestamp"],
                                    "scene_type": row["scene_type"],
                                    "frame_description": row["frame_description"]
                                }
                                for idx, row in st.session_state.metadata_manager.metadata_df.iterrows()
                            }
                            visualizer.show_output(json.dumps(scene_data, indent=2)[:1000] + "...")
                            st.download_button(
                                "Download Scene Analysis",
                                json.dumps(scene_data, indent=2).encode('utf-8'),
                                f"scene_analysis_{timestamp}.json",
                                "application/json"
                            )
                            
                        else:  # Subtitles Only
                            subtitle_data = st.session_state.metadata_manager.structured_data['source']['subtitles']
                            visualizer.show_output(json.dumps(subtitle_data, indent=2)[:1000] + "...")
                            st.download_button(
                                "Download Subtitles",
                                json.dumps(subtitle_data, indent=2).encode('utf-8'),
                                f"subtitles_{timestamp}.json",
                                "application/json"
                            )
                        
                        visualizer.log_processing_step("Export", 
                            f"Successfully exported {export_format}", 
                            status="completed"
                        )
                        
                    except Exception as e:
                        visualizer.log_processing_step("Export Error", str(e), status="error")
                        st.error(f"Error exporting data: {str(e)}")

if __name__ == "__main__":
    init_session_state()
    main()