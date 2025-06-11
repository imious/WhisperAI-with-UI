import whisper
import os
import gradio as gr
import requests
from urllib.parse import parse_qs, urlparse, unquote
from io import BytesIO
import tempfile
import m3u8
import subprocess
import time

# Determine if running in Google Colab with Google Drive mounted
def get_subtitles_path():
    if os.path.exists('/content/drive'):
        drive_subtitles_path = '/content/drive/My Drive/subtitles'
        os.makedirs(drive_subtitles_path, exist_ok=True)
        return drive_subtitles_path
    else:
        local_subtitles_path = "subtitles"
        os.makedirs(local_subtitles_path, exist_ok=True)
        return local_subtitles_path

SUBTITLES_FOLDER = get_subtitles_path()

# Timestamp formatting function
def format_timestamp(seconds):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Extract file name from URL
def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    filename = None
    if "response-content-disposition" in query_params:
        content_disposition = unquote(query_params["response-content-disposition"][0])
        if "filename=" in content_disposition or "filename*" in content_disposition:
            filename = content_disposition.split("filename*=")[-1].split("'")[-1]
    if filename:
        filename = os.path.splitext(filename)[0]
    return filename

# Check if URL or file is M3U8
def is_m3u8(url_or_path):
    if url_or_path:
        return url_or_path.lower().endswith('.m3u8') or 'm3u8' in url_or_path.lower()
    return False

# Process M3U8 playlist and convert to audio
def process_m3u8_to_audio(m3u8_url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "audio_from_m3u8.wav")
        command = ['ffmpeg', '-i', m3u8_url, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', output_path]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return output_path
        else:
            print(f"FFmpeg WAV conversion failed, trying MP3. Error: {result.stderr}")
            output_path_mp3 = output_path.replace('.wav', '.mp3')
            command_fallback = ['ffmpeg', '-i', m3u8_url, '-vn', '-acodec', 'libmp3lame', '-y', output_path_mp3]
            result = subprocess.run(command_fallback, capture_output=True, text=True)
            if result.returncode == 0:
                return output_path_mp3
            else:
                raise Exception(f"Failed to process M3U8 after fallback: {result.stderr}")
    except Exception as e:
        raise e

# Transcribe audio function
def transcribe_audio(file_path_or_bytesio, model_name, language, progress):
    progress(0.4, desc=f"Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)
    print(f"Model '{model_name}' loaded successfully.")
    time.sleep(2)
    
    if isinstance(file_path_or_bytesio, BytesIO):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(file_path_or_bytesio.read())
            tmp_file_path = tmp_file.name
        audio = whisper.load_audio(tmp_file_path)
        os.remove(tmp_file_path)
    else:
        audio = whisper.load_audio(file_path_or_bytesio)
    
    options = {"language": language if language != "Auto" else None}
    
    progress(0.6, desc="Transcribing... (This may take a while)")
    result = model.transcribe(audio, **options)
    
    return result["text"], result["segments"]

# Save transcription as text file
def save_transcription(text, file_name):
    txt_filename = os.path.join(SUBTITLES_FOLDER, file_name + ".txt")
    with open(txt_filename, 'w') as f:
        f.write(text)
    print(f"Transcription saved to {txt_filename}")
    return txt_filename

# Save subtitles in SRT format
def save_subtitles(segments, file_name):
    srt_filename = os.path.join(SUBTITLES_FOLDER, file_name + ".srt")
    with open(srt_filename, 'w') as f:
        for i, segment in enumerate(segments, 1):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            start_time_str = format_timestamp(start_time)
            end_time_str = format_timestamp(end_time)
            f.write(f"{i}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{text}\n\n")
    print(f"Subtitles saved to {srt_filename}")
    return srt_filename

# MODIFIED: Function signature updated to accept `custom_filename`
def generate_files(file_input, url_input, custom_filename, model_name, language, progress=gr.Progress(track_tqdm=True)):
    temp_files_to_cleanup = []
    
    try:
        progress(0, desc="Starting: Reading input...")
        time.sleep(1) 

        audio_file = None
        file_name = None # Initialize file_name

        if url_input:
            if is_m3u8(url_input):
                progress(0.1, desc="Processing M3U8 stream... (This can take time)")
                audio_file = process_m3u8_to_audio(url_input)
                temp_files_to_cleanup.append(audio_file)
            else:
                progress(0.1, desc="Downloading from URL...")
                response = requests.get(url_input)
                audio_file = BytesIO(response.content)
        elif file_input:
            if is_m3u8(file_input.name):
                progress(0.1, desc="Processing M3U8 file... (This can take time)")
                with open(file_input.name, 'r') as f:
                    m3u8_content = f.read()
                with tempfile.NamedTemporaryFile(mode='w', suffix='.m3u8', delete=False) as tmp_m3u8:
                    tmp_m3u8.write(m3u8_content)
                    tmp_m3u8_path = tmp_m3u8.name
                audio_file = process_m3u8_to_audio(tmp_m3u8_path)
                temp_files_to_cleanup.extend([tmp_m3u8_path, audio_file])
            else:
                progress(0.1, desc="Reading uploaded file...")
                audio_file = file_input.name
        else:
            yield "Error: No file or URL provided.", "Error: No file or URL provided."
            return

        # --- NEW: Logic to determine the output filename ---
        if custom_filename and custom_filename.strip():
            # Priority 1: Use the custom filename if provided
            file_name = custom_filename.strip()
        elif url_input and is_m3u8(url_input):
            # Special case for M3U8 URLs which don't have a clear filename
            file_name = "m3u8_stream"
        elif url_input:
            # Priority 2: Extract filename from URL
            file_name = extract_filename_from_url(url_input)
        elif file_input:
            # Priority 2: Extract filename from uploaded file
            file_name = os.path.splitext(os.path.basename(file_input.name))[0]
        
        # Priority 3: Fallback to a default name if all else fails
        file_name = file_name or "subtitle"
        # --- End of new filename logic ---

        transcription, segments = transcribe_audio(audio_file, model_name, language, progress)

        progress(0.9, desc="Saving output files...")
        txt_filename = save_transcription(transcription, file_name)
        srt_filename = save_subtitles(segments, file_name)
        
        progress(1, desc="Done!")
        time.sleep(1)

        yield txt_filename, srt_filename
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        yield error_message, error_message
    
    finally:
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

# --- MODIFIED: Gradio UI updated with the new textbox ---
with gr.Blocks() as demo:
    gr.Markdown("# Audio Transcription and Subtitle Generation")
    gr.Markdown("*Now supports M3U8 playlists and includes a progress bar!*")
    with gr.Row():
        file_input = gr.File(label="Upload File (any type, including .m3u8)", file_types=None)
        url_input = gr.Textbox(label="Or enter an audio/M3U8 URL", placeholder="Enter audio file URL or M3U8 playlist URL here...")
    
    # NEW: Textbox for the user to enter a custom filename
    with gr.Row():
        custom_filename_input = gr.Textbox(label="Optional: Enter Custom Output Filename", placeholder="e.g., my-transcription")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
            value="turbo",
            label="Select Whisper Model"
        )
        language_dropdown = gr.Dropdown(
            choices=["Auto", "en", "es", "fr", "de", "zh", "ja", "ko", "ru", "ar"],
            value="Auto",
            label="Select Language"
        )
    with gr.Row():
        output_txt = gr.File(label="Download Transcription (.txt)")
        output_srt = gr.File(label="Download Subtitles (.srt)")
    transcribe_button = gr.Button("Transcribe")

    # MODIFIED: The new textbox `custom_filename_input` is added to the inputs list
    transcribe_button.click(
        generate_files,
        inputs=[file_input, url_input, custom_filename_input, model_dropdown, language_dropdown],
        outputs=[output_txt, output_srt]
    )

    # Footer 
    gr.Markdown("""    
    **By Iman** iman.barekatain@gmail.com  
    [iman.barekatain@student.kuleuven.be](mailto:iman.barekatain@student.kuleuven.be)  
    [GitHub](https://github.com/imious/WhisperAI-with-UI.git)  
    [LinkedIn](https://linkedin.com/in/iman-barekatain-1b33701b7)
    """)

# Launch the app with the allowed path
demo.launch(share=True, allowed_paths=[SUBTITLES_FOLDER])
