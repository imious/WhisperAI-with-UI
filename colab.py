import whisper
import os
import gradio as gr
import requests
from urllib.parse import parse_qs, urlparse, unquote
from io import BytesIO
import tempfile
import re
import subprocess

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

# Extract file name from URL (improved for M3U8, or general URLs)
def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    filename = None
    if "response-content-disposition" in query_params:
        content_disposition = unquote(query_params["response-content-disposition"][0])
        if "filename=" in content_disposition or "filename*" in content_disposition:
            filename = content_disposition.split("filename*=")[-1].split("'")[-1]
    
    if filename:
        filename = os.path.splitext(filename)[0]  # Remove the file extension
    else:
        # Fallback for URLs without content-disposition, especially for M3U8
        path_segments = parsed_url.path.split('/')
        if path_segments:
            filename = os.path.splitext(path_segments[-1])[0]
        if not filename:
            filename = "transcribed_audio" # Default if no clear filename
    
    # Clean up filename for policy/signature parts if they were picked up
    filename = re.sub(r'\?Policy=.*', '', filename)
    filename = re.sub(r'\.m3u8$', '', filename) # Remove .m3u8 if still present
    filename = re.sub(r'\.mp4$', '', filename) # Remove .mp4 if it was the last part before query
    return filename

# Function to download and process M3U8
def process_m3u8(m3u8_url, output_dir, max_duration_seconds=None): # Changed default to None, handle in generate_files
    print(f"Processing M3U8 URL: {m3u8_url}")
    
    # Ensure ffmpeg is installed (important for Colab)
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
    except FileNotFoundError:
        print("FFmpeg not found. Installing FFmpeg...")
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
        print("FFmpeg installed.")

    # **CHANGE HERE: Output to MP4 instead of MP3, and copy codec to avoid re-encoding errors**
    output_audio_path = os.path.join(output_dir, "m3u8_output.mp4") # Changed to .mp4
    
    try:
        command = [
            'ffmpeg',
            '-protocol_whitelist', 'file,http,https,tcp,tls,crypto',
            '-i', m3u8_url,
            '-vn', # no video
            '-c:a', 'copy', # **CRUCIAL CHANGE: Copy audio codec directly**
            '-y',
            output_audio_path
        ]
        
        if max_duration_seconds is not None:
            command.insert(3, str(max_duration_seconds)) # Insert -t option after -i and input URL
            command.insert(3, '-t')
            
        print(f"Executing FFmpeg command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("FFmpeg stdout:\n", result.stdout)
        print("FFmpeg stderr:\n", result.stderr)
        
        if os.path.exists(output_audio_path):
            file_size_bytes = os.path.getsize(output_audio_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            print(f"Successfully processed M3U8 and saved to: {output_audio_path} (Size: {file_size_mb:.2f} MB)")
            return output_audio_path
        else:
            raise Exception("FFmpeg failed to create output audio file.")
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error during M3U8 processing: {e.stderr}")
        print(f"FFmpeg stdout (on error): {e.stdout}")
        raise gr.Error(f"Failed to process M3U8: {e.stderr}. Check console for more details.")
    except Exception as e:
        print(f"General error processing M3U8: {e}")
        raise gr.Error(f"An error occurred while processing M3U8: {e}. Check console for more details.")

# Transcribe audio function
def transcribe_audio(file_path_or_bytesio, model_name, language):
    model = whisper.load_model(model_name)  # Load selected model
    print(f"Model '{model_name}' loaded successfully.")
    
    # Handle BytesIO for direct URL downloads (non-M3U8)
    if isinstance(file_path_or_bytesio, BytesIO):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file: # Can keep .mp3 as it's from BytesIO or change to .wav/mp4
            tmp_file.write(file_path_or_bytesio.read())
            tmp_file_path = tmp_file.name
        print(f"Temporary file created from BytesIO: {tmp_file_path}")
        audio = whisper.load_audio(tmp_file_path)
        os.remove(tmp_file_path)
    else: # This path will now also be used for processed M3U8 files (which are file paths, e.g., .mp4)
        print(f"Loading audio from file path: {file_path_or_bytesio}")
        audio = whisper.load_audio(file_path_or_bytesio) # whisper.load_audio internally handles various formats
    
    options = {"language": language if language != "Auto" else None}
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

# Main function to generate files
def generate_files(file_input, url_input, model_name, language):
    print("Transcribing...")
    
    audio_for_whisper = None
    file_name = "transcribed_audio" # Default fallback
    
    # Define a maximum duration for M3U8 streams to prevent excessive processing time
    # This can be made a UI option later if needed. For now, a reasonable default.
    M3U8_MAX_DURATION_SECONDS = 300 # 5 minutes
    
    if url_input:
        parsed_url = urlparse(url_input)
        
        # More robust M3U8 detection: Check path and query, and use regex for .m3u8 pattern
        is_m3u8 = parsed_url.path.endswith('.m3u8') or bool(re.search(r'\.m3u8($|\?)', url_input, re.IGNORECASE))
        
        if is_m3u8:
            print("Detected M3U8 URL.")
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    # Pass the max_duration_seconds to process_m3u8 for slicing the stream
                    audio_for_whisper = process_m3u8(url_input, tmp_dir, max_duration_seconds=M3U8_MAX_DURATION_SECONDS)
                    file_name = extract_filename_from_url(url_input) or "m3u8_stream"
                except Exception as e:
                    # Error already handled by gr.Error in process_m3u8
                    return None, None
        else:
            print("Detected standard URL (non-M3U8).")
            try:
                # Use stream=True for large files but read into BytesIO for whisper's temp file
                response = requests.get(url_input, stream=True)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                audio_for_whisper = BytesIO(response.content)
                
                # Warning for large direct URL downloads
                if len(audio_for_whisper.getvalue()) > 500 * 1024 * 1024: # e.g., 500MB limit
                    gr.Warning("The downloaded file is very large. Transcription might take a long time or fail due to memory limits. Consider a shorter URL or uploading a smaller file.")
                
                print("Audio downloaded from the URL.")
                file_name = extract_filename_from_url(url_input) or "url_audio"
            except requests.exceptions.RequestException as e:
                gr.Error(f"Failed to download audio from URL: {e}. Please check the URL or try uploading a file.")
                return None, None
    elif file_input:
        audio_for_whisper = file_input.name
        file_name = os.path.splitext(os.path.basename(audio_for_whisper))[0]
        
        # Also add a check for uploaded files
        if os.path.exists(audio_for_whisper):
            file_size_bytes = os.path.getsize(audio_for_whisper)
            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb > 500: # e.g., 500MB limit for uploaded files
                gr.Warning(f"The uploaded file is {file_size_mb:.2f} MB, which is very large. Transcription might take a long time or fail due to memory limits. Consider compressing or trimming the audio.")
    else:
        gr.Warning("No file or URL provided.")
        return None, None

    # Ensure a file name is set
    file_name = file_name or "subtitle"

    try:
        print(f"Starting transcription of '{file_name}' using model '{model_name}' and language '{language}'...")
        transcription, segments = transcribe_audio(audio_for_whisper, model_name, language)
        print("Transcription complete. Saving results...")
        txt_filename = save_transcription(transcription, file_name)
        srt_filename = save_subtitles(segments, file_name)
    except Exception as e:
        gr.Error(f"Transcription failed: {e}. This often happens with very large files, unsupported audio formats, or issues during audio loading by Whisper/FFmpeg. Check console for details.")
        txt_filename = None
        srt_filename = None

    # If audio_for_whisper was a temporary file from M3U8, clean it up
    # Note: tmp_dir context manager handles directory deletion, but specific file needs removal
    if isinstance(audio_for_whisper, str) and "m3u8_output.mp4" in audio_for_whisper and os.path.exists(audio_for_whisper):
        try:
            os.remove(audio_for_whisper)
            print(f"Cleaned up temporary M3U8 output file: {audio_for_whisper}")
        except Exception as e:
            print(f"Error cleaning up M3U8 temp file: {e}")
            
    return txt_filename, srt_filename

# Gradio Blocks UI
with gr.Blocks() as demo:  # type: ignore
    gr.Markdown("# Audio Transcription and Subtitle Generation")  # type: ignore
    with gr.Row():  # type: ignore
        file_input = gr.File(label="Upload File (any type)", file_types=None)  # type: ignore
        url_input = gr.Textbox(label="Or enter an audio/video URL (e.g., MP3, WAV, M3U8, direct link to TS/MP4)", placeholder="Enter audio/video file URL here...")  # type: ignore
    with gr.Row():  # type: ignore
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
    with gr.Row():  # type: ignore
        output_txt = gr.File(label="Download Transcription (.txt)")  # type: ignore
        output_srt = gr.File(label="Download Subtitles (.srt)")  # type: ignore
    transcribe_button = gr.Button("Transcribe")  # type: ignore

    transcribe_button.click(
        generate_files,
        inputs=[file_input, url_input, model_dropdown, language_dropdown],
        outputs=[output_txt, output_srt]
    )

    # Footer 
    gr.Markdown("""    
    **By Iman** iman.barekatain@gmail.com  
    [iman.barekatain@student.kuleuven.be](mailto:iman.barekatain@student.kuleuven.be)  
    [GitHub](https://github.com/imious/WhisperAI-with-UI.git)  
    [LinkedIn](https://linkedin.com/in/iman-barekatain-1b33701b7)
    """)

demo.launch(share=True)
