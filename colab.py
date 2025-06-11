import whisper
import os
import gradio as gr
import requests
from urllib.parse import parse_qs, urlparse, unquote
from io import BytesIO
import tempfile
import re
import subprocess # Added for ffmpeg

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
    return filename

# Function to download and process M3U8
def process_m3u8(m3u8_url, output_dir):
    print(f"Processing M3U8 URL: {m3u8_url}")
    
    # Ensure ffmpeg is installed (important for Colab)
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
    except FileNotFoundError:
        print("FFmpeg not found. Installing FFmpeg...")
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
        print("FFmpeg installed.")

    output_audio_path = os.path.join(output_dir, "m3u8_output.mp3")
    
    try:
        # Use ffmpeg to download and convert M3U8 to MP3
        # -i: input M3U8 URL
        # -c:a copy: copy audio stream without re-encoding if possible
        # -vn: no video
        # -y: overwrite output file without asking
        # Note: ffmpeg can directly handle M3U8 to MP3 conversion
        command = [
            'ffmpeg',
            '-i', m3u3_url,
            '-vn', # no video
            '-c:a', 'libmp3lame', # encode to mp3, requires libmp3lame
            '-q:a', '2', # VBR quality for MP3
            '-y',
            output_audio_path
        ]
        
        print(f"Executing FFmpeg command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("FFmpeg stdout:\n", result.stdout)
        print("FFmpeg stderr:\n", result.stderr)
        
        if os.path.exists(output_audio_path):
            print(f"Successfully processed M3U8 and saved to: {output_audio_path}")
            return output_audio_path
        else:
            raise Exception("FFmpeg failed to create output audio file.")
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error during M3U8 processing: {e.stderr}")
        raise gr.Error(f"Failed to process M3U8: {e.stderr}")
    except Exception as e:
        print(f"General error processing M3U8: {e}")
        raise gr.Error(f"An error occurred while processing M3U8: {e}")


# Transcribe audio function
def transcribe_audio(file_path_or_bytesio, model_name, language):
    model = whisper.load_model(model_name)  # Load selected model
    print(f"Model '{model_name}' loaded successfully.")
    
    # Handle BytesIO for direct URL downloads (non-M3U8)
    if isinstance(file_path_or_bytesio, BytesIO):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(file_path_or_bytesio.read())
            tmp_file_path = tmp_file.name
        print(f"Temporary file created: {tmp_file_path}")
        audio = whisper.load_audio(tmp_file_path)
        os.remove(tmp_file_path)
    else: # This path will now also be used for processed M3U8 files
        audio = whisper.load_audio(file_path_or_bytesio)
    
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
    
    if url_input:
        if url_input.endswith('.m3u8'):
            print("Detected M3U8 URL.")
            # Create a temporary directory for M3U8 processing
            with tempfile.TemporaryDirectory() as tmp_dir:
                audio_for_whisper = process_m3u8(url_input, tmp_dir)
                file_name = extract_filename_from_url(url_input) or "m3u8_stream"
        else:
            print("Detected standard URL.")
            response = requests.get(url_input)
            audio_for_whisper = BytesIO(response.content)
            print("Audio downloaded from the URL.")
            file_name = extract_filename_from_url(url_input) or "url_audio"
    elif file_input:
        audio_for_whisper = file_input.name
        file_name = os.path.splitext(os.path.basename(audio_for_whisper))[0]
    else:
        gr.Warning("No file or URL provided.")
        return None, None

    # Ensure a file name is set
    file_name = file_name or "subtitle"

    transcription, segments = transcribe_audio(audio_for_whisper, model_name, language)
    txt_filename = save_transcription(transcription, file_name)
    srt_filename = save_subtitles(segments, file_name)
    
    # If audio_for_whisper was a temporary file from M3U8, clean it up
    if isinstance(audio_for_whisper, str) and "m3u8_output.mp3" in audio_for_whisper:
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
        url_input = gr.Textbox(label="Or enter an audio/video URL (e.g., MP3, WAV, M3U8)", placeholder="Enter audio/video file URL here...")  # type: ignore
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
