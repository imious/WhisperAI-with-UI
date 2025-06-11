import whisper
import os
import gradio as gr
import requests
from urllib.parse import parse_qs, urlparse, unquote
from io import BytesIO
import tempfile
import ffmpeg  # New import

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
    # Handle M3U8 file names from path
    if parsed_url.path.endswith('.m3u8'):
        return os.path.splitext(os.path.basename(parsed_url.path))[0]
        
    query_params = parse_qs(parsed_url.query)
    filename = None
    if "response-content-disposition" in query_params:
        content_disposition = unquote(query_params["response-content-disposition"][0])
        if "filename=" in content_disposition or "filename*" in content_disposition:
            filename = content_disposition.split("filename*=")[-1].split("'")[-1]
    if filename:
        filename = os.path.splitext(filename)[0]  # Remove the file extension
    return filename

# --- NEW FUNCTION to handle M3U8 files ---
def process_m3u8(input_path, output_filename):
    """
    Processes an M3U8 playlist (from a URL or local file) and converts it to a standard audio format.
    FFmpeg handles the downloading of segments and concatenating them.
    """
    print(f"Processing M3U8 file: {input_path}")
    try:
        # Use ffmpeg-python to process the m3u8 file.
        # '-y' overwrites the output file if it exists.
        (
            ffmpeg
            .input(input_path, y=None)
            .output(output_filename, acodec='mp3', audio_bitrate='192k')
            .run(cmd=['ffmpeg', '-y'], capture_stdout=True, capture_stderr=True)
        )
        print(f"M3U8 file successfully converted to {output_filename}")
        return output_filename
    except ffmpeg.Error as e:
        print("FFmpeg Error:", e.stderr.decode())
        raise IOError(f"FFmpeg failed to process the M3U8 file: {e.stderr.decode()}") from e

# Transcribe audio function
def transcribe_audio(file_path_or_bytesio, model_name, language):
    model = whisper.load_model(model_name)
    print(f"Model '{model_name}' loaded successfully.")
    
    if isinstance(file_path_or_bytesio, BytesIO):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(file_path_or_bytesio.read())
            tmp_file_path = tmp_file.name
        print(f"Temporary file created: {tmp_file_path}")
        audio = whisper.load_audio(tmp_file_path)
        os.remove(tmp_file_path)
    else:
        # This path can now be a regular audio file or the ffmpeg-converted file
        audio = whisper.load_audio(file_path_or_bytesio)
    
    options = {"language": language if language != "Auto" else None}
    result = model.transcribe(audio, **options)
    return result["text"], result["segments"]

# Save transcription as text file
def save_transcription(text, file_name):
    txt_filename = os.path.join(SUBTITLES_FOLDER, file_name + ".txt")
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Transcription saved to {txt_filename}")
    return txt_filename

# Save subtitles in SRT format
def save_subtitles(segments, file_name):
    srt_filename = os.path.join(SUBTITLES_FOLDER, file_name + ".srt")
    with open(srt_filename, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            start_time_str = format_timestamp(start_time)
            end_time_str = format_timestamp(end_time)
            f.write(f"{i}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{text.strip()}\n\n")
    print(f"Subtitles saved to {srt_filename}")
    return srt_filename

# --- MODIFIED main function to generate files ---
def generate_files(file_input, url_input, model_name, language):
    print("Starting transcription process...")
    
    temp_audio_file_path = None
    file_name = "subtitle" # Default filename

    try:
        if url_input:
            file_name = extract_filename_from_url(url_input) or file_name
            if url_input.endswith('.m3u8'):
                print("M3U8 URL detected.")
                # Create a temporary file to store the converted audio
                temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_audio_file_path = temp_audio_file.name
                temp_audio_file.close() # Close the file so ffmpeg can write to it
                process_m3u8(url_input, temp_audio_file_path)
                audio_source = temp_audio_file_path
            else:
                print("Downloading audio from standard URL.")
                response = requests.get(url_input)
                response.raise_for_status()
                audio_source = BytesIO(response.content)

        elif file_input:
            file_name = os.path.splitext(os.path.basename(file_input.name))[0]
            if file_input.name.endswith('.m3u8'):
                print("M3U8 file upload detected.")
                temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_audio_file_path = temp_audio_file.name
                temp_audio_file.close()
                process_m3u8(file_input.name, temp_audio_file_path)
                audio_source = temp_audio_file_path
            else:
                audio_source = file_input.name
        else:
            return "No file or URL provided.", "No file or URL provided."

        transcription, segments = transcribe_audio(audio_source, model_name, language)
        
        txt_filename = save_transcription(transcription, file_name)
        srt_filename = save_subtitles(segments, file_name)
        
        return txt_filename, srt_filename

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}", f"Error: {e}"
        
    finally:
        # Clean up the temporary file if one was created
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)
            print(f"Cleaned up temporary file: {temp_audio_file_path}")

# --- UPDATED Gradio Blocks UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Audio Transcription and Subtitle Generation")
    gr.Markdown("Upload an audio/video file, provide a URL, or even an **M3U8 playlist** URL to get started.")
    with gr.Row():
        file_input = gr.File(label="Upload File (Audio, Video, or .m3u8)")
        url_input = gr.Textbox(label="Or enter an Audio/Video/M3U8 URL", placeholder="Enter URL here...")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large-v3"],
            value="base",
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

demo.launch(share=True, debug=True)
