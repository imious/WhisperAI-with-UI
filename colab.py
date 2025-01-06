import whisper
import os
import gradio as gr
import requests
from urllib.parse import parse_qs, urlparse, unquote
from io import BytesIO
import tempfile

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
        filename = os.path.splitext(filename)[0]  # Remove the file extension
    return filename

# Transcribe audio function
def transcribe_audio(file_path_or_bytesio, model_name, language):
    model = whisper.load_model(model_name)  # Load selected model
    print(f"Model '{model_name}' loaded successfully.")
    
    if isinstance(file_path_or_bytesio, BytesIO):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(file_path_or_bytesio.read())
            tmp_file_path = tmp_file.name
        print(f"Temporary file created: {tmp_file_path}")
        audio = whisper.load_audio(tmp_file_path)
        os.remove(tmp_file_path)
    else:
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
def generate_files(audio_input, url_input, model_name, language):
    print("Transcribing...")
    if url_input:
        response = requests.get(url_input)
        audio_file = BytesIO(response.content)
        print("Audio downloaded from the URL.")
        file_name = extract_filename_from_url(url_input)
    elif audio_input:
        audio_file = audio_input
        file_name = os.path.splitext(os.path.basename(audio_input))[0]
    else:
        return "No audio file or URL provided."
    
    file_name = file_name or "subtitle"  # Fallback name if none found
    transcription, segments = transcribe_audio(audio_file if isinstance(audio_file, BytesIO) else audio_input, model_name, language)
    txt_filename = save_transcription(transcription, file_name)
    srt_filename = save_subtitles(segments, file_name)
    return txt_filename, srt_filename

# Gradio Blocks UI
with gr.Blocks() as demo:# type: ignore
    gr.Markdown("# Audio Transcription and Subtitle Generation")# type: ignore
    with gr.Row():# type: ignore
        audio_input = gr.Audio(type="filepath", label="Upload Audio File")# type: ignore
        url_input = gr.Textbox(label="Or enter an audio URL", placeholder="Enter audio file URL here...")# type: ignore
    with gr.Row():# type: ignore
        model_dropdown = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
            value="turbo",
            label="Select Whisper Model"
        )
        language_dropdown = gr.Dropdown(
            choices=["Auto", "en", "es", "fr", "de", "zh", "ja", "ko", "ru", "ar"],  # Add more as needed
            value="Auto",
            label="Select Language"
        )
    with gr.Row():# type: ignore
        output_txt = gr.File(label="Download Transcription (.txt)")# type: ignore
        output_srt = gr.File(label="Download Subtitles (.srt)")# type: ignore
    transcribe_button = gr.Button("Transcribe")# type: ignore

    transcribe_button.click(
        generate_files,
        inputs=[audio_input, url_input, model_dropdown, language_dropdown],
        outputs=[output_txt, output_srt]
    )
    
    # Footer 
    gr.Markdown("""    
    **By Iman**  
    iman.barekatain@gmail.com  
    [iman.barekatain@student.kuleuven.be](mailto:iman.barekatain@student.kuleuven.be)  
    [GitHub](https://github.com/imious/WhisperAI-with-UI.git)  
    [LinkedIn](https://linkedin.com/in/iman-barekatain-1b33701b7)
    """)

demo.launch(share=True)
