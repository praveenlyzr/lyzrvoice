import streamlit as st
from pydub import AudioSegment
import os
import tempfile
from audio_recorder_streamlit import audio_recorder

if not os.path.exists('tempDir'):
    os.makedirs('tempDir')

# Make sure to replace 'your_openai_api_key' with your actual OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets['openapikey']

def text_to_notes(text):   
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {
          "role": "system",
          "content": "You are an expert in taking down notes as bullet points and summarizing big conversations. you make sure no detail is left out"
        },
        {
          "role": "user",
          "content": f"Here is my conversation: {text}, create notes for this"
        }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text = response.choices[0].message.content
    return text

def transcribe(location):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v2"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(location)
    return result['text']

def save_uploadedfile(uploaded_file):
    with open(os.path.join('tempDir',uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploaded_file.name))

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Audio Lyzr App")

# Record audio
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    # Save the recorded audio for transcription
    with open('tempDir/output.wav', 'wb') as f:
        f.write(audio_bytes)
    transcript = transcribe('tempDir/output.wav')
    st.write(transcript)
    if transcript:
        ainotes = text_to_notes(transcript)
        st.write(ainotes)

# Or upload audio file
uploaded_file = st.file_uploader("Upload Files",type=['wav'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)
    save_uploadedfile(uploaded_file)
    audio_file = open(os.path.join('tempDir',uploaded_file.name),"rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    transcript = transcribe(os.path.join('tempDir',uploaded_file.name))
    st.write(transcript)
    if transcript:
        ainotes = text_to_notes(transcript)
        st.write(ainotes)


