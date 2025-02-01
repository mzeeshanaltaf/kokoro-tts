import streamlit as st
from kokoro import KPipeline
from pydub import AudioSegment
import numpy as np
import io
import torch
import time

# Page title of the application
page_title = "Kokoro TTS"
page_icon = "🗣️"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")

# Voice names, language codes and sample text for different languages
voice_names = {'American': {'Female': ['Alloy', 'Aoede', 'Bella', 'Jessica', 'Kore', 'Nicole', 'Nova', 'River', 'Sarah', 'Sky'],
                            'Male': ['Adam', 'Echo', 'Eric', 'Fenrir', 'Liam', 'Michael', 'Onyx', 'Puck', 'Santa']},
               'British': {'Female': ['Alice' , 'Emma', 'Isabella', 'Lily'], 'Male': ['Daniel', 'Fable', 'George', 'Lewis']},
               'Spanish': {'Female': ['Dora'], 'Male': ['Alex', 'Santa']},
               'French': {'Female': ['Siwis'], 'Male': ['']},
               'Italian': {'Female': ['Sara'], 'Male': ['Nicola']},
               'Brazilian': {'Female': ['Dora'], 'Male': ['Alex', 'Santa']},
               }
language_codes = {'American': 'a', 'British': 'b','Spanish': 'e','French': 'f','Italian': 'i','Brazilian': 'p'}

sample_text_english = '''
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.
'''
sample_text_spanish = '''
Fue el mejor de los tiempos, fue el peor de los tiempos, fue la era de la sabiduría, fue la era de la necedad.
'''
sample_text_french = '''
C’était le meilleur des temps, c’était le pire des temps, c’était l’âge de la sagesse, c’était l’âge de la folie.
'''
sample_text_italian = '''
Era il migliore dei tempi, era il peggiore dei tempi, era l'età della saggezza, era l'età della stoltezza.
'''
sample_text_portuguese = '''
Foi o melhor dos tempos, foi o pior dos tempos, foi a era da sabedoria, foi a era da tolice.
'''

sample_text_mapping = {'American': sample_text_english, 'British': sample_text_english, 'Spanish': sample_text_spanish,
                       'French': sample_text_french, 'Italian': sample_text_italian, 'Brazilian': sample_text_portuguese,}

# Application Title and description
st.title(f'{page_title}🗨️🔉')
st.write('***:blue[🗣️ Turn Text into Voice, Your Way! 🎙️]***')
st.write("""
*Kokoro TTS is a cutting-edge text-to-speech application that brings your words to life! 🌟 With support for multiple 
languages, you can generate natural, lifelike speech in the language of your choice. Select from expressive male and 
female voices to create audio that resonates with your needs — whether for storytelling, presentations, accessibility, or 
multilingual applications. With seamless and high-quality voice synthesis, Kokoro TTS makes converting text into speech 
easier than ever! 🎙️✨*
""")
st.info("[Powered by Kokoro-82M Text to Speech Model](https://huggingface.co/hexgrad/Kokoro-82M)", icon='ℹ️')

# Configuration for Language selection and voice type
col1, col2 = st.columns(2, border=True)
with col1:
    st.subheader('Language Selection', divider="gray")
    language_selection = st.radio("Supported Language(s)", ["American English", "British English", "Spanish", "French",
                                                            "Italian", "Brazilian Portuguese"],
                          index=0, horizontal=False, label_visibility='collapsed')
    selected_language = language_selection.split()[0]

with col2:
    st.subheader('Voice Type', divider="gray")
    voice_type = st.radio("Supported Voice Type", ["Female :female-office-worker:", "Male :male-office-worker:"],
                          index=0, horizontal=False, label_visibility='collapsed')
    selected_voice_type = voice_type.split()[0]

# Voice name selection
st.subheader('Voice Names:', divider='gray')
selected_voice_name = st.selectbox("Voice Names:", voice_names[selected_language][selected_voice_type],
                                   label_visibility='collapsed')

# Display error if voice name does not exist. Also, disable the text input and generate audio button
voice_error = False
if selected_voice_name == '':
    st.error('Voice Name for this voice type does not exist. Please select another voice type.', icon='❗')
    voice_error = True

# Extract language code, voice pack and sample text
lang_code = language_codes[selected_language]
voice_pack = f"{lang_code}{selected_voice_type[0].lower()+'_'}{selected_voice_name.lower()}"
sample_text = sample_text_mapping[selected_language].strip()

# Text input
st.subheader('Input Text:', divider='gray')
text = st.text_area('Input Text', sample_text, label_visibility="collapsed", disabled=voice_error)

# Button for audio generation
generate_tts = st.button('Generate Audio', type='primary', icon=":material/text_to_speech:", disabled=voice_error)

# Convert text to speech if button is pressed
if generate_tts:
    with st.spinner('Converting Text to Speech ...'):
        start_time = time.time() # Start the time to keep track of time taken for text to speech conversion

        pipeline = KPipeline(lang_code=lang_code)
        combined_audio = AudioSegment.silent(duration=0)
        generator = pipeline(text, voice=voice_pack, speed=1, split_pattern=r'\n+')
        for i, (gs, ps, audio) in enumerate(generator):
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()

            # Normalize if float (PyTorch TTS models often output float32 in range -1 to 1)
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = (audio * 32767).astype(np.int16)  # Convert float to int16

            # Convert to bytes
            audio_bytes = audio.tobytes()
            segment = AudioSegment.from_raw(io.BytesIO(audio_bytes), sample_width=2, frame_rate=24000, channels=1)
            combined_audio += segment  # Append each segment

        # Convert to bytes and play directly
        buffer = io.BytesIO()
        combined_audio.export(buffer, format="wav")
        buffer.seek(0)

        # Stop the time and calculate time taken for tts conversion
        end_time = time.time()
        time_taken = end_time - start_time

        # Notification message with time taken for tts conversion
        st.toast(f"Audio Generated in {time_taken:.2f} seconds", icon='✅')

        # Display audio player
        st.subheader('Audio :loud_sound::', divider='gray')
        st.audio(buffer.read(), format="audio/mpeg")




