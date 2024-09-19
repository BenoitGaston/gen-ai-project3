from openai import OpenAI

import os
from scipy.io.wavfile import write
import io
import subprocess

import numpy as np


#openai_api_key = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

openai_client = OpenAI()


"""def speech_to_text_whisper_local(audio_binary):

    with open("my_file.opus", "wb") as binary_file:

        binary_file.write(audio_binary)

    command = 'ffmpeg -i  my_file.opus -vn -y -ar 44100 -ac 2 -b:a 192k input.mp3'   
    subprocess.call(command,shell=True)


    print('----- os.listdir() wav :',[f for f in os.listdir() if f.endswith('.wav')])
    
    # Load Whisper model and transcribe
    model = whisper.load_model("tiny")
    result = model.transcribe("input.mp3")#,options ={"language" : "en",})
    if 'input.mp3' in os.listdir():
        os.remove('input.mp3')

    if "my_file.opus" in os.listdir():
        os.remove("my_file.opus")

    print('* result stt : ', result )
    

    return result['text']"""



def speech_to_text(audio_binary):

    with open("my_file.opus", "wb") as binary_file:

        binary_file.write(audio_binary)

    command = 'ffmpeg -i  my_file.opus -vn -y -ar 44100 -ac 2 -b:a 192k input.mp3'   
    subprocess.call(command,shell=True)

    audio_file= open("input.mp3", "rb")

    transcription = openai_client.audio.transcriptions.create(model="whisper-1", 
                                                                file=audio_file)

    if 'input.mp3' in os.listdir():
        os.remove('input.mp3')

    if "my_file.opus" in os.listdir():
        os.remove("my_file.opus")

    print('* result transcription : ', transcription.text )
    

    return transcription.text

def split_text_into_short_sentences(text):

    text_temp_list = text.replace('.',' _ ').replace('!',' _ ').replace('?',' _ ').split(' _ ')
    text_list = []
    for sentence in text_temp_list:

        if len(sentence) >= 600:
            sentence = sentence.split(',')
            sub_list = []
            for sub_sentence in sentence:
                if len(sub_sentence) >= 600:
                    sub_sentence = sub_sentence.split(',')
                else:
                    sub_sentence = [sub_sentence]
                sub_list+=sub_sentence
            sentence = sub_list
                

        else:
            if len(sentence) > 0:
                sentence = [sentence]

        text_list+=sentence
    return text_list




"""def text_to_speec_microsoft(text,voice ="Matthijs/cmu-arctic-xvectors"):

    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_from_disk('embeddings_dataset.hf')
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    text_list = split_text_into_short_sentences(text)

    # You can replace this embedding with your own as well.
    sentence_count = 0
    np_array = np.array([])

    for sentence in text_list:

        speech = synthesiser(sentence, 
                        forward_params={"speaker_embeddings": speaker_embedding})

        if sentence_count == 0:
            np_array = speech["audio"]
        else:
            np_array = np.concatenate((np_array, speech["audio"]), axis=0)
        sentence_count+=1


    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)

    write(byte_io, speech["sampling_rate"], np_array)

    output_wav = byte_io.read()

    return output_wav"""


def text_to_speech(text,voice=""):

    text_list = split_text_into_short_sentences(text)

    # You can replace this embedding with your own as well.
    sentence_count = 0
    np_array = np.array([])

    for sentence in text_list:

        response = openai_client.audio.speech.create(model="tts-1",
                                            voice="alloy",
                                            input=sentence,
                                            response_format = "wav",)
        response.stream_to_file("output.wav")
        from scipy.io import wavfile
        samplerate, data = wavfile.read('output.wav')

        if sentence_count == 0:
            np_array = data
        else:
            np_array = np.concatenate((np_array, data), axis=0)
        sentence_count+=1


    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)

    write(byte_io, samplerate, np_array)

    output_wav = byte_io.read()

    return output_wav

def openai_process_message(conversation_history):
    # Set the prompt for OpenAI Api
    # Call the OpenAI Api to process our prompt
    openai_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=conversation_history,
        max_tokens=4000
    )
    #print("openai response:", openai_response)
    # Parse the response to get the response message for our prompt
    response_text = openai_response.choices[0].message.content
    return response_text
