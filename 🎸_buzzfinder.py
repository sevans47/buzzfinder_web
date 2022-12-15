import streamlit as st
import numpy as np
import requests
import os
import json
import random
from const import ROOT_DIR, SERVICE_URL
from preprocess import preprocess

## page config
st.set_page_config(page_title="buzzfinder", page_icon="🎸",layout="wide")

MODEL_ACCURACY = 95.8
NUM_AUGMENTATION_TECHNIQUES = 6
NUM_ORIGINAL_CLIPS = 886
NUM_TOTAL_CLIPS = 2651
TONES = ["buzzy 😬", "clean ✨", "muted 🔇"]

def get_random_audio_clip():
    audio_path = os.path.join(ROOT_DIR, "data", "audio_clips")
    random_file = random.choice(os.listdir(audio_path))
    if "buzzy" in random_file:
        tone = TONES[0]
    elif "clean" in random_file:
        tone = TONES[1]
    else:
        tone = TONES[2]
    return os.path.join(audio_path, random_file), tone

def make_request_to_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data = serialized_input_data,
    )

    return response.text

def call_api():

    # get random audio clip
    random_clip, tone = get_random_audio_clip()

    st.session_state['random_clip_path'] = random_clip
    st.session_state['expected_tone'] = tone

    # get mfccs from audio clip
    mfccs = preprocess(random_clip)

    # call api
    api_url = f"{SERVICE_URL}/classify"
    response = make_request_to_bento_service(api_url, mfccs)
    # response = ast.literal_eval(make_request_to_bento_service(api_url, mfccs))
    predicted_index = np.argmax(np.fromstring(response.strip("[]"), sep=","))

    st.session_state['predicted_index'] = predicted_index
    predicted_tone = TONES[predicted_index]
    st.session_state['predicted_tone'] = predicted_tone


st.title("buzzfinder 🎸")
st.header("An API for identifying buzzy notes when playing guitar")

with st.expander('Information'):

    st.markdown("**Background**")
    st.write("""The BuzzFinder API is able to identify whether a note played on guitar has a clean, buzzy, or muted tone. For
             the fretting hand (ie the hand whose fingers push down on the strings), these three tones correspond to
             three levels of pressure :
    """)

    st.markdown("""
                - clean: pushing down hard enough so that the string touches the fret, and the string vibrates cleanly
                - buzzy: pushing down so that the string touches the fret, but the pressure is too light resulting in the string buzzing against the fret when vibrating
                - muted: pushing down just a little so that the finger is dampening the string's vibration, but the string isn't touching the fret
    """)
    st.write("""Of these three tones, a buzzy tone is universally hated.  When practicing, it's essential to identify
             buzzy notes, diagnose their cause, and eradicate them.  The BuzzFinder aims to help guitarists in this endeavor.
    """)

    st.markdown("**Libraries used**")
    st.markdown("""
                - **Librosa**, **Numpy**, and **Audiomentations**: creating the dataset
                - **Tensorflow Keras**: training the model
                - **BentoML**: making the API
                - **Docker and GCP**: deploying the API
                - **Streamlit**: building the API demo website
    """)

    st.markdown("**Data**")
    st.write("""I created the dataset used to train the BuzzFinder model by recording myself playing single notes up and down my guitar.  I then used Librosa's
             onset detection function to automatically locate where in the recording I played notes, and made two second clips from
             those instances.  I then extracted the comprehensive MFCCs from the audio clips, which is an audio feature
             that's very good at describing the tone of a sound while ignoring pitch.  As I only wanted to identify the guitar's tone,
             this was the perfect feature to use.
             """)
    st.write(f"""
             Due to the limited size of my dataset, I used the Audiomentations library to augment my data.  In order to
             identify the most effective augmentation techniques, I created a custom cross validation function, and tested
             {NUM_AUGMENTATION_TECHNIQUES} different data augmentation techniques.  I then used Audiomentations's Compose function
             to create a variety of augmented data clips.  From {NUM_ORIGINAL_CLIPS} original audio clips, I was able to create
             a dataset with {NUM_TOTAL_CLIPS} total clips.
             """)

    st.markdown("**Model**")
    st.write(f"I made buzzfinder's model using a convolutional neural network and acheived {MODEL_ACCURACY}% accuracy when classifying tones from my test set.")
    st.write("""
             The model has 3 convolusional layers, each one with normalization and max pooling layers.  Next is a dense layer, and a final
             dense softmax output layer with three outputs for buzzy, clean, or muted predictions.""")

    # st.markdown("**Future improvements**")
    # st.markdown("""
    #             - Get data from different types of guitars.  I used a full sized classical guitar and a miniature nylon string guitar to make my dataset, but I would also like recordings from  steel string acoustic guitars and electric guitars.
    #             - Add one or two extra guitar tones to each clip to simulate getting a buzzy note when multiple notes are being played
    #             - Add and additional data type that includes people talking and ambient sounds in order to make the model more robust against non-guitar sounds.
    # """)

    st.markdown("**Possible uses**")
    st.write("""I see two main use cases for the BuzzFinder API:""")
    st.markdown("""
             - *Identify all the buzzy or muted notes in a recording.*
             This would help a guitarist to more quickly locate trouble spots, and help bring awareness to their playing.
             - *Train a guitarist to better control their finger pressure.*
             If a guitarist pushes on the strings too hard, they risk finger pain and possible injury. It also hurts
             their technique as they're less able to move smoothly and freely. A great exercise to remedy this
             problem is to push on the string lightly enough to play a buzzy note.  Then, push down just a bit more
             to make it a clean note, but using the least amount of pressure. The buzzfinder API could be used to
             teach new guitarists this important exercise.
    """)


if 'expected_tone' not in st.session_state:
    st.session_state['expected_tone'] = None
if 'predicted_index' not in st.session_state:
    st.session_state['predicted_index'] = None
if 'predicted_tone' not in st.session_state:
    st.session_state['predicted_tone'] = None
if 'random_clip_path' not in st.session_state:
    st.session_state['random_clip_path'] = None

st.header("Test buzzfinder 🧪")

st.write("""
         Click the button below to test the api with an audio clip the model hasn't seen before.
         \n(please be patient while it loads...)
         """)

st.button("⚡Find Buzz!⚡", on_click=call_api)

st.write("**Expected tone**: ", st.session_state['expected_tone'])
# st.write("predicted_index", st.session_state['predicted_index'])
st.write("**Predicted tone**:", st.session_state['predicted_tone'])

if st.session_state['expected_tone'] == st.session_state['predicted_tone'] and st.session_state['expected_tone'] != None:
    st.write("✅ Booyah! A correct prediction!")
elif st.session_state['expected_tone'] != st.session_state['predicted_tone']:
    st.write("❌ Oh no! An incorrect prediction.")
else:
    st.write("")

if st.session_state['expected_tone'] != None:
    st.write("Listen to the audio below, or click the 'Find Buzz!' button to try another audio clip.")

# display audio player
if st.session_state['random_clip_path'] != None:
    audio_file = open(st.session_state['random_clip_path'], 'rb')
    audio_bytes = audio_file.read()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.audio(audio_bytes, format='audio/wav')

if __name__ == "__main__":
    response = call_api()

#     print(st.session_state['expected_tone'])
#     print(response)
