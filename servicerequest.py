import json
import os
import io
import random
import numpy as np
import requests
from preprocess import preprocess
from const import ROOT_DIR, SERVICE_URL

# SERVICE_URL = "http://localhost:3000/classify"
# SERVICE_URL = "https://buzzfinder-classifier-mnvqg6klaa-uc.a.run.app/classify"
# SERVICE_URL = "https://buzzfinder-classifier-v2-mnvqg6klaa-uc.a.run.app/classify"

def get_random_audio_clip():
    audio_path = os.path.join(ROOT_DIR, "data")
    random_file = random.choice(os.listdir(audio_path))
    tone = "buzzy" if "buzzy" in random_file else "clean"
    return os.path.join(audio_path, random_file), tone

def make_request_to_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data = serialized_input_data,
        # header={"content-type": "application/json"}
    )

    return response.text


def main():
    audio_clip_path, expected_output = get_random_audio_clip()
    with open(audio_clip_path, "rb") as f:
        binary_clip = io.BytesIO(f.read())
    mfccs = preprocess(binary_clip)
    prediction = make_request_to_bento_service(f"{SERVICE_URL}/classify", mfccs)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()

    # audio_clip_path, expected_output = sample_random_audio_clip()
    # with open(audio_clip_path, "rb") as f:
    #     binary_clip = io.BytesIO(f.read())
    # bfs = Buzz_Finder_Service()
    # mfccs = bfs.preprocess(binary_clip)

    # print(type(binary_clip))
    # print(type(io.BytesIO(binary_clip)))
