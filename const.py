import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_TO_CONSIDER = 44100
N_MFCC=13
HOP_LENGTH=512
N_FFT=2048
SERVICE_URL = "https://buzzfinder-classifier-v2-mnvqg6klaa-uc.a.run.app"

if __name__ == "__main__":
    print(SERVICE_URL)
