import numpy as np
from openwakeword.model import Model
import sounddevice as sd


SAMPLE_RATE = 16000
CHUNK_SIZE = 1280



def wake_for_activation_word():
    model = Model()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels = 1, dtype = "int16", blocksize = CHUNK_SIZE) as stream:
        print("started listening")
        while True:
            audio_data,overflowed = stream.read(CHUNK_SIZE)
            samples = audio_data.flatten()
            pred = model.predict(samples)
            
            confidence = pred["hey_jarvis"]
            if confidence > 0.5:
                print("Found word ", confidence)
                return True
    return False
