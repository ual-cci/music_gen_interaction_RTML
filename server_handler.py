import numpy as np
import audio_handler
import model_handler_lstm
from utils import file_functions

class ServerHandler(object):
    """
    Does all the processing on the side of the Server
    (so that the entry point server.py can be as little as possible).

    - generate sample

    (maybe rename: GenerationHandler?)

    """

    def __init__(self):

        self.audio_handler = audio_handler.AudioHandler()
        self.model_handler = model_handler_lstm.ModelHandlerLSTM()

        # Create a model
        self.model_handler.create_model()
        print("Created model:", self.model_handler.model)
        self.model_handler.load_model()

        # Load impulse samples
        self.preloaded_impulses = []
        self.preloaded_impulses = file_functions.load_compressed("data/saved_impulses_100")
        print("Preloaded", len(self.preloaded_impulses), "impulse samples.")

    def generate_audio_sample(self, requested_length):

        impulse_scale = 1.0

        random_index = np.random.randint(0, (len(self.preloaded_impulses) - 1))
        impulse = np.array(self.preloaded_impulses[random_index]) * impulse_scale

        predicted_spectrogram = self.model_handler.generate_sample(impulse, requested_length)
        print("predicted_spectrogram.shape", predicted_spectrogram.shape)

        audio = self.audio_handler.spectrogram2audio(predicted_spectrogram)
        print("audio.shape", audio.shape)

        return audio

        audio_arr = np.ones([requested_length, ])

        return audio_arr