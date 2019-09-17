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

        model_path = "/media/vitek/Data/Vitek/Projects/2019_LONDON/music generation/saved_models/trained_model_last___dnb1_300ep_default.tfl"
        if file_functions.file_exists(model_path+".data-00000-of-00001"):
            self.model_handler.load_model(model_path)
        else:
            print("[WARN!] No weights loaded in the model - it won't generate anything meaningful ...")

        # Load impulse samples
        self.preloaded_impulses = []

        impulses_to_load_path = "data/saved_impulses_15" # or 100
        if file_functions.file_exists(impulses_to_load_path+".npz"):
            self.preloaded_impulses = file_functions.load_compressed(impulses_to_load_path)
        else:
            print("[WARN!] No impulses for loading! Will generate some nonesence instead, but it won't really work.")
            self.preloaded_impulses = np.random.rand(4, 40, 1025)

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