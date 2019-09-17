import numpy as np
import audio_handler
import model_handler_lstm

class ServerHandler(object):
    """
    Does all the processing on the side of the Server
    (so that the entry point server.py can be as little as possible).

    - generate sample

    """

    def __init__(self):

        self.audio_handler = audio_handler.AudioHandler()
        self.model_handler = model_handler_lstm.ModelHandlerLSTM()


    def generate_audio_sample(self):

        lenght = 1024
        audio_arr = np.ones([lenght, ])

        return audio_arr