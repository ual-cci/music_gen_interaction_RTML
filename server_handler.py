import numpy as np
import audio_handler
import model_handler_lstm
from utils import file_functions
from timeit import default_timer as timer
import cooked_files_handler

class ServerHandler(object):
    """
    Does all the processing on the side of the Server
    (so that the entry point server.py can be as little as possible).

    - generate sample

    (maybe rename: GenerationHandler?)

    """

    def __init__(self, settings, args):
        print("Server settings: args.lstm_layers=", args.lstm_layers, ", args.lstm_units=", args.lstm_units,
              ", args.griffin_iterations=", args.griffin_iterations, ", args.sample_rate=", args.sample_rate)
        self.settings = settings

        lstm_layers = int(args.lstm_layers)
        lstm_units = int(args.lstm_units)
        griffin_iterations = int(args.griffin_iterations)
        sample_rate = int(args.sample_rate)

        self.model_i = 0
        self.song_i = 0

        self.audio_handler = audio_handler.AudioHandler(griffin_iterations=griffin_iterations, sample_rate=sample_rate)
        self.model_handler = model_handler_lstm.ModelHandlerLSTM(lstm_layers, lstm_units)

        # Create a model
        self.model_handler.create_model()
        print("Created model:", self.model_handler.model)

        self.songs_models = cooked_files_handler.CookedFilesHandler(settings)
        self.songs_models.prepare_songs_models_paths()

        # Load model weights & song impulses
        t_load_model = timer()
        self.load_weights(model_i = self.model_i)
        t_load_model = timer()-t_load_model

        t_load_song = timer()
        self.load_impulses(song_i= self.song_i)
        t_load_song = timer()-t_load_song

        print("Model loaded in", t_load_model, ", song loaded in", t_load_song, "(sec).")
        # Time on potato pc: Model loaded in 1.007, song loaded in 12.365 (sec).

    def load_weights(self, model_i):
        model_path = self.songs_models.model_paths[model_i]
        print("Loading ...", model_path.split("/")[-1])
        # Load model weights
        if file_functions.file_exists(model_path+".data-00000-of-00001"):
            self.model_handler.load_model(model_path)
        else:
            print("[WARN!] No weights loaded in the model - it won't generate anything meaningful ...")

        self.model_i = model_i

    def load_impulses(self, song_i):
        """
        impulses_to_load_path = "data/saved_impulses_100" # or 100
        if file_functions.file_exists(impulses_to_load_path+".npz"):
            self.preloaded_impulses = file_functions.load_compressed(impulses_to_load_path)
        else:
            print("[WARN!] No impulses for loading! Will generate some nonesence instead, but it won't really work.")
            self.preloaded_impulses = np.random.rand(4, 40, 1025)
        """

        song_path = self.songs_models.song_paths[song_i]
        print("Loading music data...", song_path.split("/")[-2:])

        dataset = self.audio_handler.load_dataset(song_path)

        if dataset is not None:
            # now we care about the unsorted x_frames
            self.preloaded_impulses = dataset.x_frames
        else:
            print("[WARN!] No impulses for loading! Will generate some nonesence instead, but it won't really work.")
            self.preloaded_impulses = np.random.rand(10, 40, 1025)

        self.song_i = song_i
        print("Preloaded", len(self.preloaded_impulses), "impulse samples.")


    def generate_audio_sample(self, requested_length, interactive_i=0.0):
        #audio_chunk = np.random.rand(36352, )
        #return audio_chunk, 0, 0

        impulse_scale = 1.0

        # seed it with an old sample
        #random_index = np.random.randint(0, (len(self.preloaded_impulses) - 1))

        random_index = 0
        # float interactive_i to int 0 - len(self.preloaded_impulses)
        random_index = int((len(self.preloaded_impulses) - 1) * interactive_i)
        print("random_index selected as",random_index,"from interactive_i=",interactive_i)
        impulse = np.array(self.preloaded_impulses[random_index]) * impulse_scale # shape = (40, 1025)

        # seed it from random
        #impulse = np.random.rand(40, 1025)
        #print(impulse.shape)

        t_predict_start = timer()
        predicted_spectrogram = self.model_handler.generate_sample(impulse, requested_length)
        print("predicted_spectrogram.shape", predicted_spectrogram.shape)

        t_reconstruct_start = timer()

        print("Requested versus from impulse seed: requested_length=",requested_length,"seed=",self.model_handler.sequence_length)
        perc = float(requested_length) / (float(self.model_handler.sequence_length) + float(requested_length))
        print("Percentage = ",perc, "(only this is new)")

        audio = self.audio_handler.spectrogram2audio(predicted_spectrogram)

        # Return only the generated audio?
        #""" # Keep only generated:
        perc_idx = int((len(audio) - 1) * (1.0-perc))
        audio = audio[perc_idx:]
        #"""

        print("audio.shape", audio.shape)
        t_reconstruct_end = timer()

        t_predict = t_reconstruct_start - t_predict_start
        t_reconstruct = t_reconstruct_end - t_reconstruct_start

        return audio, t_predict, t_reconstruct

