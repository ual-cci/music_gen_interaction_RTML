import numpy as np
import audio_handler
import model_handler_lstm
from utils import file_functions
from timeit import default_timer as timer
import cooked_files_handler
import tensorflow as tf
import tflearn
import threading
import utils.audio_dataset_generator

class ServerHandler(object):
    """
    Does all the processing on the side of the Server
    (so that the entry point server.py can be as little as possible).

    - generate sample

    (maybe rename: GenerationHandler?)

    """

    def __init__(self, settings, manual_loading = False):
        settings.print_settings()
        self.settings = settings

        self.preloaded_impulses = None

        self.first_iteration = True

        self.keep_only_newly_generated = True

        self.continue_impulse_from_previous_batch = True
        # SPECIAL MODE - RESTART SEED EVERY "requested_length"-worth of generated music
        #    ... when self.continue_impulse_from_previous_batch is False
        #self.continue_impulse_from_previous_batch = False

        self.previous_audio_overlap = None

        self.change_impulses_smoothly = True
        #self.change_impulses_smoothly = False

        self._is_changing_impulses = False
        self._change_step = 0
        self._change_steps = 120

        self.model_i = 0
        self.song_i = 0
        self.interactive_i = 0
        self.weights_multiplier = 1.0

        self.audio_handler = audio_handler.AudioHandler(griffin_iterations=settings.griffin_iterations, sample_rate=settings.sample_rate,
                                                        fft_size=settings.fft_size, window_size=settings.window_size,
                                                        hop_size=settings.hop_size, sequence_length=settings.sequence_length)
        self.model_handler = model_handler_lstm.ModelHandlerLSTM(settings.lstm_layers, settings.lstm_units, self.settings)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model_handler.create_model()  # recreate?

        # Create a model
        #self.model_handler.create_model()
        print("Created model:", self.model_handler.model)

        if not manual_loading:

            self.songs_models = cooked_files_handler.CookedFilesHandler(settings)
            self.songs_models.prepare_songs_models_paths()

            # Load model weights & song impulses
            t_load_model = timer()
            self.load_weights(model_i = self.model_i)
            t_load_model = timer()-t_load_model

            t_load_song = timer()
            if self.preloaded_impulses is None:
                print("Also have to load the song")
                # this happens in Regular load mode (aka with older models)
                self.load_impulses(song_i= self.song_i) # no need we have the first one with the weights already
                self.change_impulse(interactive_i = self.interactive_i)
            t_load_song = timer()-t_load_song

            #print("Model loaded in", t_load_model, ", song loaded in", t_load_song, "(sec).")
            print("Model and song loaded in", t_load_model, "(sec).")
            # Time on potato pc: Model loaded in 1.007, song loaded in 12.365 (sec).

        self.VERBOSE = 1

        # Additional HAXs
        self.extended_functionality_function = None

    def manual_init_song_model(self, song_file, model_file):
        self.songs_models = cooked_files_handler.CookedFilesHandler(self.settings, False)
        self.songs_models.model_paths = [model_file]
        self.songs_models.song_paths = [song_file]
        self.songs_models.names_for_debug = [song_file]

        self.model_i = 0
        self.song_i = 0

        print("We have loaded these songs and models (ps: their order should match, please name them accordingly):")
        print("models:")
        for m in self.songs_models.model_paths:
            print(m)
        print("songs:")
        for s in self.songs_models.song_paths:
            print(s)

        # Load model weights & song impulses
        t_load_model = timer()
        self.load_weights(model_i=self.model_i, mode='graph')
        t_load_model = timer() - t_load_model

        t_load_song = timer()
        if self.preloaded_impulses is None:
            print("Also have to load the song")
            # this happens in Regular load mode (aka with older models)
            self.load_impulses(song_i=self.song_i)  # no need we have the first one with the weights already
            self.change_impulse(interactive_i=self.interactive_i)
        t_load_song = timer() - t_load_song

        # print("Model loaded in", t_load_model, ", song loaded in", t_load_song, "(sec).")
        print("Model and song loaded in", t_load_model, "(sec).")
        # Time on potato pc: Model loaded in 1.007, song loaded in 12.365 (sec).

    def load_weights_ASYNC_FUNCTION(self, model_i):

        model_path = self.songs_models.model_paths[model_i]
        print("Loading ...", model_path.split("/")[-1])

        new_model_handler = model_handler_lstm.ModelHandlerLSTM(self.settings.lstm_layers, self.settings.lstm_units, self.settings)
        new_graph = tf.Graph()
        with new_graph.as_default():
            new_model_handler.create_model()
            # Load model weights
            if file_functions.file_exists(model_path + ".data-00000-of-00001"):
                new_model_handler.load_model(model_path)
            else:
                print("[WARN!] No weights loaded in the model - it won't generate anything meaningful ...")

        # HAVE LOADED:
        print(">>>>>>>>>>>>>>>>>> FINISHED ASYNC LOADING MODEL:")
        print("new_model_handler", new_model_handler)
        print("new_graph", new_graph)
        old_model_handler = self.model_handler
        old_graph = self.graph
        self.model_handler = new_model_handler
        self.graph = new_graph

        del old_model_handler
        del old_graph


    def load_weights_ASYNC(self, model_i):
        """ Load on another thread, only then swap self.preloaded_impulses + self.model_handler.load_model(model_path) """

        threads = []
        t = threading.Thread(target=self.load_weights_ASYNC_FUNCTION, args=(model_i,))
        threads.append(t)
        t.start()

        self.model_i = model_i

    def load_weights(self, model_i, mode=''):
        model_path = self.songs_models.model_paths[model_i]
        print("Loading ...", model_path.split("/")[-1])

        if ("Model_" not in model_path) and (mode != 'graph'):
            print("Regular load")
            self.model_handler.create_model()

            # Load model weights
            if file_functions.file_exists(model_path+".data-00000-of-00001"):
                self.model_handler.load_model(model_path)
            else:
                print("[WARN!] No weights loaded in the model - it won't generate anything meaningful ...")
        else:
            print("Load with Graph")

            #new_graph = tf.Graph()
            with self.graph.as_default():
                #self.model_handler.create_model()  # recreate?

                # Load model weights
                if file_functions.file_exists(model_path+".data-00000-of-00001"):
                    self.model_handler.load_model(model_path)
                else:
                    print("[WARN!] No weights loaded in the model - it won't generate anything meaningful ...")

            # Hacky Load Of Dataset, oh yeaaah
            settings_file = model_path + ".settings"
            from settings import Settings
            tmp_settings = Settings()
            tmp_settings.load_from_txt(settings_file)
            audio_file = tmp_settings.debug_file
            dataset = self.audio_handler.load_dataset(audio_file)

            self.preloaded_impulses = dataset.x_frames
            print("Loaded self.preloaded_impulses from dataset!")

        self.model_i = model_i

    def load_impulses_ASYNC_FUNCTION(self, song_i):

        song_path = self.songs_models.song_paths[song_i]
        print("Loading music data...", song_path.split("/")[-2:])

        ## should this be in the audio_handler?
        dataset_path = song_i
        dataset_path = dataset_path.replace('/home/vitek/Projects', '/home/ubuntu/Projects')

        dataset = utils.audio_dataset_generator.AudioDatasetGenerator(
            fft_size = self.fft_size, window_size = self.window_size, hop_size = self.hop_size,
            sequence_length = self.sequence_length, sample_rate = self.sample_rate)
        try:
            dataset.load(dataset_path, prevent_shuffling=True)
        except:
            print("!!!! Datase is not loaded")
            dataset = None

        ###dataset = self.audio_handler.load_dataset(song_path)
        print(">>>>>>>>>>>>>>>>>> FINISHED ASYNC LOADING IMPULSES:")

        if dataset is not None:
            # now we care about the unsorted x_frames
            self.preloaded_impulses = dataset.x_frames
        else:
            print("[WARN!] No impulses for loading! Will generate some nonesence instead, but it won't really work.")
            self.preloaded_impulses = np.random.rand(10, 40, 1025)

        self.song_i = song_i
        print("Preloaded", len(self.preloaded_impulses), "impulse samples.")

    def load_impulses_ASYNC(self, song_i):
        """ Load on another thread, only then swap self.preloaded_impulses + self.model_handler.load_model(model_path) """

        threads = []
        t = threading.Thread(target=self.load_impulses_ASYNC_FUNCTION, args=(song_i,))
        threads.append(t)
        t.start()

        self.song_i = song_i

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

    def change_impulse(self, interactive_i = 0.0):
        impulse_scale = 1.0
        # seed it with an old sample
        # random_index = np.random.randint(0, (len(self.preloaded_impulses) - 1))

        # float interactive_i to int 0 - len(self.preloaded_impulses)
        random_index = int((len(self.preloaded_impulses) - 1) * interactive_i)
        print("random_index selected as", random_index, "from interactive_i=", interactive_i)
        impulse = np.array(self.preloaded_impulses[random_index]) * impulse_scale  # shape = (40, 1025)
        self.impulse = impulse

        self.interactive_i = interactive_i

    def change_impulse_smoothly_start(self, interactive_i = 0.0):

        # float interactive_i to int 0 - len(self.preloaded_impulses)
        selected_index = int((len(self.preloaded_impulses) - self._change_steps) * interactive_i)
        print("selected_index selected as", selected_index, "from interactive_i=", interactive_i)
        new_impulses = np.array(self.preloaded_impulses[selected_index:selected_index+self._change_steps]) # shape = (40, 1025)
        #print("new_impulses.shape:", new_impulses.shape)

        self.target_impulses = new_impulses

        self._is_changing_impulses = True
        self.interactive_i = interactive_i

    def change_lstm_net(self, target_tensor_name, operation, *kwargs):
        #self.model_handler.inspect_tensors()
        try:
            self.model_handler.change_lstm_net(target_tensor_name, operation, *kwargs)
        except Exception as e:
            print("Exception", e)
            print("Debug output, available tensors:")
            self.model_handler.inspect_tensors()
            assert False

    def generate_audio_sample(self, requested_length, interactive_i=0.0, method="Griff"):
        #audio_chunk = np.random.rand(36352, )
        #return audio_chunk, 0, 0

        # SPECIAL MODE - RESTART SEED EVERY "requested_length"-worth of generated music
        if not self.continue_impulse_from_previous_batch:
            self.change_impulse(interactive_i)

        # seed it from random
        #impulse = np.random.rand(40, 1025)
        #print(impulse.shape)

        impulse = self.impulse

        t_predict_start = timer()

        if self._is_changing_impulses and (self._change_steps is not 0):

            predicted_spectrogram, last_impulse, self._change_step, self._change_steps = self.model_handler.generate_sample__whileInterpolating(impulse, self.target_impulses, self._change_step, self._change_steps, requested_length)

            if self._change_step >= self._change_steps:
                self._change_step = 0
                self._is_changing_impulses = False # transition done
                del self.target_impulses
                print("change is over")

        else:
            predicted_spectrogram, last_impulse = self.model_handler.generate_sample(impulse, requested_length)
            #print("predicted_spectrogram.shape", predicted_spectrogram.shape)

        self.impulse = last_impulse

        t_reconstruct_start = timer()

        if self.VERBOSE > 1:
            print("Requested versus from impulse seed: requested_length=",requested_length,"seed=",self.model_handler.sequence_length)
        perc = float(requested_length) / (float(self.model_handler.sequence_length) + float(requested_length))
        if self.VERBOSE > 1:
            print("Percentage = ",perc, "(only this is new)")

        # Additional HAXs
        predicted_spectrogram = self.extended_functionality_spectrogram(predicted_spectrogram)

        audio = self.audio_handler.spectrogram2audio(predicted_spectrogram, method)

        print("precut audio.shape", audio.shape)

        if method == "LWS":
            print("HAX!") # sooo
            audio = audio[:-2048]
            # I think that lws is automaticaly zero padding ... so if we cut it here? (wont work with other lenghts tho)
            # the diff seems to have always been 2048

        # Return only the generated audio?
        if self.keep_only_newly_generated:
            # Keep only generated:
            perc_idx = int((len(audio) - 1) * (1.0-perc))
            audio = audio[perc_idx:]

        print("audio.shape", audio.shape)
        t_reconstruct_end = timer()

        t_predict = t_reconstruct_start - t_predict_start
        t_reconstruct = t_reconstruct_end - t_reconstruct_start

        return audio, t_predict, t_reconstruct


    def extended_functionality_spectrogram(self, spectrogram):
        if self.extended_functionality_function is not None:
            spectrogram = self.extended_functionality_function(
                spectrogram
            )
        return spectrogram