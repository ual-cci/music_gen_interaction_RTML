from timeit import default_timer as timer
import json

class Settings(object):
    """
    Shared settings for all hardcoded values (easier for migrations of code and such...)

    """

    def __init__(self, args=None):
        self.server_model_paths_start = "/media/vitek/SCAN/LONDON_external_data/ProcessedMusicData/__saved_models/"
        self.server_model_paths_start = "__saved_models/"

        self.server_songs_paths_start = "/media/vitek/SCAN/LONDON_external_data/ProcessedMusicData/_music_samples/"
        self.server_songs_paths_start = "__music_samples/"

        self.debug_file = ""

        if args is not None:
            # load params for the model
            self.lstm_layers = int(args.lstm_layers)
            self.lstm_units = int(args.lstm_units)
            self.griffin_iterations = int(args.griffin_iterations)
            self.sample_rate = int(args.sample_rate)
            self.async_loading = (args.async_loading == "True")

            self.fft_size = 2048
            self.window_size = 1024
            self.hop_size = 512
            self.sequence_length = int(args.sequence_length)

            # training specific
            if 'amount_epochs' in args:
                self.amount_epochs = int(args.amount_epochs)
            else:
                self.amount_epochs = 300
            if 'batch_size' in args:
                self.batch_size = int(args.batch_size)
            else:
                self.batch_size = 64
        else:
            self.lstm_layers = None
            self.lstm_units = None
            self.griffin_iterations = None
            self.sample_rate = None



    def print_settings(self):
        print("Settings:")
        print("\t- server_model_paths_start:", self.server_model_paths_start)
        print("\t- server_songs_paths_start:", self.server_songs_paths_start)
        print("Server settings: settings.lstm_layers=", self.lstm_layers, ", settings.lstm_units=", self.lstm_units,
              ", settings.griffin_iterations=", self.griffin_iterations, ", settings.sample_rate=", self.sample_rate)


    def save_into_txt(self, filename):
        filename = filename + ".settings"

        data = {}
        data['settings'] = []
        data['settings'].append({
            'server_model_paths_start': self.server_model_paths_start,
            'server_songs_paths_start': self.server_songs_paths_start,

            'lstm_layers': self.lstm_layers,
            'lstm_units': self.lstm_units,
            'griffin_iterations': self.griffin_iterations,
            'sample_rate': self.sample_rate,
            'fft_size': self.fft_size,
            'window_size': self.window_size,
            'hop_size': self.hop_size,
            'sequence_length': self.sequence_length,

            'amount_epochs': self.amount_epochs,
            'batch_size': self.batch_size,

            'debug_file': self.debug_file,
        })

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)


    def load_from_txt(self, filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            j = data['settings'][0]


            self.server_model_paths_start = j['server_model_paths_start']
            self.server_songs_paths_start = j['server_songs_paths_start']
            self.lstm_layers = j['lstm_layers']
            self.lstm_units = j['lstm_units']
            self.griffin_iterations = j['griffin_iterations']
            self.sample_rate = j['sample_rate']
            self.fft_size = j['fft_size']
            self.window_size = j['window_size']
            self.hop_size = j['hop_size']
            self.sequence_length = j['sequence_length']
            self.amount_epochs = j['amount_epochs']
            self.batch_size = j['batch_size']

            self.debug_file = j['debug_file']

