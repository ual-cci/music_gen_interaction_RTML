from timeit import default_timer as timer

class Settings(object):
    """
    Shared settings for all hardcoded values (easier for migrations of code and such...)

    """

    def __init__(self, args=None):
        self.server_model_paths_start = "/media/vitek/SCAN/LONDON_external_data/ProcessedMusicData/__saved_models/"
        self.server_model_paths_start = "__saved_models/"

        self.server_songs_paths_start = "/media/vitek/SCAN/LONDON_external_data/ProcessedMusicData/_music_samples/"
        self.server_songs_paths_start = "__music_samples/"

        if args is not None:
            # load params for the model
            self.lstm_layers = args.lstm_layers
            self.lstm_units = args.lstm_units
            self.griffin_iterations = args.griffin_iterations
            self.sample_rate = args.sample_rate





    def print_settings(self):
        print("Settings:")
        print("\t- server_model_paths_start:", self.server_model_paths_start)
        print("\t- server_songs_paths_start:", self.server_songs_paths_start)
        print("Server settings: settings.lstm_layers=", self.lstm_layers, ", settings.lstm_units=", self.lstm_units,
              ", settings.griffin_iterations=", self.griffin_iterations, ", settings.sample_rate=", self.sample_rate)
