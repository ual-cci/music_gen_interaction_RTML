from timeit import default_timer as timer

class Settings(object):
    """
    Shared settings for all hardcoded values (easier for migrations of code and such...)

    """

    def __init__(self, args=None):
        if args is None:
            # default values:

            self.server_model_paths_start = "/media/vitek/SCAN/LONDON_external_data/ProcessedMusicData/__saved_models/"
            self.server_model_paths_start = "__saved_models/"

            self.server_songs_paths_start = "/media/vitek/SCAN/LONDON_external_data/ProcessedMusicData/_music_samples/"
            self.server_songs_paths_start = "__music_samples/"


    def print_settings(self):
        print("Settings:")
        print("\t- server_model_paths_start:", self.server_model_paths_start)
        print("\t- server_songs_paths_start:", self.server_songs_paths_start)