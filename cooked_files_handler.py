from timeit import default_timer as timer

class CookedFilesHandler(object):
    """
    Takes care about all trained and prepared in advance (aka the models and the songs!)

    """

    def __init__(self, settings):
        self.settings = settings
        self.prepare_songs_models_paths()

    def prepare_songs_models_paths(self):
        # from model_i and song_i
        model_path_start = self.settings.server_model_paths_start

        # Get a list of "*.tfl.data-00000-of-00001" files
        import glob
        model_paths = glob.glob(model_path_start+"*.tfl.data-00000-of-00001")
        model_paths = sorted(model_paths)
        print(model_paths)

        model_paths = [p.replace(".tfl.data-00000-of-00001", ".tfl") for p in model_paths]

        self.model_paths = model_paths

        # Loading from WAVs will be slow ... maybe load from the NPY's directly?

        song_paths_start = self.settings.server_songs_paths_start
        import os
        song_paths = [d for d in os.listdir(song_paths_start) if os.path.isdir(os.path.join(song_paths_start, d))
                      and "baked_files" != d and "_all_mp3s" != d and "__outside" != d]

        self.names_for_debug = [name.split("/")[-1] for name in model_paths]

        song_paths = sorted(song_paths)
        self.song_paths = [song_paths_start+s+"/" for s in song_paths]

        print("prepared", len(self.model_paths), "models and ", len(self.song_paths), "songs.")

        print("We have loaded these songs and models (ps: their order should match, please name them accordingly):")
        print("models:")
        for m in self.model_paths:
            print(m)
        print("songs:")
        for s in self.song_paths:
            print(s)
