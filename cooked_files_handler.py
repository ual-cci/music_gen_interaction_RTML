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
        model_paths = ["modelBest_DNB1.tfl", "modelBest_LPAmbient.tfl", "modelBest_Mehldau.tfl", "modelBest_Ambient.tfl",
                       "modelBest_Dungeon.tfl", "modelBest_Orchestral.tfl", "modelBest_Sneak.tfl",
                       "modelBest_Glass1.tfl", "modelBest_Glass2.tfl", "modelBest_Twinpeaks.tfl",
                       "modelBest_Providence.tfl"]
        self.model_paths = [model_path_start+p for p in model_paths]

        # Loading from WAVs will be slow ... maybe load from the NPY's directly?

        song_paths_start = self.settings.server_songs_paths_start
        song_paths = ["dnb", "lp", "mehldau", "ambience", "dungeon","orchestral", "sneak", "glass1", "glass2", "twinpeaks"]

        self.names_for_debug = song_paths + ["providence"]

        self.song_paths = [song_paths_start+s+"/" for s in song_paths]

        print("prepared", len(self.model_paths), "models and ", len(self.song_paths), "songs.")

