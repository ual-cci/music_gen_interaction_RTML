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
        """
        model_paths = ["modelBest_DNB1.tfl", "modelBest_LPAmbient.tfl", "modelBest_Mehldau.tfl", "modelBest_Ambient.tfl",
                       "modelBest_Dungeon.tfl", "modelBest_Orchestral.tfl", "modelBest_Sneak.tfl",
                       "modelBest_Glass1.tfl", "modelBest_Glass2.tfl", "modelBest_Twinpeaks.tfl",
                       #"modelBest_Providence.tfl",
                       #"dnb_lws_experiment_300_22khz_2048-2048-512fft.tfl"  # < here lws data had float64 - maybe caused issues?
                            "dnb_lws_experiment_300_22khz_2048-2048-512fft_forcedToFloat32.tfl",
                            "mehldau_lws_experiment_300_22khz_2048-2048-512fft_forcedToFloat32.tfl",
                            "mehldau_lws_experiment_300_22khz_2048-2048-512fft_forcedToFloat32_v126.tfl"
                       ]
        """
        model_paths = ["modelBest_DNB1.tfl", "modelBest_Mehldau.tfl", "modelBest_Glass1.tfl", "modelBest_Glass2.tfl", "modelBest_Twinpeaks.tfl",
                            "dnb_lws_experiment_300_22khz_2048-2048-512fft_forcedToFloat32.tfl",
                            "mehldau_lws_experiment_300_22khz_2048-2048-512fft_forcedToFloat32_v126.tfl"
                       ]
        model_paths = [model_path_start + p for p in model_paths]

        model_paths += [

                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_Zero7-IntheWaitingLine_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_Zero7-Destiny_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_the-white-stripes-seven-nation-army_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_the-tv-show_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_the-matrix-soundtrack-juno-reactor-vs-don-davis-navras_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_TheElderScrollsIIIMorrowindTheme_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_the-dubliners-rocky-road-to-dublin_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_TaketheLead_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_stasis_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_Royk-SoEasy_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_radiohead-everything-in-its-right-place_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_prokofiev-dance-of-the-knights_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_masu-trinity_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_masu_skin_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_masu_radio_future_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_masu_dark_city_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_kottonmouth-kings-rest-of-my-life_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_jazz-jackrabbit-2-soundtrack-by-alexander-brandon_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_i-would-walk-500-miles-the-proclaimers_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_i-follow-rivers_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_hotline-miami-soundtrack-hydrogen-by-moon_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_Hooverphonic_RenaissanceAffair_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_halcyon-orbital!!!_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_gorillaz-crystalised-the-xx-cover_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_glass3_gentlemanhonor(instrumental)_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_gemini-turn-me-on_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_donnie-darko-mad-world!!!_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_chik-chik-chik-theres-no-fucking-rule-dude_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_chairlift-met-before-video_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_bas-broekhuis-the-escher-drawings-part-vii_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_architecture-in-helsinki-do-the-whirlwind_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_arcade-fire-deep-blue-hq_mp3_wav3x128_sample22khz_griff60__train300epX64bt_",
                "/home/vitek/Projects/music_gen_interaction_RTML/trained_models/Model_angelo-badalamenti-twin-peaks-theme-instrumental-1990_mp3_wav3x128_sample22khz_griff60__train300epX64bt_"

        ]


        self.model_paths = model_paths

        # Loading from WAVs will be slow ... maybe load from the NPY's directly?

        song_paths_start = self.settings.server_songs_paths_start
        #song_paths = ["dnb", "lp", "mehldau", "ambience", "dungeon","orchestral", "sneak", "glass1", "glass2", "twinpeaks", "dnb_lws"] # < here lws data had float64 - maybe caused issues?
        song_paths = ["dnb", "lp", "mehldau", "ambience", "dungeon","orchestral", "sneak", "glass1", "glass2", "twinpeaks", "dnb_lws_forced32", "mehldau_lws_forced32", "mehldau_lws_forced32_126"] # < data with float32 always
        self.names_for_debug = [name.split("/")[-1] for name in model_paths]


        self.song_paths = [song_paths_start+s+"/" for s in song_paths]

        print("prepared", len(self.model_paths), "models and ", len(self.song_paths), "songs.")

