import numpy as np
import utils.audio_dataset_generator
import settings
import model_handler_lstm
import tensorflow as tf
import tflearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import audio_handler
import librosa


class ReportHandler(object):
    """
    Handles reporting models (as a page with gen samples ideally).

    """

    def __init__(self, args):
        self.folder = args.folder

        self.settings = settings.Settings()
        self.settings.print_settings()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ReportHandler for Project: Real Time Audio Generation.')
    parser.add_argument('-folder', help='folder with models', default='trained_models/')

    args = parser.parse_args()


    reporter = ReportHandler(args)

    reporter.demo_on_folder_of_files()