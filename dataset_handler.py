import numpy as np
import utils.audio_dataset_generator
import settings
import model_handler_lstm
import tensorflow as tf
import tflearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
from audio_handler import AudioHandler

class TrainingMonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, record):
        self.record = record

    def on_epoch_end(self, training_state):
        #print("training_state >>",training_state)
        #print("training_state.global_acc >>",training_state.global_acc)
        #print("training_state.global_loss >>",training_state.global_loss)

        self.record.append(training_state.global_loss)
        """self.record.append({
            "accuracy": training_state.global_acc,
            "loss": training_state.global_loss,
        })"""



class Dataset(object):
    """
    Dataset for the training and report handlers (will be also for the main model eventually)

    """

    def __init__(self, settings):
        self.settings = settings

    def make_dataset(self, music_file):
        audio_handler = AudioHandler(griffin_iterations=self.settings.griffin_iterations, sample_rate=self.settings.sample_rate,
                                                        fft_size=self.settings.fft_size, window_size=self.settings.window_size,
                                                        hop_size=self.settings.hop_size, sequence_length=self.settings.sequence_length)

        dataset = utils.audio_dataset_generator.AudioDatasetGenerator(
            fft_size = self.settings.fft_size, window_size = self.settings.window_size, hop_size = self.settings.hop_size,
            sequence_length = self.settings.sequence_length, sample_rate = self.settings.sample_rate)

        dataset.load_from_wav_noSave(music_file, prevent_shuffling=False)

        print("Dataset:", dataset.x_frames.shape, dataset.y_frames.shape)
        return dataset, audio_handler

