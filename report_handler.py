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
from os import listdir
from os.path import isfile, join
from dataset_handler import Dataset

class ReportHandler(object):
    """
    Handles reporting models (as a page with gen samples ideally).

    """

    def __init__(self, args):
        self.settings = settings.Settings()
        self.settings.print_settings()

        folder = args.folder
        model_paths = self.find_models(folder)

        for model_path in model_paths:
            new_graph = tf.Graph()
            with new_graph.as_default():
                self.resurrect_model(folder, model_path)

    def find_models(self, folder):
        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
        onlymodels = [f for f in onlyfiles if "data-00000-of-00001" in f]
        return onlymodels

    def resurrect_model(self, folder, model_path):
        name = model_path.split(".data-00000-of-00001")[0]

        model_file = folder + name
        settings_file = folder + name + ".settings"
        self.settings.load_from_txt(settings_file)
        audio_file = self.settings.debug_file

        print(model_file)
        print(audio_file)

        dataset_handler = Dataset(self.settings)
        dataset, self.audio_handler = dataset_handler.make_dataset(audio_file)
        print("Loaded dataset.")

        model_handler = model_handler_lstm.ModelHandlerLSTM(self.settings.lstm_layers, self.settings.lstm_units,
                                                            self.settings)
        model_handler.create_model()
        print("Created model.")

        model_handler.load_model(model_file)

        print("Loaded ", model_file, "successfully ...")

        # Save samples
        self.sample(model_handler, dataset, name, n_samples=1)

    def sample(self, model_handler, dataset, filename, n_samples = 5, requested_length = 1024):

        for i in range(n_samples):
            random_index = np.random.randint(0, (len(dataset.x_frames) - 1))
            print("Generating and saving sample ",i,"with random index=", random_index)

            input_impulse = np.array(dataset.x_frames[random_index])

            predicted_spectrogram, _ = model_handler.generate_sample(input_impulse, requested_length = requested_length, window_size=1024)

            audio = self.audio_handler.spectrogram2audio(predicted_spectrogram)
            print("audio.shape", audio.shape)

            librosa.output.write_wav("reports/"+filename+"_sample_"+str(i)+".wav", audio, self.settings.sample_rate)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ReportHandler for Project: Real Time Audio Generation.')
    parser.add_argument('-folder', help='folder with models', default='trained_models/')

    args = parser.parse_args()


    reporter = ReportHandler(args)

    #reporter.demo_on_folder_of_files()