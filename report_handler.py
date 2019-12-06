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


        self.number_of_samples_per_model = 3
        self.length_of_generated = 1024 # 1024 is roughly 24 sec at 22khz


        # HAX subset
        #model_paths = model_paths[0:6]

        report_items = []

        for model_i, model_path in enumerate(model_paths):
            print("[[[[ Reporting model", model_i, "/", len(model_paths), ":", model_path)

            new_graph = tf.Graph()
            with new_graph.as_default():
                report_item = self.resurrect_model(folder, model_path)

                report_items.append(report_item)

        self.html_page(report_items)

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
        paths_to_samples = self.sample(model_handler, dataset, name, n_samples=self.number_of_samples_per_model, requested_length=self.length_of_generated)

        return self.report(model_file, audio_file, paths_to_samples, self.settings)


    def sample(self, model_handler, dataset, filename, n_samples = 5, requested_length = 1024):

        paths_to_samples = []

        for i in range(n_samples):
            random_index = np.random.randint(0, (len(dataset.x_frames) - 1))
            print("Generating and saving sample ",i,"with random index=", random_index)

            input_impulse = np.array(dataset.x_frames[random_index])

            predicted_spectrogram, _ = model_handler.generate_sample(input_impulse, requested_length = requested_length, window_size=1024)

            audio = self.audio_handler.spectrogram2audio(predicted_spectrogram)
            print("audio.shape", audio.shape)

            librosa.output.write_wav("reports/"+filename+"_sample_"+str(i)+".wav", audio, self.settings.sample_rate)
            paths_to_samples.append("reports/"+filename+"_sample_"+str(i)+".wav")

        return paths_to_samples

    def report(self, model_name, original_sample, generated_samples, settings):
        print("Reporting model", model_name)

        string_builder = "<div><p class='model_name'>" + str(model_name) + "</p>\n"
        #string_builder += "<p class='original_sample'>" + str(original_sample) + "</p>\n"

        string_builder += "<p class='original_sample'>Original audio<br><audio controls><source src='" + str(
            original_sample) + "' type='audio/wav'>Your browser does not support the audio element." + str(
            original_sample) + "</audio></p>\n"

        for i, sample in enumerate(generated_samples):

            sample_local = sample[8:]
            #string_builder += "<p class='sample'>" + str(sample) + "</p>\n"

            # <audio controls>
            #   <source src="horse.wav" type="audio/wav">
            #   <source src="horse.mp3" type="audio/mpeg">
            # Your browser does not support the audio element.
            # </audio>

            string_builder += "<p class='sample'>Generated sample "+str(i).zfill(2)+"<br><audio controls><source src='"+str(sample_local)+"' type='audio/wav'>Your browser does not support the audio element." + str(sample_local) + "</audio></p>\n"

        string_builder += "</div>\n"

        return string_builder


    def html_page(self, items, report_name = "report.html"):

        f = open('reports/'+report_name, 'w')

        message = """<html>
        <head><link rel="stylesheet" href="styles.css"></head>
        <body><h1>Generated models:</h1>
        """

        for item in items:
            message += item

        message += "</body></html>"

        f.write(message)
        f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ReportHandler for Project: Real Time Audio Generation.')
    parser.add_argument('-folder', help='folder with models', default='trained_models/')

    args = parser.parse_args()


    reporter = ReportHandler(args)

    #reporter.demo_on_folder_of_files()