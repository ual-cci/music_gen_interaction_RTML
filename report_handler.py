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
import librosa.display
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


        self.number_of_samples_per_model = 5
        self.length_of_generated = 1024 # 1024 is roughly 24 sec at 22khz



        # HAX subset
        #model_paths = model_paths[0:4]
        #self.length_of_generated = 128


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
        paths_to_samples, samples_audio = self.sample(model_handler, dataset, name, n_samples=self.number_of_samples_per_model, requested_length=self.length_of_generated)

        length_override = len(samples_audio[0])
        self.audiofile_as_spectrogram(filename = "reports/"+name+"_orig.png", audio_file=audio_file, length_override = length_override)

        return self.report(model_file, audio_file, paths_to_samples, self.settings)


    def audiofile_as_spectrogram(self, filename, audio_file, length_override = None):
        audio, sr = librosa.load(audio_file, sr=self.settings.sample_rate)
        if length_override is not None:
            audio = audio[0:length_override]
            print("(debug) Original audio has", len(audio), "samples")
        self.save_audio_as_spectrogram(filename, audio)

    def save_audio_as_spectrogram(self, filename, audio):
        plt.figure()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-frequency power spectrogram')
        plt.savefig(filename)
        plt.close()

    def sample(self, model_handler, dataset, filename, n_samples = 5, requested_length = 1024):

        paths_to_samples = []
        samples_audio = []

        for i in range(n_samples):
            random_index = np.random.randint(0, (len(dataset.x_frames) - 1))
            print("Generating and saving sample ",i,"with random index=", random_index)

            input_impulse = np.array(dataset.x_frames[random_index])

            predicted_spectrogram, _ = model_handler.generate_sample(input_impulse, requested_length = requested_length, window_size=1024)

            audio = self.audio_handler.spectrogram2audio(predicted_spectrogram)
            print("audio.shape", audio.shape)
            samples_audio.append(audio)

            self.save_audio_as_spectrogram("reports/" + filename + "_sample_" + str(i) + ".png", audio)

            librosa.output.write_wav("reports/"+filename+"_sample_"+str(i)+".wav", audio, self.settings.sample_rate)
            paths_to_samples.append("reports/"+filename+"_sample_"+str(i)+".wav")

        return paths_to_samples, samples_audio

    def report(self, model_name, original_sample, generated_samples, settings):
        print("Reporting model", model_name)

        string_builder = "<div><p class='model_name'>" + str(model_name) + "</p>\n"
        #string_builder += "<p class='original_sample'>" + str(original_sample) + "</p>\n"

        orig_spect_name = generated_samples[0][8:] + "_orig.png"
        orig_spect_name = orig_spect_name.replace("_sample_0.wav", "")

        string_builder += "<table><tr>"

        string_builder += "<td class='original_sample'><span>Original audio<img src='"+str(orig_spect_name)+"' class='image'></span><br><audio controls><source src='" + str(
            original_sample) + "' type='audio/wav'>Your browser does not support the audio element." + str(
            original_sample) + "</audio></td>\n"


        for i, sample in enumerate(generated_samples):

            sample_local = sample[8:]
            sample_local_spec = sample_local[0:-4] + ".png"
            #string_builder += "<p class='sample'>" + str(sample) + "</p>\n"

            # <audio controls>
            #   <source src="horse.wav" type="audio/wav">
            #   <source src="horse.mp3" type="audio/mpeg">
            # Your browser does not support the audio element.
            # </audio>

            # ><span>Generated sample 01 <img src='Model_hotline-miami-soundtrack-hydrogen-by-moon_mp3_wav3x128_sample22khz_griff60__train300epX64bt__sample_0.png' class='image'>
            # </span>

            string_builder += "<td class='sample'><span>Generated sample "+str(i).zfill(2)+"<img src='"+str(sample_local_spec)+"' class='image'></span><br><audio controls><source src='"+str(sample_local)+"' type='audio/wav'>Your browser does not support the audio element." + str(sample_local) + "</audio></td>\n"

        string_builder += "</tr></table>"
        string_builder += "</div>\n"

        return string_builder


    def html_page(self, items, report_name = "report.html"):

        f = open('reports/'+report_name, 'w')

        message = """<html>
        <head>
            <link rel="stylesheet" href="styles.css">
            <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
            <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
            <script src="custom.js"></script>
        </head>
        <body><h1>Generated models: <span>[+]</span></h1>
        <div id='sortable'>
        """

        for item in items:
            message += item

        message += "</div></body></html>"

        f.write(message)
        f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ReportHandler for Project: Real Time Audio Generation.')
    parser.add_argument('-folder', help='folder with models', default='trained_models/')

    args = parser.parse_args()


    reporter = ReportHandler(args)

