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
from dataset_handler import Dataset

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



class TrainingHandler(object):
    """
    Handles training of new models on music.

    Ideally be able to jut point it to a .wav (/.mp3) file and it would train a model.
    Also save the settings in a .settings file so we can load it from the model file directly...

    """

    def __init__(self, args):
        self.settings = settings.Settings(args)
        self.settings.print_settings()


    def plot_losses(self, losses, filename):
        fig, ax = plt.subplots()
        ax.plot(losses)

        ax.set(xlabel='epoch', ylabel='loss',
               title='Loss over epochs')
        ax.grid()

        plt.ylim(bottom=0)
        fig.savefig(filename+".png")
        #plt.show()
        plt.close()

    def sample(self, model_handler, dataset, filename, n_samples = 5, requested_length = 1024):

        for i in range(n_samples):
            random_index = np.random.randint(0, (len(dataset.x_frames) - 1))
            print("Generating and saving sample ",i,"with random index=", random_index)

            input_impulse = np.array(dataset.x_frames[random_index])

            predicted_spectrogram, _ = model_handler.generate_sample(input_impulse, requested_length = requested_length, window_size=1024)

            audio = self.audio_handler.spectrogram2audio(predicted_spectrogram)
            print("audio.shape", audio.shape)

            librosa.output.write_wav(filename+"_sample_"+str(i)+".wav", audio, self.settings.sample_rate)

    def train_on_file(self, music_file, model_name = "trained_models/tmp"):
        # keep this in debug record
        self.settings.debug_file = music_file

        dataset_handler = Dataset(self.settings)
        dataset, self.audio_handler = dataset_handler.make_dataset(music_file)
        print("Loaded dataset.")

        model_handler = model_handler_lstm.ModelHandlerLSTM(self.settings.lstm_layers, self.settings.lstm_units, self.settings)
        model_handler.create_model()

        print("Created model.")


        # naming scheme:
        # - Model_dnb_wav_
        # - 3x128lstm_
        # - sample22khz_
        # - griff60_

        model_name = model_name + str(self.settings.lstm_layers) + "x" + str(self.settings.lstm_units) + "_"
        model_name = model_name + "sample" + str(int(self.settings.sample_rate/1000)) + "khz_"
        # if griff?
        model_name = model_name + "griff" + str(self.settings.griffin_iterations) + "_"
        model_name = model_name + "_"
        model_name = model_name + "train" + str(self.settings.amount_epochs) + "epX" + str(self.settings.batch_size) + "bt_"
        model_name = model_name + "_" + str(self.settings.sequence_length)+"seq"

        print("Model = |", model_name, "|")


        # Train!
        losses = []
        monitorCallback = TrainingMonitorCallback(losses)

        model_handler.model.fit(dataset.x_frames, dataset.y_frames, show_metric=True, batch_size=model_handler.batch_size,
                                n_epoch=model_handler.amount_epochs, callbacks=[monitorCallback])

        # Save plot
        print("report >>>", losses)
        self.plot_losses(losses, model_name)

        # Save model
        model_handler.model.save(model_name)

        # Save settings
        self.settings.save_into_txt(model_name)

        print("Trained ", model_name,  "successfully ...")

        # Save samples
        self.sample(model_handler, dataset, model_name, n_samples = 5)

        # cleanup!
        del dataset
        del model_handler.model
        del model_handler
        del self.audio_handler

    def demo(self):

        music_file = "/home/vitek/Projects/music_gen_interaction_RTML/__music_samples/dnb/dnb.wav"

        audio_tag = music_file.split("/")[-1].replace(".","_")
        model_name = "trained_models/Model_"+audio_tag
        self.train_on_file(music_file, model_name)



    def demo_on_folder_of_files(self):

        music_files = [

            "/home/vitek/Projects/music_gen_interaction_RTML/new_audio_samples/whole_wavs/TaketheLead.mp3.wav",

        ]


        for music_i, music_file in enumerate(music_files):
            audio_tag = music_file.split("/")[-1].replace(".", "_")
            model_name = "trained_models/Model_" + audio_tag

            print("[[[[ Training on music", music_i, "/", len(music_files), ":", music_file)

            new_graph = tf.Graph()
            with new_graph.as_default():
                self.train_on_file(music_file, model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TrainerHandler for Project: Real Time Audio Generation.')
    parser.add_argument('-lstm_layers', help='number of LSTM layers the model should have', default='3')
    parser.add_argument('-lstm_units', help='number of units in each LSTM layer', default='128')
    parser.add_argument('-griffin_iterations', help='iterations to use in griffin reconstruction', default='60')
    parser.add_argument('-sample_rate', help='sample_rate', default='44100')

    parser.add_argument('-amount_epochs', help='amount_epochs', default='300')
    parser.add_argument('-batch_size', help='batch_size', default='64')

    parser.add_argument('-sequence_length', help='sequence_length', default='40')
    args = parser.parse_args()
    args.sample_rate = '22050'

    args.amount_epochs = 300

    trainer = TrainingHandler(args)

    trainer.demo_on_folder_of_files()