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
        args.async_loading = "True"
        self.settings = settings.Settings(args)
        self.settings.print_settings()


    def plot_losses(self, losses, filename):
        fig, ax = plt.subplots()
        print("losses to plot:", losses)

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

    def train_on_file(self, music_file, model_name):
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

        if self.settings.model_name == '': # default naming scheme
            audio_tag = music_file.split("/")[-1].replace(".", "_")
            model_name = "Model_" + audio_tag + "_" + str(self.settings.lstm_layers) + "x" + str(self.settings.lstm_units) + "_"
            model_name = model_name + "sample" + str(int(self.settings.sample_rate/1000)) + "khz_"
            # if griff?
            model_name = model_name + "griff" + str(self.settings.griffin_iterations) + "_"
            model_name = model_name + "_"
            model_name = model_name + "train" + str(self.settings.amount_epochs) + "epX" + str(self.settings.batch_size) + "bt_"
            model_name = model_name + "_" + str(self.settings.sequence_length)+"seq" + ".tfl"
        else:
            model_name = model_name + ".tfl"

        print("Model = |", model_name, "|")


        # Train!
        losses = []
        monitorCallback = TrainingMonitorCallback(losses)

        model_handler.model.fit(dataset.x_frames, dataset.y_frames, show_metric=True, batch_size=model_handler.batch_size,
                                n_epoch=model_handler.amount_epochs, callbacks=[monitorCallback])

        # Save plot
        print("report >>>", losses)
        print("report >>>", monitorCallback.record)
        self.plot_losses(monitorCallback.record, model_name)

        # Save model
        model_handler.model.save(model_name)

        # Save settings
        self.settings.save_into_txt(model_name)

        print("Trained ", model_name,  "successfully ...")

        # Save samples
        if self.settings.sample_after_training > 0:
            self.sample(model_handler, dataset, model_name, n_samples = self.settings.samples_after_training)

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

    def demo_on_file(self, target_file, model_name):
        print("[[[[ Training on music file", target_file, " - will save the model as ", model_name)

        new_graph = tf.Graph()
        with new_graph.as_default():
            self.train_on_file(target_file, model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TrainerHandler for Project: Real Time Audio Generation.')
    parser.add_argument('-target_file', help='path to the wav file', default='__custom_music_samples/sample/sample.wav')
    parser.add_argument('-model_name', help='where to save the model (if left empty it will use the default naming scheme)', default='')

    parser.add_argument('-lstm_layers', help='number of LSTM layers the model should have (default and suggested value, 3)', default='3')
    parser.add_argument('-lstm_units', help='number of units in each LSTM layer (default and suggested value, 128)', default='128')
    parser.add_argument('-griffin_iterations', help='iterations to use in griffin reconstruction; lower number faster and lower quality of reconstructed signal (default value, 60)', default='60')
    parser.add_argument('-sample_rate', help='sampling rate under which we represent the music data (default and suggested value, 22050)', default='22050')

    parser.add_argument('-amount_epochs', help='number of epochs for training the LSTM model (default and suggested value, 300)', default='300')
    parser.add_argument('-batch_size', help='batch size for number of frames that the LSTM model is training on; lower number will lead to lower GPU memory requirements during training (default and suggested value, 64)', default='64')

    parser.add_argument('-sequence_length', help='sequence length of each block of data when training on the task to predict the next single frame from this block of data (default and suggested value, 40)', default='40')
    parser.add_argument('-gensamples', help='how many samples we immediately generate samples from the trained models (default 0)', default='0')
    args = parser.parse_args()

    trainer = TrainingHandler(args)

    trainer.settings.samples_after_training = int(args.gensamples)

    trainer.demo_on_file(args.target_file, args.model_name)
