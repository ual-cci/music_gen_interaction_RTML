# LSTM model following the code of https://github.com/Louismac/MAGNet

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell, GRUCell
from tflearn.layers.core import dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from utils.audio_dataset_generator import AudioDatasetGenerator
import sys
import pywt

def conv_net(net, filters, kernels, non_linearity):
    """
    A quick function to build a conv net.
    At the end it reshapes the network to be 3d to work with recurrent units.
    """
    assert len(filters) == len(kernels)

    for i in range(len(filters)):
        net = conv_2d(net, filters[i], kernels[i], activation=non_linearity)
        net = max_pool_2d(net, 2)

    dim1 = net.get_shape().as_list()[1]
    dim2 = net.get_shape().as_list()[2]
    dim3 = net.get_shape().as_list()[3]
    return tf.reshape(net, [-1, dim1 * dim3, dim2])


def recurrent_net(net, rec_type, rec_size, return_sequence):
    """
    A quick if else block to build a recurrent layer, based on the type specified
    by the user.
    """
    if rec_type == 'lstm':
        net = tflearn.layers.recurrent.lstm(net, rec_size, return_seq=return_sequence)
    elif rec_type == 'gru':
        net = tflearn.layers.recurrent.gru(net, rec_size, return_seq=return_sequence)
    elif rec_type == 'bi_lstm':
        net = bidirectional_rnn(net,
                                BasicLSTMCell(rec_size),
                                BasicLSTMCell(rec_size),
                                return_seq=return_sequence)
    elif rec_type == 'bi_gru':
        net = bidirectional_rnn(net,
                                GRUCell(rec_size),
                                GRUCell(rec_size),
                                return_seq=return_sequence)
    else:
        raise ValueError('Incorrect rnn type passed. Try lstm, gru, bi_lstm or bi_gru.')
    return net


class ModelHandlerLSTM(object):
    """
    Will handle everything around the LSTM model.
    """

    def __init__(self):
        self.model = None

        # General Network
        self.learning_rate = 1e-3
        self.amount_epochs = 300
        self.batch_size = 64
        self.keep_prob = 0.2
        self.loss_type = "mean_square"
        self.activation = 'tanh'
        self.optimiser = 'adam'
        self.fully_connected_dim = 1024

        # Recurrent Neural Network
        self.rnn_type = "lstm"
        self.number_rnn_layers = 3
        self.rnn_number_units = 128

        # Convolutional Neural Network
        self.use_cnn = False
        self.number_filters = [32]
        self.filter_sizes = [3]

        # auto init?
        #self.model = create_model()

    def create_model(self, sequence_length = 40):
        self.sequence_length = sequence_length
        #self.sequence_length = 45
        # x data (22983, 40, 1025)
        # y data (22983, 1025)
        self.input_shapes = (None, self.sequence_length, 1025)
        self.output_shapes = (None, 1025)

        # Input
        if self.use_cnn:
            assert False
            #net = tflearn.input_data([None, self.input_shapes[1], self.input_shapes[2], self.input_shapes[3]
            #                          ], name="input_data0")
            #net = conv_net(net, self.number_filters, self.filter_sizes, self.activation)
        else:
            net = tflearn.input_data([None, self.input_shapes[1], self.input_shapes[2]
                                      ], name="input_data0")

            # Batch Norm
        net = tflearn.batch_normalization(net, name="batch_norm0")

        # Recurrent
        for layer in range(self.number_rnn_layers):
            return_sequence = False if layer == (self.number_rnn_layers - 1) else True
            net = recurrent_net(net, self.rnn_type, self.rnn_number_units, return_sequence)
            net = dropout(net, 1 - self.keep_prob) if self.keep_prob < 1.0 else net

            # Dense + MLP Out
        net = tflearn.fully_connected(net, self.output_shapes[1],
                                      activation=self.activation,
                                      regularizer='L2',
                                      weight_decay=0.001)

        net = tflearn.fully_connected(net, self.output_shapes[1],
                                      activation='linear')

        net = tflearn.regression(net, optimizer=self.optimiser, learning_rate=self.learning_rate,
                                 loss=self.loss_type)

        model = tflearn.DNN(net, tensorboard_verbose=1)

        self.model = model

    def generate(self, dataset, amount_samples = 8, sequence_length_max = 700):
        # Dataset
        self.use_wavelets = False
        self.wavelet = 'db10'
        self.window_size = 1024


        impulse_scale = 1.0
        griffin_iterations = 60
        random_chance = 0.8
        random_strength = 0.8

        dimension1 = self.input_shapes[1]
        dimension2 = self.input_shapes[2]
        shape = (1, dimension1, dimension2, 1) if self.use_cnn else (1, dimension1, dimension2)

        audio = []

        if self.use_wavelets:
            temp_audio = np.array(0)
        for i in range(amount_samples):

            random_index = np.random.randint(0, (len(dataset.x_frames) - 1))
            impulse = np.array(dataset.x_frames[random_index]) * impulse_scale
            predicted_magnitudes = impulse

            if self.use_wavelets:
                for seq in range(impulse.shape[0]):
                    coeffs = pywt.array_to_coeffs(impulse[seq], dataset.coeff_slices)
                    recon = (pywt.waverecn(coeffs, wavelet=self.wavelet))
                    temp_audio = np.append(temp_audio, recon)
            for j in range(sequence_length_max):
                prediction = self.model.predict(impulse.reshape(shape))

                # Wavelet audio
                if self.use_wavelets:
                    coeffs = pywt.array_to_coeffs(prediction[0], dataset.coeff_slices)
                    recon = (pywt.waverecn(coeffs, wavelet=self.wavelet))
                    temp_audio = np.append(temp_audio, recon)

                if self.use_cnn:
                    prediction = prediction.reshape(1, self.output_shapes[1], 1)

                # print("prediction.shape", prediction.shape)
                # prediction.shape (1, 1025)

                ### add the last prediction to the predicted_magnitudes
                predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))
                impulse = predicted_magnitudes[-self.sequence_length:]

                # print("impulse.shape", impulse.shape)

                if (np.random.random_sample() < random_chance):
                    idx = np.random.randint(0, self.sequence_length)
                    impulse[idx] = impulse[idx] + np.random.random_sample(impulse[idx].shape) * random_strength

                done = int(float(i * sequence_length_max + j) / float(amount_samples * sequence_length_max) * 100.0) + 1
                sys.stdout.write('{}% audio generation complete.   \r'.format(done))
                sys.stdout.flush()

            if self.use_wavelets:
                audio += [temp_audio]
            else:
                predicted_magnitudes = np.array(predicted_magnitudes).reshape(-1, self.window_size + 1)
                # print("predicted_magnitudes.shape", predicted_magnitudes.shape)
                audio += [dataset.griffin_lim(predicted_magnitudes.T, griffin_iterations)]

        audio = np.array(audio)
        return audio


# Example calls and outputs:
#"""
test_handler = ModelHandlerLSTM()
test_handler.create_model()

model = test_handler.model
print(model)
print(model.net)
print(model.inputs)
print(model.targets)


#<tflearn.models.dnn.DNN object at 0x7f8938542e80>
#Tensor("FullyConnected_1/BiasAdd:0", shape=(?, 1025), dtype=float32)
#[<tf.Tensor 'input_data0/X:0' shape=(?, 40, 1025) dtype=float32>]
#[<tf.Tensor 'TargetsData/Y:0' shape=(?, 1025) dtype=float32>]
#"""

pretrained_path = "/media/vitek/Data/Vitek/Projects/2019_LONDON/music generation/saved_models/trained_model_last___dnb1_300ep_default.tfl"
test_handler.model.load(pretrained_path)

# Second part of the example ...
from utils.audio_dataset_generator import AudioDatasetGenerator
import librosa

dataset = AudioDatasetGenerator(sequence_length=40)
print("loading dataset from a wav file")
audio_data_path = "/media/vitek/Data/Vitek/Projects/2019_LONDON/music generation/small_file/"
#dataset.load(audio_data_path, force=True, prevent_shuffling=False)
dataset.load(audio_data_path, force=True, prevent_shuffling=True)

audio = test_handler.generate(dataset, amount_samples = 2)

print("audio generated, saving")
print("audio", audio.shape)

for sample_i, sample in enumerate(audio):
    sample_rate = 44100
    librosa.output.write_wav('data/sample_i-'+str(sample_i)+'.wav', audio[sample_i], sample_rate)
