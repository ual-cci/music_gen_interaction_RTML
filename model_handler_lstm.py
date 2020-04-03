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
    Can create a new model architecture. It's model can be trained, loaded, used for generation/prediction.
    (Shouldn't have any audio processing specific code though.)

    - create a model
    - predict

    """

    def __init__(self, number_rnn_layers, rnn_number_units, settings = None):
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
        self.number_rnn_layers = number_rnn_layers
        self.rnn_number_units = rnn_number_units

        # Convolutional Neural Network
        self.use_cnn = False
        self.number_filters = [32]
        self.filter_sizes = [3]

        if settings is not None:
            self.amount_epochs = settings.amount_epochs
            self.batch_size = settings.batch_size
            self.sequence_length = settings.sequence_length


        self.original_weights = {}

        # auto init?
        #self.model = create_model()

    def load_model(self, path):
        print("Loading a pretrained model")

        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        print("We have", len(tensor_name_list), "tensors loaded in default graph!")
        #for tensor_name in tensor_name_list:
        #    print(tensor_name, '\n')

        self.model.load(path)

    def create_model(self):
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

        #model = tflearn.DNN(net, tensorboard_verbose=1)
        model = tflearn.DNN(net, tensorboard_verbose=0)

        self.model = model

    # Neural Network direct weights editation
    def inspect_tensors(self):
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        print("We have", len(tensor_name_list), "tensors loaded in default graph!")
        for tensor_name in tensor_name_list:
            if "save" not in tensor_name and "Adam" not in tensor_name:
                print(tensor_name, '\n')

    def print_weights_shape(self, target_tensor_name):
        layer_tensor = tflearn.variables.get_layer_variables_by_name(target_tensor_name)
        if len(layer_tensor) > 0:
            weights = self.model.get_weights(layer_tensor[0])
            print(target_tensor_name, "has shape", weights.shape)

    def times_a(self, a, np_arr):
        return a * np_arr

    def change_lstm_net(self, target_tensor_name, operation, *kwargs):
        # restore old weights
        for target_tensor_name in self.original_weights.keys():
            layer_tensor = tflearn.variables.get_layer_variables_by_name(target_tensor_name)
            orig_val = self.original_weights[target_tensor_name]
            self.model.set_weights(layer_tensor[0], orig_val)

        if target_tensor_name not in self.original_weights:
            layer_tensor = tflearn.variables.get_layer_variables_by_name(target_tensor_name)
            weights = self.model.get_weights(layer_tensor[0])
            self.original_weights[target_tensor_name] = weights

        # reload() # causes memory issues
        layer_tensor = tflearn.variables.get_layer_variables_by_name(target_tensor_name)
        weights = self.original_weights[target_tensor_name]

        print("weights:", type(weights), weights.shape)
        modified_weights = operation(weights, kwargs)
        self.model.set_weights(layer_tensor[0], modified_weights)


    def generate_sample(self, input_impulse, requested_length, window_size=1024):

        random_chance = 0.0
        random_strength = 0.8

        dimension1 = self.input_shapes[1]
        dimension2 = self.input_shapes[2]

        next_generation_impulse = None

        predicted_magnitudes = input_impulse
        for j in range(requested_length):
            shape = (1, dimension1, dimension2, 1) if self.use_cnn else (1, dimension1, dimension2)
            prediction = self.model.predict(input_impulse.reshape(shape))

            if self.use_cnn:
                prediction = prediction.reshape(1, self.output_shapes[1], 1)

            # add the last prediction to the predicted_magnitudes
            predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))
            input_impulse = predicted_magnitudes[-self.sequence_length:]

            next_generation_impulse = input_impulse

            # mix in?
            if (np.random.random_sample() < random_chance):
                idx = np.random.randint(0, self.sequence_length)
                input_impulse[idx] = input_impulse[idx] + np.random.random_sample(input_impulse[idx].shape) * random_strength

        predicted_magnitudes = np.array(predicted_magnitudes).reshape(-1, window_size + 1)

        return predicted_magnitudes, next_generation_impulse

    def generate_sample__whileInterpolating(self, input_impulse, target_impulses, _change_step, _change_steps, requested_length, window_size=1024):


        dimension1 = self.input_shapes[1]
        dimension2 = self.input_shapes[2]

        next_generation_impulse = None

        predicted_magnitudes = input_impulse
        for j in range(requested_length):


            shape = (1, dimension1, dimension2, 1) if self.use_cnn else (1, dimension1, dimension2)
            prediction = self.model.predict(input_impulse.reshape(shape))

            if self.use_cnn:
                prediction = prediction.reshape(1, self.output_shapes[1], 1)

            # add the last prediction to the predicted_magnitudes
            predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))
            input_impulse = predicted_magnitudes[-self.sequence_length:]




            # mix in?
            #if (np.random.random_sample() < random_chance):
            #    idx = np.random.randint(0, self.sequence_length)
            #    input_impulse[idx] = input_impulse[idx] + np.random.random_sample(input_impulse[idx].shape) * random_strength

            # Interpolation here
            _change_step += 1
            if _change_step < _change_steps:
                print("We are on step", _change_step, "from", _change_steps)

                # while it's smaller, we should change the impulse:
                alpha = (_change_steps - (_change_step+1))/_change_steps # 0.0 with _change_step==0, 1.0 with _change_step==_change_steps

                print("blending A=",input_impulse.shape, (alpha))
                print("blending B=",target_impulses[_change_step].shape, 1.0-alpha)

                input_impulse = input_impulse*(alpha) + target_impulses[_change_step]*(1.0-alpha)
            else:
                print("no longer changing, reached the transition")

            next_generation_impulse = input_impulse


        predicted_magnitudes = np.array(predicted_magnitudes).reshape(-1, window_size + 1)

        return predicted_magnitudes, next_generation_impulse, _change_step, _change_steps

# Example calls and outputs:
"""
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

pretrained_path = "/media/vitek/Data/Vitek/Projects/2019_LONDON/music generation/saved_models/trained_model_last___dnb1_300ep_default.tfl"
test_handler.model.load(pretrained_path)

"""