import numpy as np
from keras.models import load_model
import numpy as np


def AE_data_transform(train):
    # with zero padding
    train = np.reshape(train, train.shape+(1,))
    train_shape = list(train.shape) # (10359, 1025, 1)
    train_shape[1] = 1028
    zeropad = np.zeros(train_shape)
    zeropad[:train.shape[0],:train.shape[1]] = train
    train = zeropad
    return train


def VAE_data_transform(train):
    cut_val = 1020

    train = train[:, 0:cut_val]
    train = np.reshape(train, train.shape + (1,))

    return train

def VAE_data_reconstruct(train):
    # data in (N, 1020, 1)
    # back to (N, 1028, 1)
    train_shape = (len(train), 1025, 1)
    zeropad = np.zeros(train_shape)
    zeropad[:train.shape[0],:train.shape[1]] = train
    train = zeropad[:,:,0]
    return train

def load_AE():
    #### problem ... used tf's Keras ... duh
    ae_encoder = load_model('autoencoder/saved_model_4-goodfitIhope-40ep/encoder.h5')
    ae_encoder.summary()
    return ae_encoder


def load_VAE():
    vae_encoder = load_model('autoencoder/saved_model_vae_2/encoder.h5')
    vae_encoder.summary()
    vae_decoder = load_model('autoencoder/saved_model_vae_2/decoder.h5')
    vae_decoder.summary()
    return vae_encoder, vae_decoder

vae_enc, vae_dec = load_VAE()

def to_latent_vector_using_VAE(data):
    global vae_enc, vae_dec
    #sample = np.random.rand(20, 1025)
    sample_ed = VAE_data_transform(data)
    z_mean, z_log_var, z = vae_enc.predict(sample_ed)
    return z


def process_using_VAE(data):
    global vae_enc, vae_dec
    #sample = np.random.rand(20, 1025)
    #print("in", sample.shape)

    sample_ed = VAE_data_transform(data)
    #print("in cut", sample_ed.shape)
    z_mean, z_log_var, z = vae_enc.predict(sample_ed)
    #print("mid", z.shape)

    out = vae_dec.predict(z)
    #print("out", out.shape)

    foo = VAE_data_reconstruct(out)
    #print("foo", foo.shape)

    #diff = sample - foo
    #print(np.min(diff), np.max(diff), np.mean(diff), np.std(diff))
    return foo

#data = np.random.rand(20, 1025)
#reconstruction = process_using_VAE(data)

