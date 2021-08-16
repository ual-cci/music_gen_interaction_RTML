
"""
def resender_function(arr):
    pass
"""

from autoencoder.autoencoders_handler import process_using_VAE, to_latent_vector_using_VAE

def resender_function(arr):
    # Send one frame only - the last generated one
    #send_one_frame(arr)
    send_all_gen_frames(arr)
    #encode_spectrogram_VAE(arr)

    #arr = process_spectrogram_VAE(arr)

    # return spectrogram ...
    return arr

def process_spectrogram_VAE(arr):
    print("processing")
    processed = process_using_VAE(arr)
    print("HAX OUTPUT === ", processed.shape)
    return processed

def encode_spectrogram_VAE(arr):
    print("encoding")
    latents = to_latent_vector_using_VAE(arr)
    print("HAX OUTPUT === ", latents.shape)

    sequence_length = 40
    to_send = latents #[sequence_length:]
    for count, one_frame in enumerate(to_send):

        print("HAX OUTPUT === ", one_frame.shape)
        if count % 4 == 0:
            osc_handler.send_arr(one_frame)


def send_one_frame(arr):
    last_spectum = arr[-1]

    low_frequencies = last_spectum[0:512]

    #print("HAX OUTPUT === ", low_frequencies.shape)

    global osc_handler
    osc_handler.send_arr(low_frequencies)


def send_all_gen_frames(arr):
    # settings/server sequence_length by default on 40
    sequence_length = 40
    global osc_handler
    to_send = arr[sequence_length:]
    for count, one_frame in enumerate(to_send):
        
        low_frequencies = one_frame[0:512]

        #print("HAX OUTPUT === ", low_frequencies.shape)
        if count % 4 == 0:
            osc_handler.send_arr(low_frequencies)


# https://github.com/kivy/oscpy
from oscpy.client import OSCClient

class OSCSender(object):
    """
    Sends OSC messages from GUI
    """

    def send_arr(self,arr):
        signal_latent = arr
        signal_latent = [float(v) for v in signal_latent]

        print("Sending message=", [0, 0, len(signal_latent)])
        self.osc.send_message(b'/send_gan_i', [0, 0] + signal_latent)

    def __init__(self):
        #address = "127.0.0.1"
        #port = 8000
        address = '0.0.0.0'
        port = 8000
        self.osc = OSCClient(address, port)

osc_handler = OSCSender()
