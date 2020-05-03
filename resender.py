
"""
def resender_function(arr):
    pass
"""

def resender_function(arr):
    last_spectum = arr[-1]

    low_frequencies = last_spectum[0:512]

    print("HAX OUTPUT === ", low_frequencies.shape)

    global osc_handler
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