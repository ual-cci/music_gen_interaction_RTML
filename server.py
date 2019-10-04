from threading import Thread
import time

from PIL import Image
import flask
import os
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool
import numpy as np
import socket
import cv2
import server_handler

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

# Production vs development - not entirely sure what the differences are, except for debug outputs
PRODUCTION = True # waitress - production-quality pure-Python WSGI server with very acceptable performance
PRODUCTION = False # Flask

SERVER_VERBOSE = 2  # 2 = all messages, 1 = only important ones, 0 = none!

if PRODUCTION:
    from waitress import serve

app = flask.Flask(__name__)
serverside_handler = None
pool = ThreadPool()


class Server(object):
    """
    Server
    """

    def __init__(self, args):
        print("Server ... starting server and loading model ... please wait until its started ...")

        self.load_serverside_handler(args)

        frequency_sec = 10.0
        if SERVER_VERBOSE > 0:
            t = Thread(target=self.mem_monitor_deamon, args=([frequency_sec]))
            t.daemon = True
            t.start()

        # hack to distinguish server by hostnames
        hostname = socket.gethostname()  # gpu048.etcetcetc.edu
        print("server hostname is", hostname)

        if PRODUCTION:
            serve(app, host='127.0.0.1', port=5000)
        else:
            app.run()

    def mem_monitor_deamon(self, frequency_sec):
        import subprocess
        while (True):
            out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                                   stdout=subprocess.PIPE).communicate()[0].split(b'\n')
            vsz_index = out[0].split().index(b'RSS')
            mem = float(out[1].split()[vsz_index]) / 1024

            print("Memory:", mem)
            time.sleep(frequency_sec)  # check every frequency_sec sec

    def load_serverside_handler(self, args):
        global serverside_handler
        serverside_handler = server_handler.ServerHandler(args)
        print('Server handler loaded.')



@app.route("/handshake", methods=["POST"])
def handshake():
    # Handshake

    data = {"success": False}
    start = timer()

    if flask.request.method == "POST":
        if flask.request.files.get("client"):
            client_message = flask.request.files["client"].read()
            print("Handshake, received: ",client_message)

            backup_name = flask.request.files["backup_name"].read()
            # try to figure out what kind of server we are, what is our name, where do we live, what are we like,
            # which gpu we occupy
            # and return it at an identifier to the client ~

            try:
                hostname = socket.gethostname() # gpu048.etcetcetc.edu
                machine_name = hostname.split(".")[0]
                data["server_name"] = machine_name
            except Exception as e:
                data["server_name"] = backup_name

            end = timer()
            data["internal_time"] = end - start
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route("/get_audio", methods=["POST"])
def get_audio():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        t_decode_start = timer()

        DEFAULT_lenght = 1024

        if len(flask.request.files) and SERVER_VERBOSE > 1:
            print("Recieved flask.request.files = ",flask.request.files)

        try:
            requested_length = flask.request.files["requested_length"].read()
            interactive_i = flask.request.files["interactive_i"].read()
            print("received: ",requested_length, interactive_i)

            requested_length = int(requested_length)
            interactive_i = float(interactive_i)

        except Exception as e:
            print("failed to read the requested_length", e)
            requested_length = DEFAULT_lenght

        print("Server will generate audio of requested length",requested_length,".")

        t_decode_end = timer()

        global serverside_handler
        audio_arr, t_predict, t_reconstruct = serverside_handler.generate_audio_sample(requested_length)
        data["audio_response"] = audio_arr.tolist()
        data["time_predict"] = t_predict
        data["time_reconstruct"] = t_reconstruct
        data["time_decode"] = t_decode_end-t_decode_start


        t_server_end = timer()
        time_server_total = t_server_end - t_decode_start
        print("time server total =",time_server_total)
        data["time_server_total"] = time_server_total

        # indicate that the request was a success
        data["success"] = True

    t_to_jsonify = timer()
    as_json = flask.jsonify(data)
    t_to_jsonify = timer() - t_to_jsonify
    if SERVER_VERBOSE > 1:
        print("JSONify took", t_to_jsonify, "sec.")

    return as_json

def get_gpus_buses():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x for x in local_device_protos if x.device_type == 'GPU']
    buses = ""
    for device in gpu_devices:
        desc = device.physical_device_desc # device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:81:00.0
        bus = desc.split(",")[-1].split(" ")[-1][5:] # split to get to the bus information
        bus = bus[0:2] # idk if this covers every aspect of gpu bus
        if len(buses)>0:
            buses += ";"
        buses += str(bus)
    return buses

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Project: Real Time Audio Generation.')
    parser.add_argument('-lstm_layers', help='number of LSTM layers the model should have', default='3')
    parser.add_argument('-lstm_units', help='number of units in each LSTM layer', default='128')
    parser.add_argument('-griffin_iterations', help='iterations to use in griffin reconstruction', default='60')
    parser.add_argument('-sample_rate', help='sample_rate', default='44100')
    args = parser.parse_args()

    args.sample_rate = '22050'

    server = Server(args)