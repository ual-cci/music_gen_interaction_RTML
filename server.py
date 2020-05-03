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
import settings

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

# Production vs development - not entirely sure what the differences are, except for debug outputs
PRODUCTION = True # waitress - production-quality pure-Python WSGI server with very acceptable performance
PRODUCTION = False # Flask

SERVER_VERBOSE = 1  # 2 = all messages, 1 = only important ones, 0 = none!

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

        self.settings = settings.Settings(args)
        self.load_serverside_handler()

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
            app.run(threaded=False) # < with forbiding threaded we still have the same default graph

    def mem_monitor_deamon(self, frequency_sec):
        import subprocess
        while (True):
            out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                                   stdout=subprocess.PIPE).communicate()[0].split(b'\n')
            vsz_index = out[0].split().index(b'RSS')
            mem = float(out[1].split()[vsz_index]) / 1024

            print("Memory:", mem)
            time.sleep(frequency_sec)  # check every frequency_sec sec

    def load_serverside_handler(self):
        global serverside_handler
        serverside_handler = server_handler.ServerHandler(self.settings)
        serverside_handler.VERBOSE = SERVER_VERBOSE
        print('Server handler loaded.')

        #### SPECIAL CODE
        from resender import resender_function
        serverside_handler.extended_functionality_function = resender_function



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
        DEFAULT_interactive_i = 0.0
        DEFAULT_model_i = 0
        DEFAULT_song_i = 0
        DEFAULT_change_speed = 120
        DEFAULT_weights_multiplier = 1.0
        weights_multiplier = DEFAULT_weights_multiplier
        requested_length = DEFAULT_lenght
        interactive_i = DEFAULT_interactive_i
        model_i = DEFAULT_model_i
        song_i = DEFAULT_song_i
        change_speed = DEFAULT_change_speed



        if len(flask.request.files) and SERVER_VERBOSE > 1:
            print("Recieved flask.request.files = ",flask.request.files)

        try:
            requested_length = flask.request.files["requested_length"].read()
            interactive_i = flask.request.files["interactive_i"].read()
            model_i = flask.request.files["model_i"].read()
            song_i = flask.request.files["song_i"].read()
            change_speed = flask.request.files["change_speed"].read()
            weights_multiplier = flask.request.files["weights_multiplier"].read()

            if SERVER_VERBOSE > 1:
                print("received: ",requested_length, interactive_i, model_i, song_i, weights_multiplier)

            requested_length = int(requested_length)
            interactive_i = float(interactive_i)
            weights_multiplier = float(weights_multiplier)
            model_i = int(model_i)
            song_i = int(song_i)
            change_speed = int(change_speed)

        except Exception as e:
            print("failed to parse the flask.request.files!", e)

        print("Server will generate audio of requested length",requested_length,". Interactive i=",interactive_i)

        t_decode_end = timer()

        global serverside_handler

        current_model_i = serverside_handler.model_i
        current_song_i = serverside_handler.song_i
        current_interactive_i = serverside_handler.interactive_i
        current_weights_multiplier = serverside_handler.weights_multiplier

        if not serverside_handler._is_changing_impulses:
            serverside_handler._change_steps = change_speed

        if SERVER_VERBOSE > 1:
            print("current_model_i=",current_model_i)
            print("current_song_i=",current_song_i)
            print("current_interactive_i=",current_interactive_i)
            print("current_weights_multiplier=",current_weights_multiplier)
            print("sent model_i=",model_i)
            print("sent song_i=",song_i)
            print("sent interactive_i=",interactive_i)
            print("sent weights_multiplier=",weights_multiplier)

        # Perhaps do this differently ... so it doesn't get the server stuck!
        if song_i != current_song_i:
            print("Loading new song data! (",song_i,")")
            t_load_song = timer()
            #serverside_handler.load_impulses(song_i=song_i)
            serverside_handler.load_impulses_ASYNC(song_i=song_i)
            t_load_song = timer() - t_load_song
            print("Loading took = ", t_load_song, "sec")

        if model_i != current_model_i:
            print("Loading new model weights! (",model_i,")")
            t_load_model = timer()
            ##serverside_handler.model_handler.create_model() # << REDO EVERYTHING? Hope that it wont be needed
            #serverside_handler.load_weights(model_i=model_i)
            serverside_handler.load_weights_ASYNC(model_i=model_i)
            t_load_model = timer() - t_load_model
            print("Loading took = ", t_load_model, "sec")

        print(">>>>>>>>", current_model_i)


        if not serverside_handler.continue_impulse_from_previous_batch or (interactive_i != current_interactive_i):
            # Either change impulse every generation - or when it was changed
            print("Start with a new impulse:")

            if not serverside_handler.change_impulses_smoothly or serverside_handler.first_iteration or (serverside_handler._change_steps is 0):
                serverside_handler.change_impulse(interactive_i)
            else:
                serverside_handler.change_impulse_smoothly_start(interactive_i)

        # weights_multiplier
        if weights_multiplier != 1.0 and weights_multiplier != current_weights_multiplier:
            print("Neural Network weights adjustment! Set weights_multiplier=",weights_multiplier)

            # LSTM has shape (1153, 512)
            # LSTM_1 has shape (256, 512)
            # LSTM_2 has shape (256, 512)
            # FullyConnected has shape (128, 1025)
            # FullyConnected_1 has shape (1025, 1025)
            if False: # if old model:
                target_tensor_name = "LSTM_1"
                operation = serverside_handler.model_handler.times_a
                serverside_handler.change_lstm_net(target_tensor_name, operation, weights_multiplier)

            # if new model:
            with serverside_handler.graph.as_default():
                target_tensor_name="LSTM_1"
                operation = serverside_handler.model_handler.times_a
                serverside_handler.change_lstm_net(target_tensor_name, operation, weights_multiplier)

            # then save it
            serverside_handler.weights_multiplier = weights_multiplier

        # hard code (for now) if we are using GriffLim or LWS for reconstruction
        method = "Griff"

        #print("serverside_handler.songs_models[current_model_i]", serverside_handler.songs_models.model_paths[current_model_i], "_lws_" in serverside_handler.songs_models.model_paths[current_model_i])
        #if current_model_i > 9: # HAX for 10 and 11
        if "_lws_" in serverside_handler.songs_models.model_paths[current_model_i]:
            method = "LWS"

        audio_arr, t_predict, t_reconstruct = serverside_handler.generate_audio_sample(requested_length, interactive_i, method)

        serverside_handler.first_iteration = False

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

@app.route("/debugMethod", methods=["GET"])
def debugMethod():
    # This just does something I want to test...
    data = {"success": False}
    try:
        global serverside_handler
        serverside_handler.load_weights(model_i=5)

        # indicate that the request was a success
        data["success"] = True
    except Exception as e:
        print("something went wrong!", e)

    as_json = flask.jsonify(data)
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
    parser.add_argument('-sequence_length', help='sequence_length', default='40')
    args = parser.parse_args()

    args.sample_rate = '22050'

    server = Server(args)