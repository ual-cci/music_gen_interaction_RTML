# Using the Pygame example from https://github.com/xamox/pygame/blob/master/examples/midi.py

import sys
import math

import pygame
import pygame.midi
from pygame.locals import *

class MIDI_Input_Handler(object):
    """
    Mapping of midi signals ...
    """
    # Tested status mapping:
    # these were valid with Win10, nanoPAD2 and pygame 1.9.6
    KEY_PRESS_DOWN_status = 144
    KEY_PRESS_UP_status = 128
    PAD_MOVE_status = 176

    PAD_MOVE_FINISHED_data1 = 16  # at least I think ...
    PAD_MOVE_ONLY_X = 1
    PAD_MOVE_ONLY_Y = 2

    KEYS_RANGE_FIRST_SCENE_data1 = [36, 51]  # 16 pads on the device have 4 scenes
    KEYS_RANGE_ALL_SCENES_data1 = [36, 99]  # including

    # ps: data2 includes the information about the strength of the press - from around 64 (normal press) to 100 (hard press)
    # ps2: "Touch Scale" causes X-Y pad movement being detected as some PAD presses ... we don't want that!

    def __init__(self, device_id, function_to_call_pad_click, function_to_call_xy_pad, verbose = 0):
        self.device_id = device_id
        self.xy_pad_x = 0
        self.xy_pad_y = 0

        self.xy_pad_delta_x = 0
        self.xy_pad_delta_y = 0

        self.function_to_call_pad_click = function_to_call_pad_click
        self.function_to_call_xy_pad = function_to_call_xy_pad

        self.verbose = verbose

        self.prepare()

    def prepare(self):
        pygame.init()
        pygame.fastevent.init()
        self.event_get = pygame.fastevent.get
        self.event_post = pygame.fastevent.post

        pygame.midi.init()

        _print_device_info()

        if self.device_id is None:
            input_id = pygame.midi.get_default_input_id()
        else:
            input_id = self.device_id

        print("using input_id :%s:" % input_id)
        self.i = pygame.midi.Input(input_id)
        pygame.display.set_mode((1, 1))

    def __del__(self):
        del self.i
        pygame.midi.quit()

    def input_loop(self):
        # MIDI demo:
        print("Reading incoming signals / keypresses")

        self.going = True
        while self.going:
            events = self.event_get()
            for e in events:
                if e.type in [pygame.midi.MIDIIN]:
                    self.filter_key(incoming_event=e)

            if self.i.poll():
                midi_events = self.i.read(10)
                # convert them into pygame events.
                midi_evs = pygame.midi.midis2events(midi_events, self.i.device_id)

                for m_e in midi_evs:
                    self.event_post(m_e)


    def filter_key(self, incoming_event):
        #print("-called filter_key")

        status = incoming_event.status
        data1 = incoming_event.data1
        data2 = incoming_event.data2
        data3 = incoming_event.data3

        # print(incoming_event)
        if status == self.KEY_PRESS_DOWN_status:
            # Pad press
            even = (data1 % 2) == 0
            pad_to_number = math.floor((data1 - self.KEYS_RANGE_FIRST_SCENE_data1[0]) / 2) + 1

            if self.verbose > 0:
                print("PAD PRESS (even:", even,")", pad_to_number, " ", incoming_event)
            self.function_to_call_pad_click(incoming_event, pad_to_number, even)
            """
            if even:
                print("PAD PRESS bottom row, SAVE AS ", pad_to_number, " ", incoming_event)


                # DEMO FUNCTIONALITY:
                percentage = pad_to_number * 100 # HAX DEMO

                from oscpy.client import OSCClient
                address = "127.0.0.1"
                port = 8008
                self.osc = OSCClient(address, port, encoding='utf8')

                model_i=0 # defaults ... might need better handling ...
                requested_lenght=32
                change_speed=80
                volume=100
                print("Sending message=", [percentage, model_i, 0, requested_lenght, change_speed, volume])
                self.osc.send_message(b'/send_i', [percentage, model_i, 0, requested_lenght, change_speed, volume])
                # END

            else:
                print("PAD PRESS up row, LOAD FROM ", pad_to_number, " ", incoming_event)
            """

        elif status == self.PAD_MOVE_status and data1 != self.PAD_MOVE_FINISHED_data1:
            # data2 goes from 0 to 127 for both axes - x and y
            # X-Y touch pad event (PAD_MOVE_FINISHED_data1 is leaving your finger from the pad)
            if data1 == self.PAD_MOVE_ONLY_X:
                # print("X-Y PAD x-axis:", incoming_event)
                previous = self.xy_pad_x
                self.xy_pad_x = pad_xy_to_plusminus_one_range_float(data2)
                self.xy_pad_delta_x = self.xy_pad_x - previous

            if data1 == self.PAD_MOVE_ONLY_Y:
                # print("X-Y PAD y-axis:", incoming_event)
                previous = self.xy_pad_y
                self.xy_pad_y = pad_xy_to_plusminus_one_range_float(data2)
                self.xy_pad_delta_y = self.xy_pad_y - previous

            if self.verbose > 0:
                print("X-Y PAD location: xy=", round(self.xy_pad_x, 2), round(self.xy_pad_y, 2), " ... deltas xy=", round(self.xy_pad_delta_x, 2), round(self.xy_pad_delta_y, 2))

            xy = self.xy_pad_x, self.xy_pad_y, self.xy_pad_delta_x, self.xy_pad_delta_y
            self.function_to_call_xy_pad(incoming_event, xy)

        # else: # ignored ...
        #    print("-------?:", incoming_event)

    def print_device_info(self):
        print_device_info()

def pad_xy_to_plusminus_one_range_float(xy_pad_value):
    # 0 to 127 .. -63.5 to 63.5 .. -1.0 to 1.0
    return (float(xy_pad_value - 63.5) / 63.5)

def pad_xy_to_zeroone_range_float(xy_pad_value):
    # 0 to 127 .. 0.0 to 1.0
    return (float(xy_pad_value) / 127.0)


def print_device_info():
    pygame.midi.init()
    _print_device_info()
    pygame.midi.quit()

def _print_device_info():
    for i in range(pygame.midi.get_count()):
        r = pygame.midi.get_device_info(i)
        (interf, name, input, output, opened) = r

        in_out = ""
        if input:
            in_out = "(input)"
        if output:
            in_out = "(output)"

        print("%5i: interface :%s:, name :%s:, opened :%s:  %s" %
              (i, interf, name, opened, in_out))





if __name__ == '__main__':
    try:
        device_id = int(sys.argv[-1])
    except:
        device_id = None

    print_device_info()

    # on my UBUNTU install:
    import platform
    if 'linux' in platform.system().lower(): # linux / windows
        device_id = 3

    def foo(*a):
        pass

    verbose = 1
    midi_handler = MIDI_Input_Handler(device_id, foo, foo, verbose)
    midi_handler.input_loop()
