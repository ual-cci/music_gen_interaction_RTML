import sys, os
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
# PyQt5 version:
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from oscpy.client import OSCClient
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QSlider
from midi_input_handler import MIDI_Input_Handler, print_device_info
import threading, time
import json
from gui_functions import DoubleSlider

# OSC: https://github.com/kivy/oscpy
# MIDI: https://github.com/xamox/pygame/blob/master/examples/midi.py

class GUI_OSC(QWidget):


    MIDI_MODE_XY_PAD_REACTION = "MovePercentageALittleBit" # nudge by swiping
    # Not very sensitive:
    #MIDI_MODE_XY_PAD_REACTION = "MovePercentageMappedLeftToRight" # sides left=0, right=1000


    DEFAULT_POSITION = 200
    MIN_POSITION = 0
    MAX_POSITION = 1000

    DEFAULT_LENGTH = 64 #32
    DEFAULT_CHANGESPEED = 80
    DEFAULT_VOLUME = 100
    # default selected song i = 0

    def __init__(self, parent=None):
        super(GUI_OSC, self).__init__(parent)
        self.setWindowTitle("Interactive Music Generation")
        self.font_size = 14

        self.verbose = 1 # 2 = too much, 1 = enough, 0 will be silent
        self.last = None

        # Init OSC - output
        address = "127.0.0.1"
        port = 8008
        self.osc = OSCClient(address, port, encoding='utf8')
        self.text = ""

        # Init MIDI controller - input
        device_id = 1
        import platform
        if 'linux' in platform.system().lower():  # linux / windows
            device_id = 3

        print_device_info()
        function_to_call_pad_click = self.midi_bound_pad_click
        function_to_call_xy_pad = self.midi_bound_xy_pad
        midi_controller = MIDI_Input_Handler(device_id, function_to_call_pad_click, function_to_call_xy_pad)

        threading.Thread(target=midi_controller.input_loop,args=[]).start()
        print("Initiated midi controller! device_id=", device_id)

        # empty saved positions for midi bound clicks
        self.saved_positions = {}
        self.load_midi_positions()

        import settings
        import cooked_files_handler

        self.settings = settings.Settings()
        self.songs_models = cooked_files_handler.CookedFilesHandler(self.settings)
        self.songs_models.prepare_songs_models_paths()

        # Init on screen GUI - input
        layout = QVBoxLayout()
        style = "QWidget {font-size: "+str(self.font_size)+"pt;}"

        hbox_model_select = QHBoxLayout()
        text_model_select = QLabel()
        text_model_select.setText("Model selection:")
        text_model_select.setStyleSheet(style)
        model_select = QComboBox()
        for model_path in self.songs_models.model_paths:
            just_model = model_path.split("/")[-1]
            just_model = just_model[0:min(len(just_model), 30)]
            model_select.addItem(just_model)

        self.model_select = model_select
        model_select.currentIndexChanged.connect(self.onChangeSend) # <<< On change function
        font = model_select.font()
        font.setPointSize(self.font_size)
        model_select.setFont(font)

        hbox_model_select.addWidget(text_model_select)
        hbox_model_select.addWidget(model_select)

        layout.addLayout(hbox_model_select)


        # Sliders
        percentage, self.percentage_slider = self.add_slider("Relative position in audio:", style, value=self.DEFAULT_POSITION, maximum=1000)
        length, self.length_slider = self.add_slider("Length:", style, value=self.DEFAULT_LENGTH, maximum=124, minimum=4)
        change_speed, self.change_speed_slider = self.add_slider("Transition speed:", style, value=self.DEFAULT_CHANGESPEED, maximum=200)
        volume, self.volume_slider = self.add_slider("Volume:", style, value=self.DEFAULT_VOLUME, maximum=300)
        weights_multiplier, self.weights_multiplier = self.add_double_slider("Weights Multiplier:", style, minimum=-1, value=1, maximum=2)

        layout.addLayout(percentage)
        layout.addLayout(length)
        layout.addLayout(change_speed)
        layout.addLayout(volume)
        layout.addLayout(weights_multiplier)

        # Record button
        hbox_record = QHBoxLayout()
        recButton = QPushButton("Save!")
        recButton.setStyleSheet(style)
        hbox_record.addWidget(recButton)
        recButton.setCheckable(True)


        file_textbox = QLineEdit()
        file_textbox.setText("tmp_recording1")
        hbox_record.addWidget(file_textbox)

        self.file_textbox = file_textbox
        self.recButton = recButton
        self.recButton_state = False
        recButton.clicked.connect(self.recording_toggle)

        layout.addLayout(hbox_record)

        self.setLayout(layout)


    def add_slider(self, txt, style, value = 0, minimum = 0, maximum = 100):
        layout = QHBoxLayout()
        text = QLabel()
        text.setText(txt)
        text.setStyleSheet(style)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)


        val = QLabel()
        val.setNum(value)
        val.setStyleSheet(style)

        slider.valueChanged.connect(val.setNum)
        slider.valueChanged.connect(self.onChangeSend)

        layout.addWidget(text)
        layout.addWidget(val)
        layout.addWidget(slider)
        return layout, slider

    def add_double_slider(self, txt, style, value = 0, minimum = 0, maximum = 100, decimals=3):
        layout = QHBoxLayout()
        text = QLabel()
        text.setText(txt)
        text.setStyleSheet(style)
        slider = DoubleSlider(decimals, Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)


        val = QLabel()
        val.setNum(value)
        val.setStyleSheet(style)

        slider.doubleValueChanged.connect(val.setNum)
        slider.doubleValueChanged.connect(self.onChangeSend)

        layout.addWidget(text)
        layout.addWidget(val)
        layout.addWidget(slider)
        return layout, slider

    def onChangeSend(self, i):

        percentage = self.percentage_slider.value()
        requested_lenght = self.length_slider.value()
        change_speed = self.change_speed_slider.value()
        volume = self.volume_slider.value()
        model_i = self.model_select.currentIndex()
        weights_multiplier = self.weights_multiplier.value()

        ### NOT SURE IF FLOAT IS OK ???
        weights_multiplier = int(100.0 * weights_multiplier)

        message = [percentage, model_i, 0, requested_lenght, change_speed, volume, weights_multiplier]

        # Send only if we actually change some values:
        changed = False
        if self.last is not None:
            for idx in range(len(message)):
                if message[idx] != self.last[idx]:
                    changed = True
                    break
        else:
            changed = True
        self.last = message

        if changed:
            print("Sending message=", message)
            self.osc.send_message(b'/send_i', message)

    def recording_toggle(self):
        self.recButton_state = not self.recButton_state
        if self.recButton_state:
            #print("button pressed - start recording!")
            self.osc.send_message(b'/record', [0, ""])

        else:
            #print("button released - end recording + save into |"+self.file_textbox.text()+"|")
            self.osc.send_message(b'/record', [1, self.file_textbox.text()])

    # Functions to be bound to MIDI controller events:
    #function_to_call_pad_click, function_to_call_xy_pad

    def midi_bound_pad_click(self, event, pad_number, to_save):
        pad_number = str(pad_number)

        if to_save:
            percentage_slider_value = self.percentage_slider.value()
            print("Save position",percentage_slider_value,"as", pad_number)
            self.saved_positions[pad_number] = percentage_slider_value

            self.save_midi_positions()

        else:
            if pad_number in self.saved_positions:
                new_value = self.saved_positions[pad_number]
                print("Load from position", pad_number, "value", new_value)

                self.percentage_slider.setValue(new_value)
            else:
                print("Load from empty position", pad_number)

    def midi_bound_xy_pad(self, event, xy):
        xy_pad_x, xy_pad_y, xy_pad_delta_x, xy_pad_delta_y = xy

        if self.verbose > 1:
            print("Detected X-Y PAD movement: xy=", round(xy_pad_x, 2), round(xy_pad_y, 2), " ... deltas xy=", round(xy_pad_delta_x, 2), round(xy_pad_delta_y, 2))

        # x movement to percentage position?
        # change GUI element position - this will also trigger an event

        if self.MIDI_MODE_XY_PAD_REACTION == "MovePercentageALittleBit":
            old_value = float(self.percentage_slider.value()) # from 0 to 1000
            print(old_value)
            move_scale = 100.0
            move_by = xy_pad_delta_x * move_scale # delta is usually +-0.05 => +- 5?
            new_value = int(max(0, min(old_value + move_by, self.MAX_POSITION)))

            print("called change from ", old_value, "to", new_value)
            self.percentage_slider.setValue(new_value)
        elif self.MIDI_MODE_XY_PAD_REACTION == "MovePercentageMappedLeftToRight":
            scale = (1.0 + xy_pad_x) / 2.0 # now in 0.0 to 1.0
            new_value = int(scale * self.MAX_POSITION) # now this goes from 0 to 1000

            self.percentage_slider.setValue(new_value)

    # Keep MIDI controls saved between runs!
    def save_midi_positions(self):
        with open('midi_saved_positions.json', 'w') as fp:
            json.dump(self.saved_positions, fp)

    def load_midi_positions(self):
        if os.path.isfile('midi_saved_positions.json'):
            with open('midi_saved_positions.json', 'r') as fp:
                self.saved_positions = json.load(fp)

            print("Loaded", len(self.saved_positions.keys()), "saved positions from before:")
            print(self.saved_positions)


def main():
    app = QApplication(sys.argv)
    ex = GUI_OSC()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
