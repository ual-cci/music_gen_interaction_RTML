import sys
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
# PyQt5 version:
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from oscpy.client import OSCClient
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QSlider

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html#trackbar
# https://github.com/kivy/oscpy


class GUI_OSC(QWidget):
    def __init__(self, parent=None):
        super(GUI_OSC, self).__init__(parent)
        self.setWindowTitle("Interactive Music Generation")
        self.font_size = 14

        address = "127.0.0.1"
        port = 8008
        self.osc = OSCClient(address, port, encoding='utf8')
        self.text = ""

        import settings
        import cooked_files_handler

        self.settings = settings.Settings()
        self.songs_models = cooked_files_handler.CookedFilesHandler(self.settings)
        self.songs_models.prepare_songs_models_paths()

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
        percentage, self.percentage_slider = self.add_slider("Relative position in audio:", style, value=200, maximum=1000)
        length, self.length_slider = self.add_slider("Length:", style, value=32, maximum=124, minimum=4)
        change_speed, self.change_speed_slider = self.add_slider("Transition speed:", style, value=80, maximum=200)
        volume, self.volume_slider = self.add_slider("Volume:", style, value=100, maximum=300)

        layout.addLayout(percentage)
        layout.addLayout(length)
        layout.addLayout(change_speed)
        layout.addLayout(volume)

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


    def onChangeSend(self, i):

        percentage = self.percentage_slider.value()
        requested_lenght = self.length_slider.value()
        change_speed = self.change_speed_slider.value()
        volume = self.volume_slider.value()
        model_i = self.model_select.currentIndex()

        print("Sending message=", [percentage, model_i, 0, requested_lenght, change_speed, volume])
        self.osc.send_message(b'/send_i', [percentage, model_i, 0, requested_lenght, change_speed, volume])

    def recording_toggle(self):
        self.recButton_state = not self.recButton_state
        if self.recButton_state:
            #print("button pressed - start recording!")
            self.osc.send_message(b'/record', [0, ""])

        else:
            #print("button released - end recording + save into |"+self.file_textbox.text()+"|")
            self.osc.send_message(b'/record', [1, self.file_textbox.text()])



def main():
    app = QApplication(sys.argv)
    ex = GUI_OSC()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
