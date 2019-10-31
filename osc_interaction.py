# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html#trackbar
# https://github.com/kivy/oscpy

import cv2
import numpy as np
from oscpy.client import OSCClient

class OSCSender(object):
    """
    Sends OSC messages from GUI
    """

    def onChangeSend(self,x):

        self.percentage = cv2.getTrackbarPos('Percentage', 'InteractiveMusicGeneration')
        self.model_i = cv2.getTrackbarPos('Model', 'InteractiveMusicGeneration')
        self.song_i = cv2.getTrackbarPos('Song as seed', 'InteractiveMusicGeneration')
        self.requested_lenght = cv2.getTrackbarPos('Length', 'InteractiveMusicGeneration')

        self.update_text()

        print("Sending message=", [self.percentage, self.model_i, self.song_i, self.requested_lenght])
        self.osc.send_message(b'/send_i', [self.percentage, self.model_i, self.song_i, self.requested_lenght])

    def update_text(self):
        names = self.songs_models.names_for_debug

        self.text = "MODEL="+str(names[self.model_i])+", "+"SONG="+str(names[self.song_i])


    def __init__(self):
        address = "127.0.0.1"
        port = 8008
        self.osc = OSCClient(address, port)
        self.text = ""

        import settings
        import cooked_files_handler

        self.settings = settings.Settings()
        self.songs_models = cooked_files_handler.CookedFilesHandler(self.settings)
        self.songs_models.prepare_songs_models_paths()

    def __del__(self):
        self.stop_osc()

    def stop_osc(self):
        print("Stopping OSC (interactor)")
        self.osc.stop()

    def start_window_rendering(self):

        # Create a black image, a window
        h = 75
        img = np.zeros((h, 512, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (0, h - 25)
        fontScale = 0.8
        fontColor = (128, 128, 128)
        lineType = 2

        cv2.namedWindow('InteractiveMusicGeneration')

        # create trackbars for color change
        cv2.createTrackbar('Percentage', 'InteractiveMusicGeneration', 200, 1000, self.onChangeSend)
        cv2.createTrackbar('Model', 'InteractiveMusicGeneration', 0, len(self.songs_models.model_paths)-1, self.onChangeSend)
        cv2.createTrackbar('Song as seed', 'InteractiveMusicGeneration', 0, len(self.songs_models.song_paths)-1, self.onChangeSend)
        cv2.createTrackbar('Length', 'InteractiveMusicGeneration', 32, 64, self.onChangeSend)

        self.onChangeSend(x=None) # toggle once at start

        while (1):
            # also keep another inf. loop

            cv2.imshow('InteractiveMusicGeneration', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            r = cv2.getTrackbarPos('Percentage', 'InteractiveMusicGeneration')
            r = int(r)

            img[:] = [r, r, r]
            #text = 'Select value: (0 to 1000 => %)'
            text = self.text
            cv2.putText(img, text, position, font, fontScale, fontColor, lineType)

        cv2.destroyAllWindows()


from threading import Thread
# Maybe move this into client?
osc_handler = OSCSender()
thread = Thread(target=osc_handler.start_window_rendering())
thread.start()

