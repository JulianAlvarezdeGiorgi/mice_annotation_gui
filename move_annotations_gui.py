import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui
import pandas as pd
import numpy as np
import cv2
import time


NROW = 5
NCOL = 7
NFRAME = NROW * NCOL
NSIDE = NFRAME // 2
BIN = 4
SKIP = 2


class MovieViewer(QMainWindow):
    '''A simple movie viewer application using OpenCV and PyQt5.'''

    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Movie Viewer")
        # self.setGeometry(100, 100, 400, 300)
        self.setGeometry(10, 600, 400, 300)

        # Main widget
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Layout: left column for buttons, the rest for video
        self.main_layout = QtWidgets.QHBoxLayout()  # Horizontal box layout that divides the window into two sections: one for controls (left column) and one for video display (right area).
        self.main_widget.setLayout(self.main_layout)
        self.controls_layout = QtWidgets.QGridLayout()  # A vertical box
        row = 0
        # layout within the left column for buttons and controls.
        self.main_layout.addLayout(self.controls_layout)
        self.video_layout = QtWidgets.QVBoxLayout()  # A vertical box layout within the right area for the video display.
        self.main_layout.addLayout(self.video_layout)

        # OpenCV video capture object
        self.cap = None
        self.frame_number = self.frame_count = None
        self.nx = self.ny = None
        self.frame_rate = None
        self.frame_stack = []

        # OpenCV video display widget
        self.video_widget = QtWidgets.QLabel(self)
        self.video_layout.addWidget(
            self.video_widget)  # Add the video widget to the video layout.

        # Buttons

        # Load Movie button
        self.load_button = QPushButton("Load Movie")
        self.controls_layout.addWidget(self.load_button, row, 0, 1, 2)
        row += 1
        self.load_button.clicked.connect(self.load_movie)

        # Play Movie button
        self.play_button = QPushButton("Play Movie")
        self.play_button.setCheckable(True)
        self.controls_layout.addWidget(self.play_button, row, 0, 1, 2)
        row += 1
        self.play_button.clicked.connect(self.toggle_play)

        # Label for frame number
        self.frame_number_label = QtWidgets.QLabel("Frame:")
        # (increase width to make room for the frame number)
        self.frame_number_label.setFixedWidth(
            self.frame_number_label.fontMetrics().boundingRect("Frame: 00000").width())
        self.controls_layout.addWidget(self.frame_number_label, row, 0)

        # Frame Navigation Controls
        frame_by_frame_layout = QtWidgets.QHBoxLayout()  # A horizontal box layout for the frame-by-frame navigation controls.
        self.controls_layout.addLayout(frame_by_frame_layout, row, 1)
        row += 1

        self.previous_frame_button = QPushButton("<")
        self.controls_layout.addWidget(self.previous_frame_button)
        self.previous_frame_button.clicked.connect(self.previous_frame)
        frame_by_frame_layout.addWidget(self.previous_frame_button)

        self.next_frame_button = QPushButton(">")
        self.controls_layout.addWidget(self.next_frame_button)
        self.next_frame_button.clicked.connect(self.next_frame)
        frame_by_frame_layout.addWidget(self.next_frame_button)

        # add a spacer to the controls layout to push the buttons to the top
        vertical_stretch = QtWidgets.QWidget()
        vertical_stretch.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.controls_layout.addWidget(vertical_stretch, row, 0)
        row += 1

        # Time slider
        self.time_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.video_layout.addWidget(self.time_slider)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.valueChanged.connect(self.set_movie_position)

        # Try to load the last movie file
        try:
            with open('check_pose_gui_last_file.txt', 'r') as file:
                file_path = file.read()
            self.load_movie(file_path)
        except FileNotFoundError:
            pass

    def display_frame(self, frame):
        # display current frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                             QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_widget.setPixmap(pixmap)
        self.video_widget.setScaledContents(True)

    def load_movie(self, file_path=False):
        # attempt to read folder_path from a file
        print('a', file_path)
        if file_path is False:
            try:
                with open('check_pose_gui_last_file.txt', 'r') as file:
                    file_path = file.read()
                    folder_path = os.path.dirname(file_path)
            except FileNotFoundError:
                folder_path = None
            print('b')
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self,
                                                       "Select Movie File",
                                                       folder_path,
                                                       "Movie files (*.mp4 *.avi *.mov)")
            print(file_path)
        if not file_path:
            return
        print('c', file_path)
        # Remember file for next time
        with open('check_pose_gui_last_file.txt', 'w') as file:
            file.write(file_path)

        # Open the video file with OpenCV
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(file_path)

        # Check if the video file was successfully opened
        if not self.cap.isOpened():
            print("Error opening video file")
            return

        # Update slider upper bound
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nx = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ny = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_slider.setMaximum(self.frame_count - 1)

        # Display the first frame
        self.set_frame(0)

    def previous_frame(self):
        try:
            self.set_frame(self.frame_number - 1)
        except Exception as e:
            print(e)

    def next_frame(self):
        try:
            self.set_frame(self.frame_number + 1)
        except Exception as e:
            print(e)

    def toggle_play(self, value):
        if value:
            self.play_button.setText("Stop playing")
            target = time.time() + 1 / self.frame_rate
            while self.play_button.isChecked():
                self.next_frame()
                QtWidgets.QApplication.processEvents()
                time.sleep(max(0, target - time.time()))
        else:
            self.play_button.setText("Play Movie")

    def set_movie_position(self):
        try:
            value = self.time_slider.value()
            self.set_frame(value, update_slider=False)
        except Exception as e:
            print(e)

    def set_frame(self, frame_number, update_slider=True):

        # reading will be faster if we are reading frames sequentially
        do_next_frame = (frame_number - 1 == self.frame_number)

        self.frame_number = frame_number

        frame_start = frame_number - NSIDE
        n_read = NFRAME
        n_miss_left = n_miss_right = 0
        if frame_start < 0:
            frame_start = 0
            n_miss_left = NSIDE - frame_number
            n_read -= n_miss_left
        elif frame_start + n_read >= self.frame_count:
            n_read = self.frame_count - frame_start
            n_miss_right = NFRAME - n_read

        # read n_read frames
        if do_next_frame:
            ret, frame = self.cap.read()
            if ret:
                # bin frame
                frame = cv2.resize(frame, (self.nx // BIN, self.ny // BIN))
                self.frame_stack.pop(0)
                self.frame_stack.append(frame)
            else:
                raise Exception(f"Error reading frame {frame_number}")
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            self.frame_stack = []
            for i in range(n_read):
                ret, frame = self.cap.read()
                if ret:
                    # bin frame
                    frame = cv2.resize(frame, (self.nx // BIN, self.ny // BIN))
                    if frame_start + i == frame_number:
                        # highlight current frame
                        frame[:, :, 1:] += 30
                    self.frame_stack.append(frame)
                else:
                    raise Exception(f"Error reading frame {frame_number + i}/{self.frame_count}")
            if n_miss_left:
                z = np.zeros_like(frame)
                self.frame_stack = [z] * n_miss_left + self.frame_stack
            elif n_miss_right:
                z = np.zeros_like(frame)
                self.frame_stack = self.frame_stack + [z] * n_miss_right

        # tile images in frames according to a NROW x NCOL grid
        frames = np.array(self.frame_stack)
        frames = frames.reshape(NROW, NCOL, *frames.shape[1:])
        frames = frames.transpose(0, 2, 1, 3, 4)
        frames = frames.reshape(NROW * frames.shape[1], NCOL * frames.shape[3], frames.shape[4])

        self.display_frame(frames)
        self.frame_number_label.setText(f"Frame: {frame_number}")
        if update_slider:
            self.time_slider.setValue(frame_number)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MovieViewer()
    viewer.show()
    sys.exit(app.exec_())