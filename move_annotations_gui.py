import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui
import pandas as pd
import numpy as np
import cv2
import time
import traceback as tb
from typing import Optional, List
from collections import namedtuple


NROW = 5
NCOL = 7
NFRAME = NROW * NCOL
NSIDE = NFRAME // 2
BIN = 4
SKIP = 2


# named tuple for behavior
Behavior = namedtuple('Behavior', ['name', 'active', 'color'])


def connect_signal(signal, slot):

    def wrapper(*args, **kwargs):
        try:
            return slot(*args, **kwargs)
        except Exception as e:
            tb.print_tb(e.__traceback__)
            print(f"Error in slot {slot.__name__}: {e}")

    signal.connect(wrapper)


class MovieViewer(QMainWindow):
    '''A simple movie viewer application using OpenCV and PyQt5.'''

    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Movie Viewer")
        # self.setGeometry(100, 100, 400, 300)
        # self.setGeometry(10, 600, 400, 300)
        self.setGeometry(-1800, 200, 400, 300)

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
        self.n_frame = None
        self.nx = self.ny = None
        self.frame_rate = None

        # Frame data and display
        self.frame_pos = None
        self.frame_stack = self.frame_stack_colored = None

        # Annotations
        self.behaviors = []  # type: List[Behavior]
        self.n_behavior = 0
        self.annotations = None  # type: Optional[nd.array] # nframe x nbehavior

        # OpenCV video display widget
        self.video_widget = QtWidgets.QLabel(self)
        self.video_layout.addWidget(
            self.video_widget)  # Add the video widget to the video layout.

        # Buttons

        # Load Movie button
        btn = QPushButton("Load Movie...")
        self.controls_layout.addWidget(btn, row, 0, 1, 2)
        row += 1
        connect_signal(btn.clicked, self.load_movie)

        # Load/Save annotations buttons
        btn = QPushButton("Load Annotations...")
        connect_signal(btn.clicked, self.load_annotations)
        self.controls_layout.addWidget(btn, row, 0, 1, 2)
        row += 1
        layout = QtWidgets.QHBoxLayout()
        self.controls_layout.addLayout(layout, row, 0, 1, 2)
        btn = QPushButton("Save Annotations")
        connect_signal(btn.clicked, self.save_annotations)
        layout.addWidget(btn)
        btn = QPushButton("Save Annotations as...")
        connect_signal(btn.clicked, self.save_annotations_as)
        layout.addWidget(btn)
        row += 1

        # Play Movie button
        self.play_button = QPushButton("Play Movie")
        self.play_button.setCheckable(True)
        self.controls_layout.addWidget(self.play_button, row, 0, 1, 2)
        row += 1
        connect_signal(self.play_button.clicked, self.toggle_play)

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
        connect_signal(self.previous_frame_button.clicked, self.previous_frame)
        frame_by_frame_layout.addWidget(self.previous_frame_button)

        self.next_frame_button = QPushButton(">")
        self.controls_layout.addWidget(self.next_frame_button)
        connect_signal(self.next_frame_button.clicked, self.next_frame)
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
        connect_signal(self.time_slider.valueChanged, self.set_movie_position)

        # Try to load the last movie and annotation files
        self.current_folder = None
        try:
            with open('check_pose_gui_last_file.txt', 'r') as file:
                file_path = file.read()
            self.load_movie(file_path)
            with open('check_pose_gui_last_annotations.txt', 'r') as file:
                file_path = file.read()
            self.load_annotations(file_path)
        except FileNotFoundError:
            pass

    def keyPressEvent(self, event):
        try:
            if event.key() == Qt.Key_Left:
                self.previous_frame()
            elif event.key() == Qt.Key_Right:
                self.next_frame()
            elif event.key() == Qt.Key_Space:
                self.toggle_play(not self.play_button.isChecked())
            else:
                print('key pressed', event.key())
                # pass the event to the base class
                super().keyPressEvent(event)
        except Exception as e:
            tb.print_tb(e.__traceback__)
            print(f"Error in keyPressEvent {event}: {e}")

    # region Display & Movie navigation
    def display_frame(self, frame):
        # display current frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                             QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_widget.setPixmap(pixmap)
        self.video_widget.setScaledContents(True)

    def previous_frame(self):
        self.set_frame(self.frame_pos - 1)

    def next_frame(self, _=None):
        self.set_frame(self.frame_pos + 1)

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

    def set_movie_position(self, _=None):
        value = self.time_slider.value()
        self.set_frame(value, update_slider=False)

    def set_frame(self, frame_number, update_slider=True):

        # reading will be faster if we are reading frames sequentially
        do_next_frame = (frame_number - 1 == self.frame_pos)

        self.frame_pos = frame_number

        frame_start = frame_number - NSIDE
        n_read = NFRAME
        n_miss_left = n_miss_right = 0
        if frame_start < 0:
            frame_start = 0
            n_miss_left = NSIDE - frame_number
            n_read -= n_miss_left
        elif frame_start + n_read >= self.n_frame:
            n_read = self.n_frame - frame_start
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
                    self.frame_stack.append(frame)
                else:
                    raise Exception(f"Error reading frame {frame_number + i}/{self.n_frame}")
            if n_miss_left:
                z = np.zeros_like(frame)
                self.frame_stack = [z] * n_miss_left + self.frame_stack
            elif n_miss_right:
                z = np.zeros_like(frame)
                self.frame_stack = self.frame_stack + [z] * n_miss_right

        # color frames according to annotations
        frames = np.array(self.frame_stack)
        if self.annotations is not None:
            self.frame_stack_colored = frames.copy()
            target = self.frame_stack_colored[n_miss_left:n_miss_left + n_read]
            annotations = self.annotations[frame_start:frame_start + n_read, :]
            for i, behavior in enumerate(self.behaviors):
                if not behavior.active:
                    continue
                behavior_on = annotations[:, i]
                target[behavior_on] += (30 * behavior.color).astype(np.uint8)
            frames = self.frame_stack_colored

        # tile images in frames according to a NROW x NCOL grid
        frames = frames.reshape(NROW, NCOL, *frames.shape[1:])
        frames = frames.transpose(0, 2, 1, 3, 4)
        frames = frames.reshape(NROW * frames.shape[1], NCOL * frames.shape[3], frames.shape[4])

        self.display_frame(frames)
        self.frame_number_label.setText(f"Frame: {frame_number}")
        if update_slider:
            self.time_slider.setValue(frame_number)
    # endregion

    # region Load / Save
    def load_movie(self, file_path=False):
        # attempt to read folder_path from a file
        if file_path is False:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, "Select Movie File", self.current_folder,
                "Movie files (*.mp4 *.avi *.mov)")
            print(file_path)
        if not file_path:
            return
        self.current_folder = os.path.dirname(file_path)

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
        self.n_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nx = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ny = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_slider.setMaximum(self.n_frame - 1)

        # Display the first frame
        self.set_frame(0)

    def load_annotations(self, file_path=False):
        # attempt to read folder_path from a file
        if file_path is False:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, "Select Annotations File", self.current_folder,
                "CSV files (*.csv)")
        if not file_path:
            return
        self.current_folder = os.path.dirname(file_path)

        # Remember file for next time
        with open('check_pose_gui_last_annotations.txt', 'w') as file:
            file.write(file_path)

        # Load the annotations
        table = pd.read_csv(file_path, index_col=0)
        # (remove 'Frames' column if it exists)
        if 'Frames' in table.columns:
            table = table.drop(columns='Frames')
        # (get the raw data of the table (nframe x nbehavior as a numpy array))
        self.annotations = table.to_numpy().astype(bool)
        self.n_behavior = len(table.columns)
        self.behaviors = [
            Behavior(
                name=name,
                active=True,
                color=np.array((0., 1., 0.))
            )
            for name in table.columns
        ]

    def save_annotations(self, current_file=True):
        pass

    def save_annotations_as(self):
        self.save_annotations(current_file=False)
    # endregion

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MovieViewer()
    viewer.show()
    sys.exit(app.exec_())