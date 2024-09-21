import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap, QColor
import pandas as pd
import numpy as np
import cv2
import time
import traceback as tb
from typing import Optional, List
import inspect


NROW = 7
NCOL = 7
NFRAME = NROW * NCOL
NSIDE = NFRAME // 2
BIN = 4
SKIP = 2

COLORS = np.array([
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [1, 1, 0],  # yellow
    [0, 0, 1],  # blue
    [1, 0, 1],  # magenta
    [0, 1, 1],  # cyan
    [1, 0.5, 0],  # orange
    [0.5, 1, 0],  # light green
    [0, 0.5, 1],  # light blue
    [0.5, 0, 1],  # purple
    [1, 0, 0.5],  # pink
])


app = QApplication(sys.argv)


# named tuple for behavior
class Behavior:
    def __init__(self, name, active, color):
        self.name = name
        self.active = active
        self.color = color


class NoKeyBoardWidget(QtWidgets.QWidget):
    def keyPressEvent(self, event):
        # Override and ignore key press events
        event.ignore()

    def keyReleaseEvent(self, event):
        # Override and ignore key release events
        event.ignore()


class NonEditableTableWidget(QtWidgets.QTableWidget, NoKeyBoardWidget):
    def edit(self, index, trigger, event):
        return False  # Disable all editing


class NoKeyboardSlider(QtWidgets.QSlider, NoKeyBoardWidget):
    pass


class SliderLabel(QtWidgets.QLabel):

    def __init__(self, orientation):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.slider = NoKeyboardSlider(orientation)
        layout.addWidget(self.slider)
        width = self.slider.minimumSizeHint().width()
        self.setFixedWidth(width)

    def set_background(self, image_array):
        # add transparency
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        image_array = np.concatenate(
            [image_array,
             np.full(image_array.shape[:2] + (1,), 60, dtype=np.uint8)],
            axis=2
        )
        # convert image array to QPixmap
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        q_image = QImage(image_array.data, width, height, bytes_per_line,
                         QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image)
        # apply pixmap to label
        self.setPixmap(pixmap)
        self._update_pixmap_size()

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        # rescale pixmap when label size changes
        if self.pixmap():
            self._update_pixmap_size()

    def _update_pixmap_size(self):
        self.setPixmap(self.pixmap().scaled(self.width(), self.height()))


class FramesDisplayLabel(QtWidgets.QLabel):
    def __init__(self, main):
        super().__init__()
        self.main = main  # type: MovieViewer
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        try:
            self.main._edit_annotation_start(event)
        except Exception as e:
            tb.print_tb(e.__traceback__)
            print(f"Error in callback: {e}")

    def mouseReleaseEvent(self, event):
        try:
            self.main._edit_annotation_end(event)
        except Exception as e:
            tb.print_tb(e.__traceback__)
            print(f"Error in callback: {e}")

    def mouseMoveEvent(self, event):
        try:
            self.main._edit_annotation_move(event)
        except Exception as e:
            tb.print_tb(e.__traceback__)
            print(f"Error in callback: {e}")


def connect_signal(signal, slot):

    # inspect slot function signature to check whether it accepts at least one
    # argument
    sig = inspect.signature(slot)
    no_argument = (len(sig.parameters) == 0)

    def wrapper(*args, **kwargs):
        try:
            if no_argument:
                # When connecting a clicked signal, the slot might not
                # accept any argument. If we had connected signal directly
                # to slot this would have been detected and slot would be
                # called without arguments. But since we connected signal to
                # wrapper, which accepts arguments, wrapper is called with
                # an argument, which we need to ignore when calling the slot
                # here.
                return slot()
            else:
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
        self.setGeometry(100, 100, 400, 300)
        # self.setGeometry(10, 600, 400, 300)
        # self.setGeometry(-1800, 200, 400, 300)

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

        # OpenCV video capture object
        self.cap = None
        self.n_frame = None
        self.nx = self.ny = None
        self.frame_rate = None

        # Frame data and display
        self.center_frame_mask = None
        self.single_frame_pos = self.frame_pos = None
        self.frame_stack = self.frame_stack_colored = None
        self.single_frame = self.single_frame_colored = None

        # Annotations
        self.behaviors = []  # type: List[Behavior]
        self.n_behavior = 0
        self.annotations = None  # type: Optional[nd.array] # nframe x nbehavior
        self.current_annotations_file = None
        self._edit_first_frame = None
        self._edit_value = None
        self._edit_prev_values = None
        self.needs_save = False
        self._scrolling_while_edit = False
        self.undo_stack = []  # history of annotations for undo
        self.last_save_pos_in_undo_stack = 0
        self.update_title()  # needs current_annotations_file and needs_save

        # Annotation color marks
        self.annotation_colors = None

        # Frames display widget
        self.frames_widget = FramesDisplayLabel(self)
        self.frames_widget.setScaledContents(True)
        self.main_layout.addWidget(self.frames_widget)

        # Full size current frame display
        self.single_frame_widget = QtWidgets.QLabel(self)
        self.single_frame_widget.setScaledContents(True)
        self.controls_layout.addWidget(self.single_frame_widget, row, 0, 1, 3)
        row += 1

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
        connect_signal(btn.clicked, self.save_annotations_current)
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

        # Movie speed adjustment
        self.speed_slider_label = QtWidgets.QLabel("Speed: x 1")
        self.controls_layout.addWidget(self.speed_slider_label, row, 0)
        self.speed_slider_label.setFixedWidth(
            self.speed_slider_label.fontMetrics().boundingRect(
                "Speed: x 1.5").width())
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(10)
        self.speed_slider.setPageStep(1)
        connect_signal(
            self.speed_slider.valueChanged,
            lambda: self.speed_slider_label.setText(
                f"Speed: x {self.speed_slider.value() / 10}")
        )
        self.controls_layout.addWidget(self.speed_slider, row, 1)
        row += 1

        # Label for frame number
        self.frame_pos_label = QtWidgets.QLabel("Frame:")
        # (increase width to make room for the frame number)
        self.frame_pos_label.setFixedWidth(
            self.frame_pos_label.fontMetrics().boundingRect("Frame: 00000").width())
        self.controls_layout.addWidget(self.frame_pos_label, row, 0)

        # Frame Navigation Controls
        sub_layout = QtWidgets.QHBoxLayout()  # A horizontal box layout for the frame-by-frame navigation controls.
        self.controls_layout.addLayout(sub_layout, row, 1)
        row += 1
        btn = QPushButton("<")
        self.controls_layout.addWidget(btn)
        connect_signal(btn.clicked, self.previous_frame)
        sub_layout.addWidget(btn)
        btn = QPushButton(">")
        self.controls_layout.addWidget(btn)
        connect_signal(btn.clicked, self.next_frame)
        sub_layout.addWidget(btn)

        # Currently edited behavior
        label = QtWidgets.QLabel("Current Behavior:")
        self.controls_layout.addWidget(label, row, 0)
        self.current_behavior_selection = QtWidgets.QComboBox()
        self.current_behavior_selection.addItem("(None)")
        self.controls_layout.addWidget(self.current_behavior_selection, row, 1)
        row += 1

        # Clear behavior button
        btn = QPushButton("Clear Behavior")
        self.controls_layout.addWidget(btn, row, 0, 1, 2)
        connect_signal(btn.clicked, self.clear_behavior)
        row += 1

        # Navigate to next / previous behavior event
        sub_layout = QtWidgets.QHBoxLayout()
        self.controls_layout.addLayout(sub_layout, row, 0, 1, 2)
        row += 1
        btn = QPushButton("Previous Event")
        self.controls_layout.addWidget(btn)
        connect_signal(btn.clicked, self.previous_event)
        sub_layout.addWidget(btn)
        btn = QPushButton("Next Event")
        self.controls_layout.addWidget(btn)
        connect_signal(btn.clicked, self.next_event)
        sub_layout.addWidget(btn)

        # Show keyboard shortcuts button
        btn = QPushButton("Help...")
        self.controls_layout.addWidget(btn, row, 0, 1, 2)
        row += 1
        connect_signal(btn.clicked, self.show_help)

        # Undo button
        self.undo_button = QPushButton("Undo")
        self.controls_layout.addWidget(self.undo_button, row, 0, 1, 2)
        self.undo_button.setEnabled(False)
        connect_signal(self.undo_button.clicked, self.undo)
        row += 1

        # Add a spacer to the controls layout to push the buttons to the top
        vertical_stretch = QtWidgets.QWidget()
        vertical_stretch.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.controls_layout.addWidget(vertical_stretch, row, 0)
        row += 1

        # Behavior(s) selection (table)
        self.behavior_selection = NonEditableTableWidget()
        self.behavior_selection.horizontalHeader().setVisible(False)  # Hides the column headers
        self.behavior_selection.verticalHeader().setVisible(False)  # Hides the row headers
        self.behavior_selection.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.controls_layout.addWidget(self.behavior_selection, 1, 2, row-1, 1)
        connect_signal(self.behavior_selection.itemSelectionChanged,
                       self._list_select_behaviors)
        # connect_signal(self.behavior_selection.clicked,
        #                self._behavior_selection_clicked)
        self._selection_filling = False

        # Time slider
        self.time_slider_label = SliderLabel(Qt.Vertical)
        self.time_slider = self.time_slider_label.slider
        self.time_slider.setInvertedAppearance(True)  # go from top to bottom
        self.main_layout.addWidget(self.time_slider_label)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.setPageStep(NFRAME)
        # self.time_slider.wheelEvent()
        connect_signal(self.time_slider.valueChanged, self._slider_movie_position)

        # Make controls not focusable, so the keyboard controls the main
        # program rather than some inner control
        for w in self.main_widget.children():
            if isinstance(w, QtWidgets.QWidget):
                w.setFocusPolicy(Qt.NoFocus)

        # Flags to check what is currently displayed
        self.displayed_single_frame_pos = None
        self.displayed_frame_pos = None
        self.displayed_colors_ok = False

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

    # region Properties
    @property
    def current_behavior(self):
        # read value from control
        value = self.current_behavior_selection.currentIndex()
        if value == 0:
            return None
        else:
            return value-1
    # endregion

    # region Window
    def update_needs_save(self, value):
        if self.needs_save == value:
            return
        self.needs_save = value
        self.update_title()

    def update_title(self):
        if self.current_annotations_file is None:
            self.setWindowTitle(f"Check Behavior GUI")
        else:
            self.setWindowTitle(
                f"Check Behavior GUI - {self.current_annotations_file}"
                f"{'*' if self.needs_save else ''}"
            )

    def closeEvent(self, event):
        try:
            if self.check_unsaved_changes_do_cancel():
                # Cancel the close event
                return
        except Exception as e:
            tb.print_tb(e.__traceback__)
            print(f"Error in closeEvent: {e}")

        # Stop movie playing
        self.play_button.setChecked(False)

        event.accept()
    # endregion Window

    # region Keyboard and scroll wheel actions
    def keyPressEvent(self, event):
        try:
            print('keyPressEvent', event.key())
            if event.key() == Qt.Key_Left:
                self.previous_frame()
            elif event.key() == Qt.Key_Right:
                self.next_frame()
            elif event.key() == Qt.Key_Up:
                self.move_frame('line', -1)
            elif event.key() == Qt.Key_Down:
                self.move_frame('line', 1)
            elif event.key() == Qt.Key_PageUp:
                self.move_frame('page', -1)
            elif event.key() == Qt.Key_PageDown:
                self.move_frame('page', 1)
            elif event.key() == Qt.Key_Home:
                self.move_frame('start')
            elif event.key() == Qt.Key_End:
                self.move_frame('end')
            elif event.key() == Qt.Key_Space:
                self.toggle_play(not self.play_button.isChecked())
            elif event.key() == Qt.Key_N:
                self.move_frame('event', 1)
            elif event.key() == Qt.Key_P:
                self.move_frame('event', -1)
            elif event.key() in [Qt.Key_H, Qt.Key_K]:
                self.show_help()
            elif (event.key() == Qt.Key_S
                  and event.modifiers() & Qt.ControlModifier):
                self.save_annotations_current()
            elif (event.key() == Qt.Key_Z
                  and event.modifiers() & Qt.ControlModifier):
                self.undo()
            else:
                # pass the event to the base class
                super().keyPressEvent(event)
        except Exception as e:
            tb.print_tb(e.__traceback__)
            print(f"Error in keyPressEvent {event}: {e}")

    def show_help(self):
        # Create the message box
        msg_box = QMessageBox()

        # Help content
        rich_text = """
        <h1>Check Behavior GUI</h1>
        
        <table>
            <tr><td>Select behavior in the table to highlight it</td></tr>
            <tr><td>Edit behavior using mouse in the frames 
            display</td></tr>
            <tr><td>Jump to previous/next event using buttons or 
            shortcuts</td></tr>
        </table>
        
        <h2>Mouse action</h2>
        <table>
            <tr><td><b>Left button</b></td><td>Add behavior</td></tr>
            <tr><td><b>Right button</b></td><td>Remove behavior</td></tr>
            <tr><td><b>Middle button</b></td><td>Inspect frames (no 
            edit)</td></tr>
        </table>

        <h2>Keyboard Shortcuts</h2>
        <table>
            <tr><td><b>Left/Right Arrow</b></td><td>Previous/Next frame</td></tr>
            <tr><td><b>Up/Down Arrow</b></td><td>Previous/Next line</td></tr>
            <tr><td><b>Page Up/Page Down</b></td><td>Previous/Next page</td></tr>
            <tr><td><b>Home/End</b></td><td>Go to start/end of the movie</td></tr>
            <tr><td><b>Space</b></td><td>Play/Pause</td></tr>
            <tr><td><b>N/P</b></td><td>Next/Previous event</td></tr>
            <tr><td><b>H or K</b></td><td>Show this help</td></tr>
            <tr><td><b>Ctrl+S</b></td><td>Save annotations</td></tr>
            <tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>        
        </table>
        """

        # Set the text and format
        msg_box.setText(rich_text)
        msg_box.setTextFormat(Qt.RichText)  # Enable rich text format

        # Display the information message box
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.exec()


    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        angle = ev.angleDelta().y()
        n_step = angle // 120
        self.move_frame('page', -n_step)

    def move_frame(self, step, n_step=None):
        if step == 'frame':
            self.set_frame(self.frame_pos + n_step)
        elif step == 'line':
            self.set_frame(self.frame_pos + n_step * NCOL)
        elif step == 'page':
            self.set_frame(self.frame_pos + n_step * NFRAME)
        elif step == 'start':
            self.set_frame(0)
        elif step == 'end':
            self.set_frame(self.n_frame + 1)
        elif step == 'event':
            # 'event' frames are those with behavior on and a neighbor is off
            if self.current_behavior is None:
                return
            on = self.annotations[:, self.current_behavior]
            off = np.hstack((True, np.logical_not(on), True))
            events = np.logical_and(
                on, np.logical_or(off[:-2], off[2:]))
            if n_step == 1:
                # search for the next event
                next_event = np.where(events[self.frame_pos + 1:])[0] + 1
                if len(next_event):
                    self.set_frame(self.frame_pos + next_event[0])
                else:
                    self.move_frame('end')
            elif n_step == -1:
                # search for the previous event
                prev_event = np.where(events[:self.frame_pos])[0]
                if len(prev_event):
                    self.set_frame(prev_event[-1])
                else:
                    self.move_frame('start')
            else:
                raise RuntimeError(f'Invalid move_frame arguments {step}, {n_step}')

    def previous_frame(self):
        self.move_frame('frame', -1)

    def next_frame(self):
        self.move_frame('frame', 1)

    def previous_event(self):
        self.move_frame('event', -1)

    def next_event(self):
        self.move_frame('event', 1)
    # endregion Keyboard and scroll wheel actions

    # region Display
    def _prepare_frame_display(self):
        # Mask around the central frame (corresponding to current_pos)
        nx, ny = self.nx, self.ny
        self.center_frame_mask = np.zeros((NROW * ny, NCOL * nx),
                                          dtype=bool)
        top, bottom = (NROW // 2) * ny, (NROW // 2 + 1) * ny + 1
        left, right = (NCOL // 2) * nx, (NCOL // 2 + 1) * nx + 1
        self.center_frame_mask[top:bottom + 1, [left, right]] = True
        self.center_frame_mask[[top, bottom], left:right + 1] = True

        # Set size of the viewers
        self.single_frame_widget.setFixedSize(self.nx0, self.ny0)

    def _display_frames(self, frames):
        # display current frame
        frames[self.center_frame_mask] = 0
        frames = np.clip(frames, 0, 255)
        image = QtGui.QImage(frames, frames.shape[1], frames.shape[0],
                             QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.frames_widget.setPixmap(pixmap)

    def _display_single_frame(self, frame):
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                             QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.single_frame_widget.setPixmap(pixmap)

    def update_display(self):

        # Annotation colors
        if self.annotations is not None and not self.displayed_colors_ok:
            self.annotation_colors = np.zeros(
                (len(self.behaviors), 1, 3), dtype=np.uint8)
            active = [behavior.active for behavior in self.behaviors]
            if sum(active) > 0:
                annotations = self.annotations[:, active]  # nframe x nbehavior
                colors = np.array([behavior.color
                                   for behavior in self.behaviors
                                   if behavior.active])  # nbehavior x 3
                self.annotation_colors = np.dot(
                    annotations, colors)  # nframe x 3
            else:
                self.annotation_colors = np.zeros((self.n_frame, 3))

            # apply to slider background
            self.time_slider_label.set_background(
                self.annotation_colors[:, np.newaxis, :] * 255)

        # Frame number changed?
        changed_single_frame = (
                self.displayed_single_frame_pos != self.single_frame_pos)
        changed_frame = self.displayed_frame_pos != self.frame_pos
        self.displayed_single_frame_pos = self.single_frame_pos
        self.displayed_frame_pos = self.frame_pos
        if changed_frame:
            # reading will be faster if we are reading frames sequentially
            do_next_frame = (self.displayed_frame_pos is not None and
                             self.displayed_frame_pos + 1 == self.frame_pos)
        else:
            do_next_frame = False

        # handle start and end of movie
        frame_start = self.frame_pos - NSIDE
        n_read = NFRAME
        n_miss_left = n_miss_right = 0
        if frame_start < 0:
            frame_start = 0
            n_miss_left = NSIDE - self.frame_pos
            n_read -= n_miss_left
        elif frame_start + n_read >= self.n_frame:
            n_read = self.n_frame - frame_start
            n_miss_right = NFRAME - n_read

        # read full-size current frame and display it
        if changed_single_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.single_frame_pos)
            ret, single_frame = self.cap.read()
            if not ret:
                raise Exception(
                    f"Error reading frame {self.frame_pos}/{self.n_frame}")
            self.single_frame = cv2.resize(single_frame, (self.nx, self.ny))

        # read n_read frames and store them in self.frame_stack
        if not changed_frame:
            # self.frame_stack is already up to date
            pass
        elif do_next_frame:
            read_frame = frame_start + NFRAME - 1
            if read_frame >= self.n_frame:
                frame = np.zeros_like(self.frame_stack[0])
                ret = True
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, read_frame)
                ret, frame = self.cap.read()
            if ret:
                # bin frame
                frame = cv2.resize(frame, (self.nx, self.ny))
                self.frame_stack.pop(0)
                self.frame_stack.append(frame)
            else:
                raise Exception(
                    f"Error reading frame {read_frame}/{self.n_frame}")
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            self.frame_stack = []
            frame = None  # to avoid IDE warning below...
            for i in range(n_read):
                ret, frame = self.cap.read()
                if ret:
                    # bin frame
                    frame = cv2.resize(frame, (self.nx, self.ny))
                    self.frame_stack.append(frame)
                else:
                    raise Exception(f"Error reading frame "
                                    f"{frame_start + i}/{self.n_frame}")
            if n_miss_left:
                z = np.zeros_like(frame)
                self.frame_stack = [z] * n_miss_left + self.frame_stack
            elif n_miss_right:
                z = np.zeros_like(frame)
                self.frame_stack = self.frame_stack + [z] * n_miss_right

        # color frames according to annotations
        if self.annotations is not None:
            # color single frame
            if changed_single_frame or not self.displayed_colors_ok:
                color = self.annotation_colors[self.single_frame_pos, :]
                self.single_frame_colored = (
                        self.single_frame + (30 * color).astype(np.uint8))
                single_frame = self.single_frame_colored
            else:
                single_frame = None
            # color frames stack
            if changed_frame or not self.displayed_colors_ok:
                self.frame_stack_colored = np.copy(self.frame_stack)
                target = self.frame_stack_colored[n_miss_left:n_miss_left + n_read]
                colors = self.annotation_colors[
                         frame_start:frame_start+n_read, np.newaxis, np.newaxis, :]
                target += (30 * colors).astype(np.uint8)
                self.displayed_colors_ok = True
                frames = self.frame_stack_colored
            else:
                frames = None
        else:
            if changed_single_frame:
                single_frame = self.single_frame
            else:
                single_frame = None
            if changed_frame:
                frames = np.array(self.frame_stack)
            else:
                frames = None

        # Display!
        if single_frame is not None:
            self._display_single_frame(single_frame)
        if frames is not None:
            # tile images in frames according to a NROW x NCOL grid
            frames = frames.reshape(NROW, NCOL, *frames.shape[1:])
            frames = frames.transpose(0, 2, 1, 3, 4)
            frames = frames.reshape(NROW * frames.shape[1],
                                    NCOL * frames.shape[3], frames.shape[4])
            self._display_frames(frames)
    # endregion Display

    # region Movie updates
    def load_movie(self, file_path=False):
        # attempt to read folder_path from a file
        if file_path is False:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, "Select Movie File", self.current_folder,
                "Movie files (*.mp4 *.avi *.mov)")
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
        self.nx0 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nx = self.nx0 // BIN
        self.ny0 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ny = self.ny0 // BIN
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.time_slider.setMaximum(self.n_frame - 1)
        self._prepare_frame_display()

        # Display the first frame
        self.set_frame(0)

    def toggle_play(self, value):
        if value:
            self.play_button.setChecked(True)
            self.play_button.setText("Stop playing")
            if self.frame_pos == self.n_frame - 1:
                self.set_frame(0)
            frame_step = 1  # if we get too slow we will skip frames
            while self.play_button.isChecked() and self.frame_pos < self.n_frame - 1:
                tick = time.time()
                self.move_frame('frame', frame_step)
                QtWidgets.QApplication.processEvents()
                period = 1 / (self.frame_rate * self.speed_slider.value() / 10)
                time_time = time.time()
                elapsed = time_time - tick
                frame_step = int(elapsed / period) + 1
                while time.time() - tick < frame_step * period:
                    time.sleep(min(.01, frame_step * period - elapsed))
                    app.processEvents()

        self.play_button.setChecked(False)
        self.play_button.setText("Play Movie")

    def _slider_movie_position(self):
        value = self.time_slider.value()
        self.set_frame(value, update_slider=False)

    def set_frame(self, frame_pos, update_slider=True):
        self.frame_pos = np.clip(frame_pos, 0, self.n_frame-1)
        self.frame_pos_label.setText(f"Frame: {self.frame_pos}")
        self.single_frame_pos = self.frame_pos
        if update_slider:
            self.time_slider.setValue(self.frame_pos)
        self.update_display()
    # endregion Movie updates

    # region Annotation updates
    def load_annotations(self, file_path=False):
        # warn if unsaved changes
        if self.check_unsaved_changes_do_cancel():
            return

        # attempt to read folder_path from a file
        if file_path is False:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, "Select Annotations File", self.current_folder,
                "CSV files (*.csv)")
        if not file_path:
            return
        self.current_annotations_file = file_path
        self.needs_save = False
        self.update_title()
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
                active=False,
                color=COLORS[i % len(COLORS)]
            )
            for i, name in enumerate(table.columns)
        ]

        # Update list of behaviors
        self._selection_filling = True  # avoid calling select_behaviors
        self.behavior_selection.clear()
        self.behavior_selection.setColumnCount(2)
        self.behavior_selection.setRowCount(self.n_behavior)
        self.current_behavior_selection.clear()
        self.current_behavior_selection.addItem("(None)")
        for i, behavior in enumerate(self.behaviors):
            # name
            item = QtWidgets.QTableWidgetItem(behavior.name)
            self.behavior_selection.setItem(i, 0, item)
            item.setSelected(behavior.active)

            # color
            # color = (180 + 30 * behavior.color).astype(np.uint8)
            color = (255 * behavior.color).astype(np.uint8)
            item = QtWidgets.QTableWidgetItem()
            item.setBackground(QColor(*color))
            self.behavior_selection.setItem(i, 1, item)
            # # make icon with uniform color
            # pixmap = QtGui.QPixmap(32, 32)
            # pixmap.fill(QColor(*color))
            # item.setIcon(QtGui.QIcon(pixmap))

            # also add name to current behavior selection control
            self.current_behavior_selection.addItem(behavior.name)
        self._selection_filling = False

        # Other display options for the table
        # (resize second columns to fit content and save space)
        self.behavior_selection.resizeColumnToContents(1)
        # (stretch first column to fill the available horizontal space)
        self.behavior_selection.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)

        # Update display
        self.displayed_colors_ok = False
        self.update_display()

        # Reset the undo stack
        self.undo_stack.clear()
        self.undo_button.setEnabled(False)
        self.last_save_pos_in_undo_stack = 0

    def check_unsaved_changes_do_cancel(self):
        # Check if there are unsaved changes, ask user whether to save them,
        # return True if user cancels the operation, False otherwise
        if self.needs_save:
            reply = QMessageBox.question(
                self, 'Unsaved changes',
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return True
            elif reply == QMessageBox.Yes:
                self.save_annotations()
        return False

    def save_annotations(self, current_file=True):
        if current_file and self.current_annotations_file is not None:
            file_path = self.current_annotations_file
        else:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self, "Save Annotations File", self.current_folder,
                "CSV files (*.csv)")
            if not file_path:
                return

        # Save the annotations
        columns = ["Frames"] + [b.name for b in self.behaviors]
        data = np.concatenate([
            np.arange(self.n_frame).reshape(-1, 1),
            self.annotations
        ], axis=1)
        table = pd.DataFrame(data, columns=columns)
        try:
            table.to_csv(file_path, index=False)
            self.current_annotations_file = file_path
        except PermissionError as e:
            # raise an error dialog
            QMessageBox.critical(
                self, 'Error saving annotations',
                f"Error saving annotations: {e}")
            return

        # No more unsaved changes
        self.needs_save = False
        self.update_title()
        self.last_save_pos_in_undo_stack = len(self.undo_stack)

    def save_annotations_current(self):
        self.save_annotations(current_file=True)

    def save_annotations_as(self):
        self.save_annotations(current_file=False)

    def _list_select_behaviors(self):
        if self._selection_filling:
            # list widget not valid yet
            return

        # special: single cell selection in the second column
        selection = self.behavior_selection.selectedIndexes()
        if len(selection) == 1 and selection[0].column() == 1:
            idx = selection[0].row()
            self._edit_behavior_color(idx)
            # remove selection
            self.behavior_selection.item(idx, 1).setSelected(False)
            return

        # select active behaviors
        for i, behavior in enumerate(self.behaviors):
            # check selection from first column
            behavior.active = (
                self.behavior_selection.item(i, 0).isSelected() or
                self.behavior_selection.item(i, 1).isSelected()
            )
            # cleanup selection (should be only in the first column)
            self.behavior_selection.item(i, 0).setSelected(behavior.active)
            self.behavior_selection.item(i, 1).setSelected(False)  # avoid hiding colors

        # update current behavior
        if len(selection) == 0:
            self.current_behavior_selection.setCurrentIndex(0)
        else:
            idx = self.behavior_selection.currentIndex().row()
            self.current_behavior_selection.setCurrentIndex(idx + 1)

        # update frames display
        self.displayed_colors_ok = False
        self.update_display()

    def _edit_behavior_color(self, idx: int):
        if self.behavior_selection.currentColumn() == 1:
            # edit color using color dialog
            color = (self.behaviors[idx].color * 255).astype(np.uint8)
            q_color = QColor(*color)
            q_color = QtWidgets.QColorDialog.getColor(q_color)
            color = np.array((q_color.red(), q_color.green(), q_color.blue()))
            self.behaviors[idx].color = color / 255
            # update cell in table
            item = self.behavior_selection.item(idx, 1)
            item.setBackground(q_color)

    def _edit_annotation_start(self, event: QtGui.QMouseEvent):
        # no current behavior?
        behavior = self.current_behavior

        # Get the coordinates of the mouse click
        x = event.pos().x()
        y = event.pos().y()

        # Add previous value to undo stack
        self.undo_stack.append(self.annotations.copy())
        self.undo_button.setEnabled(True)

        # Start a selection
        self._edit_first_frame = self.point_to_frame(x, y)
        if behavior is None or event.button() == QtCore.Qt.MiddleButton:
            self._edit_prev_values = None
        else:
            self._edit_value = (event.button() == QtCore.Qt.LeftButton)
            self._edit_prev_values = self.annotations[:, behavior].copy()
        self.edit_annotation(self._edit_first_frame)

    def _edit_annotation_move(self, event: QtGui.QMouseEvent):
        if self._edit_first_frame is None:
            return

        # interrupt scrolling if any
        self._scrolling_while_edit = False

        y = event.pos().y()

        scroll_limit = 50
        if y < scroll_limit:
            self._scrolling_while_edit = -1
        elif y > self.frames_widget.height() - scroll_limit:
            self._scrolling_while_edit = 1

        repeat = True
        while repeat:
            # scroll?
            if self._scrolling_while_edit:
                self.move_frame('line', self._scrolling_while_edit)

            # update selection
            x = event.pos().x()
            frame = self.point_to_frame(x, y)
            self.edit_annotation(frame)

            # repeat if scrolling and no new mouseMoveEvent occurred
            if self._scrolling_while_edit:
                app.processEvents()
                time.sleep(.15)
                app.processEvents()
                repeat = bool(self._scrolling_while_edit)
            else:
                repeat = False

    def _edit_annotation_end(self, last_frame):
        self._edit_first_frame = None
        # interrupt scrolling if any
        self._scrolling_while_edit = False
        # revert to showing current frame enlarged
        self.single_frame_pos = self.frame_pos

        self.update_display()

    def edit_annotation(self, last_frame):
        # no current behavior?
        behavior = self.current_behavior

        # show selected frame enlarged
        self.single_frame_pos = last_frame

        # edit current behavior's annotation
        if self._edit_value is not None:
            first_frame = self._edit_first_frame
            value = self._edit_value
            values = self._edit_prev_values.copy()
            if last_frame < first_frame:
                first_frame, last_frame = last_frame, first_frame
            values[first_frame:last_frame+1] = value
            self.annotations[:, behavior] = values
            self.displayed_colors_ok = False
            self.update_needs_save(True)

        # update display
        self.update_display()

    def clear_behavior(self):
        # Add previous value to undo stack
        self.undo_stack.append(self.annotations.copy())
        self.undo_button.setEnabled(True)

        # Clear current behavior's annotations
        behavior = self.current_behavior
        self.annotations[:, behavior] = False
        self.displayed_colors_ok = False
        self.update_needs_save(True)
        self.update_display()

    def undo(self):
        if not self.undo_stack:
            return
        # pop last action
        self.annotations = self.undo_stack.pop()
        # update display
        self.displayed_colors_ok = False
        self.update_display()
        # update needs save
        self.update_needs_save(
            self.last_save_pos_in_undo_stack != len(self.undo_stack))
        # edit enabled state of undo button
        self.undo_button.setEnabled(bool(self.undo_stack))

    def point_to_frame(self, x, y):
        frame_start = self.frame_pos - NSIDE
        frame = frame_start + x // self.nx + NCOL * (y // self.ny)
        frame = np.clip(frame, 0, self.n_frame - 1)
        return frame
    # endregion Annotation updates


if __name__ == "__main__":
    viewer = MovieViewer()
    viewer.show()
    sys.exit(app.exec_())