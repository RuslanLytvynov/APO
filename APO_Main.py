from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import cv2
import numpy as np


class HistWindow(QWidget):
    def __init__(self, name, is_gray, hist, b_hist, g_hist, r_hist):
        super(HistWindow, self).__init__()
        self.setWindowTitle("Table Histogram of " + name)
        self.table_widget = QTableWidget()

        if is_gray:
            self.table_widget.setRowCount(2)
            self.table_widget.setColumnCount(257)

            self.table_widget.setItem(0, 0, QTableWidgetItem("PIXEL SHADE"))
            self.table_widget.setItem(1, 0, QTableWidgetItem("NUMBER OF BLACK"))
            for i in range(1, 257):
                self.table_widget.setItem(0, i, QTableWidgetItem(str(i - 1)))
                self.table_widget.setItem(1, i, QTableWidgetItem(str(hist[i - 1])))
        else:
            self.table_widget.setRowCount(4)
            self.table_widget.setColumnCount(257)

            self.table_widget.setItem(0, 0, QTableWidgetItem("PIXEL SHADE"))
            self.table_widget.setItem(1, 0, QTableWidgetItem("RED"))
            self.table_widget.setItem(2, 0, QTableWidgetItem("GREEN"))
            self.table_widget.setItem(3, 0, QTableWidgetItem("BLUE"))
            for i in range(1, 257):
                self.table_widget.setItem(0, i, QTableWidgetItem(str(i - 1)))
                self.table_widget.setItem(1, i, QTableWidgetItem(str(r_hist[i - 1])))
                self.table_widget.setItem(2, i, QTableWidgetItem(str(g_hist[i - 1])))
                self.table_widget.setItem(3, i, QTableWidgetItem(str(b_hist[i - 1])))

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.table_widget)
        self.setLayout(self.layout)


class ImageWindow(QWidget):
    def __init__(self, fName):
        super(ImageWindow, self).__init__()
        self.name = fName.split("/")[-1]
        self.fullName = fName
        self.setWindowTitle(self.name)
        self.image = cv2.imread(fName)
        self.image = self.image.copy()
        self.label = QLabel(self)
        self.label.setPixmap(QPixmap(fName))
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        if len(self.image.shape) == 2:
            self.is_gray = True
        else:
            self.is_gray = False

        self.windows = []

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 570, 30))
        self.menubar.setObjectName("menubar")

        # Creating Lab1 menu
        self.menu_lab1 = QtWidgets.QMenu(self.menubar)
        self.menu_lab1.setObjectName("menuLab1")
        self.menu_lab1.setTitle("Lab1")

        # Creating Hist action
        self.action_hist = QtWidgets.QAction(self)
        self.action_hist.setText("Histogram")
        self.action_hist.setStatusTip("Draw a histogram")
        self.action_hist.setObjectName("actionHist")

        # Creating ConvertGray action
        self.action_convert_gray = QtWidgets.QAction(self)
        self.action_convert_gray.setText("ConvertGray")
        self.action_convert_gray.setStatusTip("Convert to Gray")
        self.action_convert_gray.setObjectName("actionConvertGray")

        # Creating RGB to HSV action
        self.action_rgb_to_hsv = QtWidgets.QAction(self)
        self.action_rgb_to_hsv.setText("RGB to HSV")
        self.action_rgb_to_hsv.setStatusTip("Convert RGB to HSV")
        self.action_rgb_to_hsv.setObjectName("actionRGBtoHSV")

        # Creating RGB to Lab action
        self.action_rgb_to_lab = QtWidgets.QAction(self)
        self.action_rgb_to_lab.setText("RGB to Lab")
        self.action_rgb_to_lab.setStatusTip("Convert RGB to Lab")
        self.action_rgb_to_lab.setObjectName("actionRGBtoLab")

        # Adding actions to Lab1 menu
        self.menu_lab1.addAction(self.action_hist)
        self.menu_lab1.addAction(self.action_convert_gray)
        self.menu_lab1.addAction(self.action_rgb_to_hsv)
        self.menu_lab1.addAction(self.action_rgb_to_lab)

        # Adding menu to menubar
        self.menubar.addAction(self.menu_lab1.menuAction())

        # Connecting signals to actions
        self.action_hist.triggered.connect(self.show_histogram)
        self.action_convert_gray.triggered.connect(self.convert_to_gray)
        self.action_rgb_to_hsv.triggered.connect(self.convert_to_hsv)
        self.action_rgb_to_lab.triggered.connect(self.convert_to_lab)

    def show_histogram(self):
        if self.is_gray:
            self.hist = self.build_histogram(self.image, 'black')
            self.tabula(self.name, self.is_gray, self.hist, None, None, None)
        else:
            b, g, r = cv2.split(self.image)
            self.b_hist = self.build_histogram(b, 'b')
            self.g_hist = self.build_histogram(g, 'g')
            self.r_hist = self.build_histogram(r, 'r')
            self.tabula(self.name, self.is_gray, None, self.b_hist, self.g_hist, self.r_hist)

    def build_histogram(self, img, color):
        my_hist = np.zeros(256)
        img = img.ravel()
        for i in img:
            my_hist[i] += 1
        a_list = list(range(0, 256))
        plt.figure("Histogram of " + self.name)
        plt.bar(a_list, my_hist, alpha=0.7, color=color)
        plt.show(block=False)
        return my_hist

    def tabula(self, name, is_gray, hist, b_hist, g_hist, r_hist):
        tabl = HistWindow(name, is_gray, hist, b_hist, g_hist, r_hist)
        self.windows.append(tabl)
        tabl.show()

    def convert_to_gray(self):
        if self.is_gray:
            QMessageBox.warning(self, "Warning", "The image is already in grayscale.")
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            height, width = self.image.shape
            bytes_per_line = width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Indexed8)
            self.label.setPixmap(QPixmap.fromImage(q_img))
            self.is_gray = True

    def convert_to_hsv(self):
        if len(self.image.shape) == 2:
            QMessageBox.warning(self, "Warning", "The image is already in grayscale. HSV conversion is not possible.")
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            height, width, _ = self.image.shape
            bytes_per_line = width * 3
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_img))
            self.is_gray = False

    def convert_to_lab(self):
        if len(self.image.shape) == 2:
            QMessageBox.warning(self, "Warning", "The image is already in grayscale. Lab conversion is not possible.")
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            height, width, _ = self.image.shape
            bytes_per_line = width * 3
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_img))
            self.is_gray = False

    def convert_cv_to_qimage(self, image):
        if len(image.shape) == 2:
            gray_image = image
            h, w = gray_image.shape
            qimage = QImage(gray_image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qimage = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
        return qimage


class UiMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.setObjectName("MainWindow")
        self.resize(640, 200)
        self.setWindowTitle("APO Ruslan Lytvynov : Image Project")

        self.windows = []

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(250, 0, 500, 60))
        self.label.setObjectName("label")
        self.label.setText("Choose an image")

        self.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 570, 30))
        self.menubar.setObjectName("menubar")

        self.menu_file = QtWidgets.QMenu(self.menubar)
        self.menu_file.setObjectName("menuFile")
        self.menu_file.setTitle("File")

        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.action_open = QtWidgets.QAction(self)
        self.action_open.setText("Open")
        self.action_open.setIconText("Open")
        self.action_open.setToolTip("Open")
        self.action_open.setStatusTip("Open an image")
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.setObjectName("actionOpen")

        self.action_exit = QtWidgets.QAction(self)
        self.action_exit.setText("Exit")
        self.action_exit.setIconText("Exit")
        self.action_exit.setToolTip("Exit")
        self.action_exit.setStatusTip("Close all windows of an app")
        self.action_exit.setShortcut("Ctrl+E")
        self.action_exit.setObjectName("actionExit")

        self.menu_file.addAction(self.action_open)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_exit)

        self.menubar.addAction(self.menu_file.menuAction())

        QtCore.QMetaObject.connectSlotsByName(self)

        self.action_open.triggered.connect(self.browse_files)
        self.action_exit.triggered.connect(self.close_event)

    def browse_files(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open file...", "", "Image files (*.jpg *.jpeg *.bmp *.png *.tiff *.tif)")
        if file_name:
            self.new_image_window(file_name)
        else:
            self.label.setText("Choose an image")

    def new_image_window(self, file_name):
        window = ImageWindow(file_name)
        self.windows.append(window)
        window.show()

    def close_event(self, event):
        QApplication.closeAllWindows()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow()
    ui.show()
    sys.exit(app.exec_())
