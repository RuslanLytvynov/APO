from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


class ImageViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('APO Ruslan Lytvynov : Image Project')
        self.image = None
        self.gray_image = None
        self.hist = None
        self.gray_hist = None

        # Przyciski i etykiety
        self.open_button = QtWidgets.QPushButton("Otwórz obraz")
        self.gray_button = QtWidgets.QPushButton("Przekształć do obrazu szaroodcieniowego")
        self.hist_button = QtWidgets.QPushButton("Histogram (grafika)")
        self.hist_table_button = QtWidgets.QPushButton("Histogram (tabela)")
        self.profile_button = QtWidgets.QPushButton("Linia profilu")
        self.image_label = QtWidgets.QLabel(self)
        self.gray_image_label = QtWidgets.QLabel(self)
        self.hist_label = QtWidgets.QLabel(self)
        self.gray_hist_label = QtWidgets.QLabel(self)

        # Ustawienie layoutu
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.open_button, 0, 0)
        layout.addWidget(self.gray_button, 0, 1)
        layout.addWidget(self.hist_button, 0, 2)
        layout.addWidget(self.hist_table_button, 0, 3)
        layout.addWidget(self.profile_button, 0, 4)
        layout.addWidget(self.image_label, 1, 0, 5, 1)
        layout.addWidget(self.gray_image_label, 1, 1, 5, 1)
        layout.addWidget(self.hist_label, 1, 2, 2, 2)
        layout.addWidget(self.gray_hist_label, 3, 2, 2, 2)
        self.setLayout(layout)

        # Podpięcie przycisków do funkcji
        self.open_button.clicked.connect(self.open_image)
        self.gray_button.clicked.connect(self.convert_to_gray)
        self.hist_button.clicked.connect(self.show_histogram)
        self.hist_table_button.clicked.connect(self.show_histogram_table)
        self.profile_button.clicked.connect(self.show_profile)

    def open_image(self):
        # Otwieranie obrazu z pliku
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Otwórz obraz", "", "Obraz (*.bmp *.jpg *.png *.gif)"
        )
        if filename:
            self.image = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.show_image(self.image)

    def show_image(self, image):
        # Wyświetlanie obrazu w etykiecie
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(
            image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def convert_to_gray(self):
        # Konwersja obrazu do szaroodcieniowego i wyświetlenie go w etykiecie
        if self.image is not None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.show_gray_image(self.gray_image)

    def show_gray_image(self, gray_image):
        # Wyświetlenie obrazu szaroodcieniowego w etykiecie
        height, width = gray_image.shape
        bytes_per_line = width
        q_image = QtGui.QImage(
            gray_image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.gray_image_label.setPixmap(pixmap)

    def calculate_histogram(self, image):
        # Obliczenie histogramu obrazu szaroodcieniowego
        if image is not None:
            hist, _ = np.histogram(image.flatten(), 256, [0, 256])
            return hist

    def show_histogram(self):
        try:
            # Wyświetlenie histogramu w postaci graficznej
            if self.gray_image is not None:
                self.hist = self.calculate_histogram(self.gray_image)
                plt.hist(self.gray_image.flatten(), 256, [0, 256], color="r")
                plt.xlim([0, 256])
                plt.show()
        except Exception as e:
            print("Wystąpił błąd:", e)

    def show_histogram_table(self):
        # Wyświetlenie histogramu w postaci tabelarycznej
        if self.gray_image is not None:
            self.gray_hist = self.calculate_histogram(self.gray_image)
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("Histogram")
            dialog.setMinimumSize(400, 300)
            layout = QtWidgets.QVBoxLayout(dialog)
            table = QtWidgets.QTableWidget()
            table.setRowCount(len(self.gray_hist))
            table.setColumnCount(2)
            for i, val in enumerate(self.gray_hist):
                table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
                table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(val)))
            table.setHorizontalHeaderLabels(["Wartość", "Liczba pikseli"])
            layout.addWidget(table)
            dialog.exec_()

    def show_profile(self):
        # Wyświetlenie linii profilu
        if self.gray_image is not None:
            x, y = self.get_profile_line()
            plt.plot(x, y, color="r")
            plt.xlim([0, self.gray_image.shape[1]])
            plt.ylim([0, 256])
            plt.show()

    def get_profile_line(self):
        # Pobranie linii profilu
        x1, y1, x2, y2 = 0, self.gray_image.shape[0] // 2, self.gray_image.shape[1], self.gray_image.shape[0] // 2
        line = cv2.line(np.zeros_like(self.gray_image), (x1, y1), (x2, y2), 255, 1)
        profile = cv2.bitwise_and(self.gray_image, self.gray_image, mask=line)
        x = np.arange(self.gray_image.shape[1])
        y = np.sum(profile, axis=0)
        return x, y


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())
