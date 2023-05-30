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

        # Adding Lab2 menu
        self.menu_lab2 = QtWidgets.QMenu(self.menubar)
        self.menu_lab2.setObjectName("menuLab2")
        self.menu_lab2.setTitle("Lab2")

        # Creating Stretch Histogram action
        self.action_stretch_hist = QtWidgets.QAction(self)
        self.action_stretch_hist.setText("Stretch Histogram")
        self.action_stretch_hist.setStatusTip("Apply histogram stretching")
        self.action_stretch_hist.setObjectName("actionStretchHist")

        # Creating Equalize Histogram action
        self.action_equalize_hist = QtWidgets.QAction(self)
        self.action_equalize_hist.setText("Equalize Histogram")
        self.action_equalize_hist.setStatusTip("Apply histogram equalization")
        self.action_equalize_hist.setObjectName("actionEqualizeHist")

        # Creating Negation action
        self.action_negation = QtWidgets.QAction(self)
        self.action_negation.setText("Negation")
        self.action_negation.setStatusTip("Apply negation")
        self.action_negation.setObjectName("actionNegation")

        # Creating Posterization action
        self.action_posterization = QtWidgets.QAction(self)
        self.action_posterization.setText("Posterization")
        self.action_posterization.setStatusTip("Apply posterization")
        self.action_posterization.setObjectName("actionPosterization")

        # Creating Selective Stretching action
        self.action_selective_stretching = QtWidgets.QAction(self)
        self.action_selective_stretching.setText("Selective Stretching")
        self.action_selective_stretching.setStatusTip("Apply selective stretching")
        self.action_selective_stretching.setObjectName("actionSelectiveStretching")

        # Adding actions to Lab2 menu
        self.menu_lab2.addAction(self.action_stretch_hist)
        self.menu_lab2.addAction(self.action_equalize_hist)
        self.menu_lab2.addAction(self.action_negation)
        self.menu_lab2.addAction(self.action_posterization)
        self.menu_lab2.addAction(self.action_selective_stretching)

        # Adding menu to menubar
        self.menubar.addAction(self.menu_lab2.menuAction())

        # Connecting signals to actions
        self.action_stretch_hist.triggered.connect(self.stretch_histogram)
        self.action_equalize_hist.triggered.connect(self.equalize_histogram)
        self.action_negation.triggered.connect(self.negation)
        self.action_posterization.triggered.connect(self.posterization)
        self.action_selective_stretching.triggered.connect(self.selective_stretching)

        # Adding Lab3 menu
        self.menu_lab3 = QtWidgets.QMenu(self.menubar)
        self.menu_lab3.setObjectName("menuLab3")
        self.menu_lab3.setTitle("Lab3")

        # Creating Blur action
        self.actionBlur = QtWidgets.QAction(self)
        self.actionBlur.setText("Blur")
        self.actionBlur.setStatusTip("Blur")
        self.actionBlur.setObjectName("actionBlur")

        # Creating Sobel action
        self.actionSobel = QtWidgets.QAction(self)
        self.actionSobel.setText("Sobel")
        self.actionSobel.setStatusTip("Sobel")
        self.actionSobel.setObjectName("actionSobel")

        # Creating Laplacian action
        self.actionLaplacian = QtWidgets.QAction(self)
        self.actionLaplacian.setText("Laplacian")
        self.actionLaplacian.setStatusTip("Laplacian")
        self.actionLaplacian.setObjectName("actionLaplacian")

        # Creating Canny action
        self.actionCanny = QtWidgets.QAction(self)
        self.actionCanny.setText("Canny")
        self.actionCanny.setStatusTip("Canny")
        self.actionCanny.setObjectName("actionCanny")

        # Creating Mask action
        self.actionMask = QtWidgets.QAction(self)
        self.actionMask.setText("Mask")
        self.actionMask.setStatusTip("Mask")
        self.actionMask.setObjectName("actionMask")

        # Adding actions to Lab3 menu
        self.menu_lab3.addAction(self.actionBlur)
        self.menu_lab3.addAction(self.actionSobel)
        self.menu_lab3.addAction(self.actionLaplacian)
        self.menu_lab3.addAction(self.actionCanny)
        self.menu_lab3.addAction(self.actionMask)

        # Adding menu3 to menubar
        self.menubar.addAction(self.menu_lab3.menuAction())

        # Connecting signals to actions
        self.actionBlur.triggered.connect(self.blur_image)
        self.actionSobel.triggered.connect(self.sobel_image)
        self.actionLaplacian.triggered.connect(self.laplacian_image)
        self.actionCanny.triggered.connect(self.canny_image)
        self.actionMask.triggered.connect(self.mask_image)

        # Creating Lab4 menu
        self.menu_lab4 = QtWidgets.QMenu(self.menubar)
        self.menu_lab4.setObjectName("menuLab4")
        self.menu_lab4.setTitle("Lab4")

        # Creating Erozja action
        self.action_erozja = QtWidgets.QAction(self)
        self.action_erozja.setText("Erozja")
        self.action_erozja.setStatusTip("Erozja")
        self.action_erozja.setObjectName("actionErozja")

        # Creating Dylacja action
        self.action_dylacja = QtWidgets.QAction(self)
        self.action_dylacja.setText("Dylacja")
        self.action_dylacja.setStatusTip("Dylacja")
        self.action_dylacja.setObjectName("actionDylacja")

        # Creating Otwarcie action
        self.action_open = QtWidgets.QAction(self)
        self.action_open.setText("Otwarcie")
        self.action_open.setStatusTip("Otwarcie")
        self.action_open.setObjectName("actionOtwarcie")

        # Creating Zamkniecie action
        self.action_close = QtWidgets.QAction(self)
        self.action_close.setText("Zamkniecie")
        self.action_close.setStatusTip("Zamkniecie")
        self.action_close.setObjectName("Zamkniecie")

        # Creating Filtracja action
        self.action_filtration = QtWidgets.QAction(self)
        self.action_filtration.setText("Filtracja")
        self.action_filtration.setStatusTip("Filtracja")
        self.action_filtration.setObjectName("actionFiltracja")

        # Creating Skeletonize action
        self.action_skeletonize = QtWidgets.QAction(self)
        self.action_skeletonize.setText("Skeletonize")
        self.action_skeletonize.setStatusTip("Skeletonize")
        self.action_skeletonize.setObjectName("actionSkeletonize")

        # Adding actions to Lab4 menu
        self.menu_lab4.addAction(self.action_erozja)
        self.menu_lab4.addAction(self.action_dylacja)
        self.menu_lab4.addAction(self.action_open)
        self.menu_lab4.addAction(self.action_close)
        self.menu_lab4.addAction(self.action_filtration)
        self.menu_lab4.addAction(self.action_skeletonize)

        # Adding menu4 to menubar
        self.menubar.addAction(self.menu_lab4.menuAction())

        # Connecting signals to actions
        self.action_erozja.triggered.connect(self.erozja)
        self.action_dylacja.triggered.connect(self.dylacja)
        self.action_open.triggered.connect(self.open)
        self.action_close.triggered.connect(self.close)
        self.action_filtration.triggered.connect(self.filtration_image)
        self.action_skeletonize.triggered.connect(self.skeletonize_image)

        # Creating Lab5 menu
        self.menu_lab5 = QtWidgets.QMenu(self.menubar)
        self.menu_lab5.setObjectName("menuLab5")
        self.menu_lab5.setTitle("Lab5")

        # Creating Save action
        self.action_save = QtWidgets.QAction(self)
        self.action_save.setText("Save")
        self.action_save.setStatusTip("Save")
        self.action_save.setObjectName("actionSave")

        # Adding actions to Lab5 menu
        self.menu_lab5.addAction(self.action_save)

        # Adding menu5 to menubar
        self.menubar.addAction(self.menu_lab5.menuAction())

        # Connecting signals to actions
        self.action_save.triggered.connect(self.save_image)

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
        self.update_image()

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
            self.update_image()
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
            self.update_image()
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
            self.update_image()
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

    def stretch_histogram(self):
        if self.is_gray:
            self.image = self.stretch_gray_histogram(self.image)
            height, width = self.image.shape
            bytes_per_line = width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Indexed8)
            self.label.setPixmap(QPixmap.fromImage(q_img))
        else:
            b, g, r = cv2.split(self.image)
            b = self.stretch_gray_histogram(b)
            g = self.stretch_gray_histogram(g)
            r = self.stretch_gray_histogram(r)
            self.image = cv2.merge((b, g, r))
            height, width, _ = self.image.shape
            bytes_per_line = width * 3
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_img))
        self.update_image()

    def stretch_gray_histogram(self, img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf[img]
        return img2

    def equalize_histogram(self):
        if self.is_gray:
            self.image = self.equalize_gray_histogram(self.image)
            height, width = self.image.shape
            bytes_per_line = width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Indexed8)
            self.label.setPixmap(QPixmap.fromImage(q_img))
        else:
            b, g, r = cv2.split(self.image)
            b = self.equalize_gray_histogram(b)
            g = self.equalize_gray_histogram(g)
            r = self.equalize_gray_histogram(r)
            self.image = cv2.merge((b, g, r))
            height, width, _ = self.image.shape
            bytes_per_line = width * 3
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_img))
        self.update_image()
    def equalize_gray_histogram(self, img):
        equ = cv2.equalizeHist(img)
        #self.update_image()
        return equ
    def update_image(self):
        # Convert the image data to a QImage object
        q_img = self.convert_cv_to_qimage(self.image)
        # Set the converted QImage as the pixmap for the QLabel widget
        self.label.setPixmap(QPixmap.fromImage(q_img))
    def negation(self):
        self.image = 255 - self.image
        self.update_image()

    def posterization(self):
        levels = 8  # Number of levels for posterization
        self.image = np.floor_divide(self.image, 256 // levels) * (256 // levels)
        self.update_image()

    def selective_stretching(self):
        p1 = 0  # Lower input range
        p2 = 127  # Upper input range
        q3 = 64  # Lower output range
        q4 = 255  # Upper output range

        if self.is_gray:
            self.image = np.interp(self.image, [p1, p2], [q3, q4])
        else:
            h, w, _ = self.image.shape
            r, g, b = cv2.split(self.image)
            r = np.interp(r, [p1, p2], [q3, q4])
            g = np.interp(g, [p1, p2], [q3, q4])
            b = np.interp(b, [p1, p2], [q3, q4])
            self.image = cv2.merge([r, g, b])

        self.image = np.clip(self.image, 0, 255)
        self.image = self.image.astype(np.uint8)
        self.update_image()

    def blur_image(self):
        number, ok = QInputDialog.getInt(self, "Blur", "Enter a number", value=0, min=0, max=255)
        if ok:
            items = ("Isolated", "Reflect", "Replicate")
            item, ok = QInputDialog.getItem(self, "Blur", "Choose border:", items, 0, False)
            if ok:
                if item == "Isolated":
                    border = cv2.BORDER_ISOLATED
                elif item == "Reflect":
                    border = cv2.BORDER_REFLECT
                else:
                    border = cv2.BORDER_REPLICATE
                self.image = self.blur(number, border)
                self.update_image()

    def blur(self, ksize, border):
        self.image = cv2.blur(self.image, (ksize, ksize), border)
        return self.image

    def sobel_image(self):
        number, ok = QInputDialog.getInt(self, "Sobel", "Enter a number", value=0, min=0)
        if ok:
            self.image = self.sobel(number)
            self.update_image()

    def sobel(self, kernel):
        sobel_x = cv2.Sobel(self.image, cv2.CV_8UC1, 1, 0, kernel)
        sobel_y = cv2.Sobel(self.image, cv2.CV_8UC1, 0, 1, kernel)
        self.image = cv2.hconcat((sobel_x, sobel_y))
        return self.image

    def laplacian_image(self):
        number, ok = QInputDialog.getInt(self, "Laplacian", "Enter a number", value=0, min=0)
        if ok:
            items = ("Isolated", "Reflect", "Replicate")
            item, ok = QInputDialog.getItem(self, "Laplacian", "Choose border:", items, 0, False)
            if ok:
                if item == "Isolated":
                    border = cv2.BORDER_ISOLATED
                elif item == "Reflect":
                    border = cv2.BORDER_REFLECT
                else:
                    border = cv2.BORDER_REPLICATE
                self.image = self.laplacian(number, border)
                self.update_image()

    def laplacian(self, kernel, border):
        ddepth = cv2.CV_8UC1
        self.image = cv2.Laplacian(self.image, ddepth, ksize=kernel, borderType=border)
        return self.image

    def canny_image(self):
        number1, ok = QInputDialog.getInt(self, "Canny", "Enter a number", value=0, min=0)
        if ok:
            number2, ok = QInputDialog.getInt(self, "Canny", "Enter a number", value=0, min=0)
            if ok:
                self.image = self.canny(number1, number2)
                self.update_image()

    def canny(self, threshold1, threshold2):
        self.image = cv2.Canny(self.image, threshold1, threshold2)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        return self.image

    def mask_image(self):
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Mask", "Choose mask:", ['0,-1,0\n-1,5,-1\n0,-1,0\n',
                                                                                   '-1,-1,-1\n-1,9,-1\n-1,-1,-1\n',
                                                                                   '1,-2,1\n-2,5,-2\n1,-2,1\n'])
        if ok:
            if choice == '0,-1,0\n-1,5,-1\n0,-1,0\n':
                mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            elif choice == '-1,-1,-1\n-1,9,-1\n-1,-1,-1\n':
                mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            else:
                mask = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
            items = ("Isolated", "Reflect", "Replicate")
            item, ok = QtWidgets.QInputDialog.getItem(self, "Mask", "Choose border:", items, 0, False)
            if ok:
                if item == "Isolated":
                    border = cv2.BORDER_ISOLATED
                elif item == "Reflect":
                    border = cv2.BORDER_REFLECT
                else:
                    border = cv2.BORDER_REPLICATE
                self.image = self.mask(mask, border)
                self.update_image()

    def mask(self, mask, border):
        self.image = cv2.filter2D(self.image, cv2.CV_8UC1, mask, border)
        return self.image

    def skeletonize_image(self):
        self.skeletonize()
        self.update_image()

    def skeletonize(self):
        img = cv2.imread(self.fullName, 0)
        ret, img = cv2.threshold(img, 127, 255, 0)

        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            # Step 2: Open the image
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            # Step 3: Substract open from the original image
            temp = cv2.subtract(img, open)
            # Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(img) == 0:
                break

        skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.hconcat((self.image, skel))
        return self.image
    def erozja(self):
        kernel = np.ones((3, 3), np.uint8)
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Operacja erozji", "Wybierz otoczenie:",
                                                    ['4 sasiadow', '8 sasiadow'])
        if ok:
            if choice == '4 sasiadow':
                kernel[0, 0] = 0  # kernel[0][0]
                kernel[0, 2] = 0
                kernel[2, 0] = 0
                kernel[2, 2] = 0
            self.image[:, :, :3] = cv2.erode(self.image[:, :, :3], kernel, iterations=1)
            self.update_image()

    def dylacja(self):
        kernel = np.ones((3, 3), np.uint8)
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Operacja erozji", "Wybierz otoczenie:",
                                                    ['4 sasiadow', '8 sasiadow'])
        if ok:
            if choice == '4 sasiadow':
                kernel[0, 0] = 0  # kernel[0][0]
                kernel[0, 2] = 0
                kernel[2, 0] = 0
                kernel[2, 2] = 0
            self.image[:, :, :3] = cv2.dilate(self.image[:, :, :3], kernel, iterations=1)
            self.update_image()

    def open(self):
        kernel = np.ones((3, 3), np.uint8)

        choice, ok = QtWidgets.QInputDialog.getItem(self, "Operacja erozji",
                                                    "Wybierz otoczenie:", ['4 sasiadow', '8 sasiadow'])
        if ok:
            if choice == '4 sasiadow':
                kernel[0, 0] = 0  # kernel[0][0]
                kernel[0, 2] = 0
                kernel[2, 0] = 0
                kernel[2, 2] = 0

            self.image[:, :, :3] = cv2.erode(self.image[:, :, :3], kernel, iterations=1)
            self.image[:, :, :3] = cv2.dilate(self.image[:, :, :3], kernel, iterations=1)
            self.update_image()

    def close(self):
        kernel = np.ones((3, 3), np.uint8)

        choice, ok = QtWidgets.QInputDialog.getItem(self, "Operacja erozji",
                                                    "Wybierz otoczenie:", ['4 sasiadow', '8 sasiadow'])
        if ok:
            if choice == '4 sasiadow':
                kernel[0, 0] = 0  # kernel[0][0]
                kernel[0, 2] = 0
                kernel[2, 0] = 0
                kernel[2, 2] = 0
            self.image[:, :, :3] = cv2.dilate(self.image[:, :, :3], kernel, iterations=1)
            self.image[:, :, :3] = cv2.erode(self.image[:, :, :3], kernel, iterations=1)
            self.update_image()

    def filtration(self, mask1, mask2, item):
        # konstrukcja maski w oparciu o dwie powyższe maski 3x3
        # wykorzystanie konwolucji do wygenerowania maski 5x5
        from scipy.signal import convolve2d as conv2  # funkcja konwolucji dwuwymiraowej
        mH = conv2(mask1, mask2, mode='full')  # mode full zapewnia odpowieni rozmiar maski

        # wykonanie dwu etapowej filtracji z maskami 3x3 (jak w Lab3)
        res_step1 = cv2.filter2D(self.image, cv2.CV_64F, mask1, item)
        res_step2 = cv2.filter2D(res_step1, cv2.CV_64F, mask2, item)
        # cv2_imshow(res_step22)

        # wykonanie jednoetapowej filtracji z maską 5x5
        res_5x5 = cv2.filter2D(self.image, cv2.CV_64F, mH, item)
        # cv2_imshow(res_5x5)

        # wizualne porównanie wyników
        self.image = cv2.hconcat((np.uint8(res_step2), np.uint8(res_5x5)))
        return self.image

    def filtration_image(self):
        choice1, ok = QtWidgets.QInputDialog.getItem(self, "Wybór maske wygładzenia",
                                                     "Wybierz macierz:", ['0,-1,0\n-1,5,-1\n0,-1,0\n',
                                                                          '-1,-1,-1\n-1,9,-1\n-1,-1,-1\n',
                                                                          '1,-2,1\n-2,5,-2\n1,-2,1\n'])
        if ok:
            if choice1 == "[[ 0,-1, 0]\n[-1, 5,-1]\n[ 0,-1, 0]]":
                mask1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            elif choice1 == "[[-1,-1,-1]\n[-1, 9,-1]\n[-1,-1,-1]]":
                mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            mask1 = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])

            choice2, ok = QtWidgets.QInputDialog.getItem(self, "Wybór maske wyostrzania",
                                                         "Wybierz macierz:", ['0,-1,0\n-1,5,-1\n0,-1,0\n',
                                                                              '-1,-1,-1\n-1,9,-1\n-1,-1,-1\n',
                                                                              '1,-2,1\n-2,5,-2\n1,-2,1\n'])
            if ok:
                if choice2 == "[[ 0,-1, 0]\n[-1, 5,-1]\n[ 0,-1, 0]]":
                    mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                elif choice2 == "[[-1,-1,-1]\n[-1, 9,-1]\n[-1,-1,-1]]":
                    mask2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                mask2 = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])

                items = ("Isolated", "Reflect", "Replicate")
                item, ok = QInputDialog.getItem(self, "Blur",
                                                "Choose border:", items, 0, False)
                if ok:
                    if item == "Isolated":
                        item = cv2.BORDER_ISOLATED
                    elif item == "Reflect":
                        item = cv2.BORDER_REFLECT
                    item = cv2.BORDER_REPLICATE
                    self.image = self.filtration(mask1, mask2, item)
                    self.update_image()

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save file...", "",
                                                   "Image files (*.jpg *.jpeg *.bmp *.png *.tiff *.tif)")
        if file_name:
              cv2.imwrite(file_name, self.image)

        else:
            self.label.setText("Choose an image")

class UiMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.setObjectName("MainWindow")
        self.resize(640, 200)
        self.setWindowTitle("APO Ruslan Lytvynov : Image Project")

        self.windows = []

        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setObjectName("centralWidget")

        self.label = QtWidgets.QLabel(self.central_widget)
        self.label.setGeometry(QtCore.QRect(250, 0, 500, 60))
        self.label.setObjectName("label")
        self.label.setText("Click 'File' -> 'Open' to choose an image.")

        self.setCentralWidget(self.central_widget)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 570, 30))
        self.menubar.setObjectName("menubar")

        self.file_menu = QtWidgets.QMenu(self.menubar)
        self.file_menu.setObjectName("fileMenu")
        self.file_menu.setTitle("File")

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

        self.file_menu.addAction(self.action_open)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_exit)

        self.menubar.addAction(self.file_menu.menuAction())

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
