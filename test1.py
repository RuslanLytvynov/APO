# Importowanie bibliotek
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2

# Funkcja do wczytywania obrazu szaroodcieniowego lub kolorowego
def open_image():
    filepath = filedialog.askopenfilename(title="Select image file",
                                         filetypes=[("BMP files", ".bmp"),
                                                    ("JPEG files", ".jpg;.jpeg"),
                                                    ("PNG files", ".png"),
                                                    ("GIF files", "*.gif")])

    if filepath:
        image = Image.open(filepath)
        if image.mode == "L":  # Sprawdzanie, czy obraz jest szaroodcieniowy
            image_tk = ImageTk.PhotoImage(image)
            canvas.image_tk = image_tk
            canvas.create_image(0, 0, anchor="nw", image=image_tk)
            canvas.config(width=image.width, height=image.height)
            update_grayscale_histogram(image)
        else:  # Obraz jest kolorowy
            image_tk = ImageTk.PhotoImage(image)
            canvas.image_tk = image_tk
            canvas.create_image(0, 0, anchor="nw", image=image_tk)
            canvas.config(width=image.width, height=image.height)
            update_color_histograms(image)

# Funkcja do aktualizowania histogramu dla obrazu szaroodcieniowego
def update_grayscale_histogram(image):
    histogram = np.zeros(256)

    for pixel in image.getdata():
        histogram[pixel] += 1
    histogram /= image.size[0] * image.size[1]  # Normalizacja
    histogram_canvas.delete("all")
    for i, value in enumerate(histogram):
        x0, y0, x1, y1 = i, 256, i + 1, 256 - int(value * 256)
    histogram_canvas.create_rectangle(x0, y0, x1, y1, fill="gray")

# Funkcja do aktualizowania histogramów dla obrazu kolorowego
def update_color_histograms(image):
    channels = cv2.split(np.array(image))
    colors = ("blue", "green", "red")
    histogram_canvas.delete("all")
    for i, channel in enumerate(channels):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        histogram /= channel.size  # Normalizacja
        for j, value in enumerate(histogram):
            x0, y0, x1, y1 = i * 85 + j, 256, i * 85 + j + 1, 256 - int(value * 256)
        histogram_canvas.create_rectangle(x0, y0, x1, y1, fill=colors[i])

# Funkcja do konwersji obrazu kolorowego na szaroodcieniowy
def convert_to_grayscale(image):
    grayscale_image = image.convert("L")
    image_tk = ImageTk.PhotoImage(grayscale_image)
    canvas.image_tk = image_tk
    canvas.create_image(0, 0, anchor="nw", image=image_tk)

# Aktualizowanei wartości histogramu po każdej modyfikacji
def update_histogram(image, histogram):
    for i in range(len(image)):
        for j in range(len(image[0])):
            pixel_value = image[i][j]
            histogram[pixel_value] += 1


# Tworzenie okna programu
root = tk.Tk()
root.title("Algorytmy przetwarzania obrazów")

# Tworzenie menu
menu = tk.Menu(root)
root.config(menu=menu)

file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=open_image)

# Tworzenie ramki z canvasami
frame = tk.Frame(root)
frame.pack(side="top", fill="both", expand=True)



canvas = tk.Canvas(frame)
canvas.pack(side="left", fill="both", expand=True)

histogram_canvas = tk.Canvas(frame, width=256, height=256)
histogram_canvas.pack(side="right", fill="y")

# Ustawienie minimalnych rozmiarów okna
root.update_idletasks()
root.minsize(root.winfo_width(), root.winfo_height())

# Uruchomienie pętli głównej i wyświetlenie okna
root.mainloop()