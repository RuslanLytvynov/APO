import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np


# Funkcja do wczytywania obrazu
def open_image():
    filepath = filedialog.askopenfilename(title="Select image file",
                                          filetypes=[("BMP files", "*.bmp"),
                                                     ("JPEG files", "*.jpg;*.jpeg"),
                                                     ("PNG files", "*.png"),
                                                     ("GIF files", "*.gif")])
    if filepath:
        image = Image.open(filepath).convert("L")  # Konwersja do obrazu szaroodcieniowego
        image_tk = ImageTk.PhotoImage(image)
        canvas.image_tk = image_tk
        canvas.create_image(0, 0, anchor="nw", image=image_tk)
        canvas.config(width=image.width, height=image.height)
        update_histogram(image)


# Funkcja do aktualizowania histogramu
def update_histogram(image):
    histogram = np.zeros(256)
    for pixel in image.getdata():
        histogram[pixel] += 1
    histogram /= image.size[0] * image.size[1]  # Normalizacja
    histogram_canvas.delete("all")
    for i, value in enumerate(histogram):
        x0, y0, x1, y1 = i, 256, i + 1, 256 - value * 256
        histogram_canvas.create_rectangle(x0, y0, x1, y1, fill="gray")


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

root.mainloop()
