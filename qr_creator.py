import tkinter as tk
from tkinter import simpledialog
import qrcode
from PIL import Image, ImageTk
import io


def generate_qr_code(link):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(link)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    pil_image = Image.open(buf)
    return pil_image


def display_qr():
    global photo_label

    url = url_entry.get()

    img = generate_qr_code(url)

    # Convert PIL image to a format Tkinter can use
    tk_image = ImageTk.PhotoImage(img)

    # If a QR code image is already displayed, update it
    if 'photo_label' in globals():
        photo_label.config(image=tk_image)
        photo_label.image = tk_image
    else:
        # Create a label to display the image and pack it into the window
        photo_label = tk.Label(root, image=tk_image)
        photo_label.image = tk_image
        photo_label.pack(side="bottom")


# Main window
root = tk.Tk()
root.title("QR Code Generator")

# Frame for input & button
frame = tk.Frame(root)
frame.pack(pady=20)

# Entry widget for URL
url_entry = tk.Entry(frame, width=40)
url_entry.pack(side="left", padx=(0, 10))

# Button to generate QR code
generate_button = tk.Button(frame, text="Generate QR Code", command=display_qr)
generate_button.pack(side="right")

# App
root.mainloop()
