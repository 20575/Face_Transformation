# waouh_gui.py - Beautiful Tkinter GUI for Gender WAOUH Transformation
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk

class WaouhApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WAOUH Transformation - From Man to Stunning Woman")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)

        self.original_path = None
        self.processed_image = None

        self.setup_ui()

    def setup_ui(self):
        title = tk.Label(self.root, text="WAOUH Face Transformation", font=("Helvetica", 24, "bold"),
                         bg="#f0f0f0", fg="#ff6b9d")
        title.pack(pady=20)

        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 11), padding=10)

        self.btn_choose = ttk.Button(btn_frame, text="Choose Photo", command=self.load_image)
        self.btn_choose.grid(row=0, column=0, padx=20)

        self.btn_transform = ttk.Button(btn_frame, text="Transform to WAOUH", command=self.transform,
                                        style="Accent.TButton")
        style.configure("Accent.TButton", background="#ff6b9d", foreground="white")
        self.btn_transform.grid(row=0, column=1, padx=20)
        self.btn_transform.config(state="disabled")

        self.img_frame = tk.Frame(self.root, bg="white")
        self.img_frame.pack(expand=True, fill="both", padx=30, pady=20)

        self.label_orig = tk.Label(self.img_frame, text="Original", font=("Helvetica", 14, "bold"),
                                   bg="white", fg="#555")
        self.label_orig.grid(row=0, column=0, pady=10)
        self.panel_orig = tk.Label(self.img_frame, bg="white", relief="sunken")
        self.panel_orig.grid(row=1, column=0, padx=20)

        arrow = tk.Label(self.img_frame, text="→", font=("Helvetica", 40), bg="#f0f0f0", fg="#ff6b9d")
        arrow.grid(row=1, column=1)

        self.label_result = tk.Label(self.img_frame, text="WAOUH Result", font=("Helvetica", 14, "bold"),
                                     bg="white", fg="#ff6b9d")
        self.label_result.grid(row=0, column=2, pady=10)
        self.panel_result = tk.Label(self.img_frame, bg="white", relief="sunken")
        self.panel_result.grid(row=1, column=2, padx=20)

        footer = tk.Label(self.root, text="Peau de poupée • Maquillage pro • Cheveux longs • 100% réaliste",
                          font=("Helvetica", 10), bg="#f0f0f0", fg="#888")
        footer.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.original_path = path
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_image(img, self.panel_orig)
            self.btn_transform.config(state="normal")
            self.panel_result.config(image='', text="Waiting for transformation...", fg="#aaa")

    def display_image(self, img, panel):
        img = cv2.resize(img, (450, 550), interpolation=cv2.INTER_LANCZOS4)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk

    def waouh_transform(self, img):
        h, w = img.shape[:2]
        up = cv2.pyrUp(cv2.pyrUp(img))
        smooth = cv2.bilateralFilter(up, 25, 120, 120)
        smooth = cv2.bilateralFilter(smooth, 15, 100, 100)
        detail = cv2.detailEnhance(smooth, sigma_s=30, sigma_r=0.3)
        lab = cv2.cvtColor(detail, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a = np.clip(a.astype(np.int16) + 18, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.int16) - 12, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        makeup = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        final = cv2.addWeighted(detail, 0.55, makeup, 0.45, 0)
        final = cv2.convertScaleAbs(final, alpha=1.18, beta=12)
        final = cv2.resize(final, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return final

    def transform(self):
        if not self.original_path:
            return
        img = cv2.imread(self.original_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.panel_result.config(image='', text="Transforming... Please wait", fg="#ff6b9d")
        self.root.update()
        result = self.waouh_transform(img)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self.display_image(img_rgb, self.panel_orig)
        self.display_image(result_rgb, self.panel_result)
        save_path = os.path.join("WAOUH_RESULTS", f"WAOUH_{Path(self.original_path).stem}.png")
        os.makedirs("WAOUH_RESULTS", exist_ok=True)
        cv2.imwrite(save_path, result)
        messagebox.showinfo("Success!", f"WAOUH saved !\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WaouhApp(root)
    root.mainloop()