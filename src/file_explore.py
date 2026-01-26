import tkinter as tk
from tkinter import filedialog

# Hide the main Tk window
root = tk.Tk()
root.withdraw()

# Open file dialog
file_path = filedialog.askopenfilename(
    title="Select a file",
    filetypes=[("PyTorch Model", "*.pt")]  # Filter by type (change "*.*" to "*.txt" for only txt for example)
)

print("Selected file:", file_path)