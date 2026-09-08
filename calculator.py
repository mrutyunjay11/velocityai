import tkinter as tk
from tkinter import ttk
import math

class CalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VelocityAI Python Calculator")
        self.root.geometry("340x460")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")

        self.expression = ""
        self.equation = tk.StringVar()
        self.equation.set("0")

        # Top Display Frame
        display_frame = tk.Frame(self.root, bg="#1e1e1e")
        display_frame.pack(expand=True, fill="both", padx=15, pady=(15, 10))

        # Entry Display
        self.display = tk.Entry(
            display_frame,
            textvariable=self.equation,
            font=("Arial", 28, "bold"),
            bg="#2d2d2d",
            fg="#ffffff",
            bd=0,
            justify="right",
            relief="flat"
        )
        self.display.pack(expand=True, fill="both", ipady=10)

        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(expand=True, fill="both", padx=15, pady=(0, 15))

        for i in range(5):
            btn_frame.rowconfigure(i, weight=1)
        for j in range(4):
            btn_frame.columnconfigure(j, weight=1)

        # Button Layout: (Text, Row, Col, Color, Action)
        buttons = [
            ("C", 0, 0, "#a5a5a5", self.clear),
            ("⌫", 0, 1, "#a5a5a5", self.backspace),
            ("%", 0, 2, "#a5a5a5", lambda: self.press("%")),
            ("÷", 0, 3, "#fe9e09", lambda: self.press("/")),
            
            ("7", 1, 0, "#3a3a3c", lambda: self.press("7")),
            ("8", 1, 1, "#3a3a3c", lambda: self.press("8")),
            ("9", 1, 2, "#3a3a3c", lambda: self.press("9")),
            ("×", 1, 3, "#fe9e09", lambda: self.press("*")),
            
            ("4", 2, 0, "#3a3a3c", lambda: self.press("4")),
            ("5", 2, 1, "#3a3a3c", lambda: self.press("5")),
            ("6", 2, 2, "#3a3a3c", lambda: self.press("6")),
            ("-", 2, 3, "#fe9e09", lambda: self.press("-")),
            
            ("1", 3, 0, "#3a3a3c", lambda: self.press("1")),
            ("2", 3, 1, "#3a3a3c", lambda: self.press("2")),
            ("3", 3, 2, "#3a3a3c", lambda: self.press("3")),
            ("+", 3, 3, "#fe9e09", lambda: self.press("+")),
            
            ("±", 4, 0, "#3a3a3c", self.toggle_sign),
            ("0", 4, 1, "#3a3a3c", lambda: self.press("0")),
            (".", 4, 2, "#3a3a3c", lambda: self.press(".")),
            ("=", 4, 3, "#fe9e09", self.evaluate),
        ]

        for text, r, c, bg_color, cmd in buttons:
            fg_color = "#000000" if bg_color == "#a5a5a5" else "#ffffff"
            btn = tk.Button(
                btn_frame,
                text=text,
                font=("Arial", 16, "bold"),
                bg=bg_color,
                fg=fg_color,
                activebackground="#555555",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                command=cmd
            )
            btn.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

        # Keyboard bindings
        self.root.bind("<Key>", self.handle_key)

    def press(self, val):
        if self.equation.get() == "0" and val not in "+-*/%":
            self.expression = str(val)
        else:
            self.expression += str(val)
        self.equation.set(self.expression)

    def clear(self):
        self.expression = ""
        self.equation.set("0")

    def backspace(self):
        self.expression = self.expression[:-1]
        self.equation.set(self.expression if self.expression else "0")

    def toggle_sign(self):
        try:
            if self.expression:
                val = float(self.expression)
                val = -val
                self.expression = str(int(val) if val.is_integer() else val)
                self.equation.set(self.expression)
        except Exception:
            pass

    def evaluate(self):
        try:
            # Safe evaluation
            total = str(eval(self.expression))
            # Format float if integer
            if "." in total and float(total).is_integer():
                total = str(int(float(total)))
            self.equation.set(total)
            self.expression = total
        except ZeroDivisionError:
            self.equation.set("Error: Div by 0")
            self.expression = ""
        except Exception:
            self.equation.set("Error")
            self.expression = ""

    def handle_key(self, event):
        if event.char in "0123456789+-*/.%":
            self.press(event.char)
        elif event.keysym in ("Return", "KP_Enter"):
            self.evaluate()
        elif event.keysym == "BackSpace":
            self.backspace()
        elif event.keysym == "Escape":
            self.clear()

if __name__ == "__main__":
    root = tk.Tk()
    app = CalculatorGUI(root)
    root.mainloop()
