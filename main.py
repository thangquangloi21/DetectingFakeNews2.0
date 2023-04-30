
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model
# from model import padding_type, trunc_type

model = load_model("model.h5")
trunc_type = 'post'
padding_type = 'post'

# Function to check if news is real or fake
def check_news(news):
    sequences = Tokenizer().texts_to_sequences([news])[0]
    sequences = pad_sequences([sequences], maxlen=54,
                              padding=padding_type,
                              truncating=trunc_type)
    if (model.predict(sequences, verbose=0)[0][0] >= 0.5):
        return True
    else:
        return False

# Function to process user input and display response
def send_message():
    # Get user input
    user_input = user_input_box.get()
    # Check if input is empty
    if not user_input:
        return
    # Display user input in chat window
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "You: " + user_input + "\n")
    chat_window.config(state=tk.DISABLED)
    # Check if news is real or fake
    is_real = check_news(user_input)
    # Display response in chat window
    chat_window.config(state=tk.NORMAL)
    if is_real:
        chat_window.insert(tk.END, "OUR PROPOSED: That's a REAL news.\n")
    else:
        chat_window.insert(tk.END, "OUR PROPOSED: That's a FAKE news.\n")
    chat_window.config(state=tk.DISABLED)
    # Clear user input box
    user_input_box.delete(0, tk.END)

# Create main window
window = tk.Tk()
window.title("FAKE NEWS IS OF PROPOSED")

# Create chat window
chat_window = tk.Text(window, state=tk.DISABLED, height=20, width=50, font=("Helvetica", 12))
chat_window.pack(padx=10, pady=10)

# Create user input box
user_input_box = tk.Entry(window, width=50, font=("Helvetica", 12))
user_input_box.pack(padx=10, pady=10)

# Create send button
send_button = tk.Button(window, text="Send", command=send_message, font=("Helvetica", 12))
send_button.pack(padx=10, pady=10)
# Set background color for window
window.configure(background="#f0f0f0")

# Set foreground color and font for chat window
chat_window.configure(foreground="#000000", font=("Helvetica", 12))

# Set background color, foreground color, and font for user input box
user_input_box.configure(background="#ffffff", foreground="#000000", font=("Helvetica", 12))

# Set background color, foreground color, and font for send button
send_button.configure(background="#008000", foreground="#ffffff", font=("Helvetica", 12))

# Run main loop
window.mainloop()
