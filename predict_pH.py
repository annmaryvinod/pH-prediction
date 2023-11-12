import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model("color_to_ph_model.h5")

# Load the tokenizer's word index
with open("word_index.json", "r") as f:
    word_index = json.load(f)

tokenizer = Tokenizer()
tokenizer.word_index = word_index

# Load the maximum sequence length
with open("max_sequence_length.txt", "r") as f:
    max_sequence_length = int(f.read())

# Tokenize and pad input for prediction
new_color_name = ["orange"]  # Replace with your desired color name
new_sequence = tokenizer.texts_to_sequences(new_color_name)
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_sequence_length)

# Make prediction
predicted_ph = model.predict(new_padded_sequence)[0][0]

print(f"Predicted pH value for {new_color_name[0]}: {predicted_ph}")
