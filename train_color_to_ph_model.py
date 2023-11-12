import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example data
color_names = ["red", "blue", "green", "yellow", "purple"]
ph_values = [3.0, 7.0, 8.5, 6.0, 4.5]

# Tokenize the color names
tokenizer = Tokenizer()
tokenizer.fit_on_texts(color_names)
vocab_size = len(tokenizer.word_index) + 1

# Convert color names to sequences and pad them
sequences = tokenizer.texts_to_sequences(color_names)
padded_sequences = pad_sequences(sequences)

# Convert ph_values to numpy array
ph_values = np.array(ph_values)

# Save the tokenizer's word index
with open("word_index.json", "w") as f:
    json.dump(tokenizer.word_index, f)

# Save the maximum sequence length
max_sequence_length = len(padded_sequences[0])
with open("max_sequence_length.txt", "w") as f:
    f.write(str(max_sequence_length))

# Define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(np.array(padded_sequences), ph_values, epochs=10)

# Save the model
model.save("color_to_ph_model.h5")
