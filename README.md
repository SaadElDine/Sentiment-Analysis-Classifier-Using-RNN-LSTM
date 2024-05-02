# Sentiment Analysis of IMDb Reviews Using LSTM and RNN

This project demonstrates the utilization of Long Short-Term Memory (LSTM) and Recurrent Neural Network (RNN) models to perform sentiment analysis on IMDb movie reviews. The objective is to compare the effectiveness of these models in understanding natural language and to emphasize the importance of early stopping in training neural networks.

## Project Setup:

### Platform
- Google Colaboratory
  
### Data Source
- IMDb Dataset hosted on Google Drive
  
### Key Libraries
- pandas
- numpy
- TensorFlow (with Keras backend)
- scikit-learn
- NLTK for Natural Language Processing
- Google Colab for Google Drive integration
  
## Data Loading and Preprocessing
1. **Mount Google Drive** to access the IMDb dataset stored there.
    ```
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2. **Load the dataset** as a pandas DataFrame.
    ```
    import pandas as pd
    df = pd.read_csv("/content/drive/MyDrive/IMDB Dataset.csv")
    ```
3. **Clean and prepare the text** by removing noise and preparing it for model ingestion.
    ```
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```
4. **Tokenize and pad the text** to ensure a uniform input length.
    ```
    from tensorflow.keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['review'])
    X_seq = tokenizer.texts_to_sequences(df['review'])
    X_pad = pad_sequences(X_seq, maxlen=200)
    ```
5. **Encode the labels** to transform categorical data into a format suitable for neural network training.
    ```
    from sklearn.preprocessing import LabelEncoder
    y = df['sentiment']
    y_enc = le.fit_transform(y)
    ```
Here, the sentiment column from the DataFrame is encoded into a numeric format that the neural network can work with. Typically, sentiments like 'positive' or 'negative' are transformed to a binary format, with one representing 'positive' and zero representing 'negative'.
## Model Building
Two types of neural network architectures are used:
1. **LSTM (Long Short-Term Memory) Model**: Builds a multi-layer LSTM model to process sequences and capture temporal dependencies. 
   ```
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    ```
2. **RNN (Recurrent Neural Network) Model**: Constructs a simpler RNN model for benchmarking against the LSTM.
    ```
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim))
    model.add(SimpleRNN(128, dropout=0.7))
    model.add(Dropout(0.09))
    model.add(Dense(1, activation='sigmoid'))Dense(units=1, activation='sigmoid')
    ])
    ```
These are the core layers. You generally start with an Embedding layer, then add LSTM or RNN layers, and end with Dense layers for output.
## Model Training with Early Stopping
# Training LSTM model
 ```
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_history = model_lstm.fit(
    X_pad, y_enc, 
    epochs=10, 
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)
```

# Training RNN model
 ```
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_history = model_rnn.fit(
    X_pad, y_enc,
    epochs=10, 
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)
```

# Model Evaluation and Saving
After training, the models are evaluated on unseen test data to assess their generalization capabilities.
```
    lstm_loss, lstm_accuracy = model_lstm.evaluate(X_test, y_test)
    rnn_loss, rnn_accuracy = model_rnn.evaluate(X_test, y_test)
    print(f'LSTM Model - Loss: {lstm_loss}, Accuracy: {lstm_accuracy}')
    print(f'RNN Model - Loss: {rnn_loss}, Accuracy: {rnn_accuracy}')
```

# Conclusion
This project highlighted the use of LSTM and RNN for sentiment analysis of IMDb reviews, demonstrating their capabilities and differences in handling sequential text data. The employment of early stopping helped significantly in preventing overfitting and optimizing the training phase. The overall analysis can be used to enhance the development of more advanced NLP models or to refine existing models for better accuracy and efficiency in sentiment analysis.


