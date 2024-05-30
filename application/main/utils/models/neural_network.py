
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import matplotlib.pyplot as plt

# def train_and_evaluate_ann(df, epochs=20):
#     X = df.iloc[:, :-1].values  
#     y = df.iloc[:, -1].values   

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = Sequential()
#     model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, activation='linear')) 

#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#     history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=10, verbose=1)

#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()

#     loss, accuracy = model.evaluate(X_test, y_test)
#     print(f'Test Accuracy: {accuracy}')
