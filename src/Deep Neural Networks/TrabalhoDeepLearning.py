import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import math, time
import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

# Etapa 1 - preparar o dataset


def get_stock_data(normalized=0, file_name=None):
    col_names = ['checkEnvironment','checkPhysical','ppd','checkEmotion','room','hour','minute','humidity','luminosity','temperature','pmv','emotion']
    stocks = pd.read_csv(file_name, header=0, names=col_names, delimiter = ';')  # fica numa especie de tabela exactamente como estava no csv )
    df = pd.DataFrame(stocks)
    return df


def load_stock_dataset():
    return get_stock_data(0, 'BuildingTestNumericNormalize.csv')


# Visualizar os top registos da tabela
def visualize():
    df = load_stock_dataset()
    print('### Antes do pré-processamento ###')
    print(df.head())  # mostra só os primeiros 5 registos


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


#utils para visulaização do historial de aprendizagem
def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def build_model2():
    model = Sequential()
    model.add(Dense(64, input_dim=11, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def print_model(model, fich):
    from keras.utils import plot_model
    plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)


def model_evaluate(model, input_attributes, output_attributes):
    print("###########inicio do evaluate###############################\n")
    scores = model.evaluate(input_attributes, output_attributes)
    print("\n metrica: %s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))


def print_series_prediction(y_test, predic):
    diff = []
    racio = []
    for i in range(len(y_test)):  # para imprimir tabela de previsoes
        racio.append((y_test[i] / predic[i]) - 1)
        diff.append(abs(y_test[i] - predic[i]))
        print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i], predic[i], diff[i], racio[i]))
    plt.plot(y_test, color='blue', label='y_test')
    plt.plot(predic, color='red', label='prediction')  # este deu uma linha em branco
    plt.plot(diff, color='green', label='diff')
    plt.plot(racio, color='yellow', label='racio')
    plt.legend(loc='upper left')
    plt.show()


def LSTM_utilizando_data():
    df = load_stock_dataset()
    print("df", df.shape)
    x = df.iloc[:, 0:11]
    y = df.iloc[:, 11]
    (X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print('y_test', y_test.shape)
    model = build_model2()
    time_callback = TimeHistory()
    history = model.fit(X_train, y_train,callbacks=[time_callback], batch_size=256, epochs=3000, verbose=1)
    times = time_callback.times
    print_history_loss(history)
    print_model(model, "lstm_model.png")
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    print(model.metrics_names)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    p = model.predict(X_test)
    predic = np.squeeze(np.asarray(p))  # para transformar uma matriz de uma coluna e n linhas em um np array de n elementos
    predicRounded = np.rint(predic) # para arrendondar para o inteiro mais proximo!
    print_series_prediction(y_test, predicRounded)
    print('Total Time of Execution: %.2f seconds' % (sum(times)))


if __name__ == '__main__':
    # visualize_GOOGL()
    LSTM_utilizando_data()
