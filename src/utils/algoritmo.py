import os
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
import time

from interface.logger import Logger

class AmbienteAlgoritmo:
    def preparar_data(self):
        data = load_breast_cancer()
        self.x = data.data #type: ignore
        self.y = data.target #type: ignore
        self.feature_names = list(data.feature_names) #type: ignore

    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))

    def normalizar(self, X):
        # Normalización: Z = (X - media) / desviación
        media = np.mean(X, axis=0)
        desviacion = np.std(X, axis=0)
        return (X - media) / desviacion

    def dividir_datos(self, X, y, porcentaje_entrenamiento=0.8):
        # Mezclar aleatoriamente
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        corte = int(porcentaje_entrenamiento * X.shape[0])
        X_train, X_test = X[:corte], X[corte:]
        y_train, y_test = y[:corte], y[corte:]

        return X_train, X_test, y_train, y_test

    def entrenar(self, idx_x, idx_y, epochs, eta, porcentaje_entrenamiento, callback_actualizar_plot = None):
        # Seleccionar características
        X_selected = self.x[:, [idx_x, idx_y]]
        y = self.y

        # Normalizar
        X_selected = self.normalizar(X_selected)

        # Dividir 
        X_train, X_test, y_train, y_test = self.dividir_datos(X_selected, y, porcentaje_entrenamiento)

        # Se inicializan los pesos (2 características y 1 bias)
        self.pesos = np.random.rand(3)
        self.errors = []

        # Agregar columna de unos (bias)
        X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

        # Se recorren las epocas
        for epoch in range(epochs):
            total_error = 0
            for x_i, target in zip(X_train_bias, y_train):
                z = np.dot(x_i, self.pesos)
                output = self.sigmoide(z)
                error = target - output
                self.pesos += eta * error * x_i
                total_error += error ** 2

            self.errors.append(total_error)
            Logger.instance().log(f"Epoca {epoch+1}, Error: {total_error:.4f}")
            time.sleep(0.1)

            if callback_actualizar_plot:
                callback_actualizar_plot(self.pesos, X_selected, y, idx_x, idx_y)

        # Evaluar exactitud en pruebas
        X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        accuracy = self.accuracy(X_test_bias, y_test)
        Logger.instance().log(f"Exactitud final: {accuracy * 100:.2f}%")

        return {
            "pesos": self.pesos,
            "errores": self.errors,
            "precision": accuracy
        }

    def predecir(self, X_bias):
        z = np.dot(X_bias, self.pesos)
        return np.where(self.sigmoide(z) >= 0.5, 1, 0)

    def accuracy(self, X_bias, y):
        y_pred = self.predecir(X_bias)
        return np.mean(y_pred == y)
