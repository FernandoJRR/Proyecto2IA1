from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget
import numpy as np

from interface.logger import Logger
from utils.algoritmo import AmbienteAlgoritmo
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ALayout(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ambiente = AmbienteAlgoritmo()
        self.ambiente.preparar_data()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        header_hlayout = QHBoxLayout()
        param_group = QGroupBox("Parámetros del Algoritmo")
        param_layout = QGridLayout()

        param_layout.addWidget(QLabel("Tasa de Aprendizaje"), 0, 0)
        self.tasa_aprendizaje_edit = QLineEdit("0.01")
        param_layout.addWidget(self.tasa_aprendizaje_edit, 0, 1)

        param_layout.addWidget(QLabel("Numero Maximo de Epocas"), 1, 0)
        self.num_max_epochs_edit = QLineEdit("50")
        param_layout.addWidget(self.num_max_epochs_edit, 1, 1)

        param_layout.addWidget(QLabel("Porcentaje de Datos a Entrenar"), 2, 0)
        self.porcentaje_entrenamiento_edit = QLineEdit("0.8")
        param_layout.addWidget(self.porcentaje_entrenamiento_edit, 2, 1)

        self.run_button = QPushButton("Entrenar Perceptron")
        self.run_button.clicked.connect(self.start_training)
        param_layout.addWidget(self.run_button, 6, 0, 1, 2)
        param_group.setLayout(param_layout)
        header_hlayout.addWidget(param_group)

        console_group = QGroupBox("Consola")
        console_layout = QVBoxLayout()
        self.console = Logger.instance()
        self.console.setSizePolicy(self.console.sizePolicy().Expanding, 
                                   self.console.sizePolicy().Expanding)
        self.console.textChanged.connect(
            lambda: self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum()) #type: ignore
        )
        self.console.moveCursor(QTextCursor.End)
        console_layout.addWidget(self.console)
        console_group.setLayout(console_layout)
        header_hlayout.addWidget(console_group)

        layout.addLayout(header_hlayout)

        self.visual_hlayout = QHBoxLayout()

        caract_group = QGroupBox("Características")
        caracteristica_layout = QGridLayout()

        caracteristica_layout.addWidget(QLabel("Primera Caracteristica (x):"), 0, 0)
        self.combo_x = QComboBox()
        self.combo_x.addItems(self.ambiente.feature_names)
        caracteristica_layout.addWidget(self.combo_x, 0, 1)

        caracteristica_layout.addWidget(QLabel("Segunda Caracteristica (y):"), 1, 0)
        self.combo_y = QComboBox()
        self.combo_y.addItems(self.ambiente.feature_names)
        caracteristica_layout.addWidget(self.combo_y, 1, 1)

        self.scatter_button = QPushButton("Mostrar ScatterPlot")
        self.scatter_button.clicked.connect(self.show_scatter)
        caracteristica_layout.addWidget(self.scatter_button, 2, 1)

        caract_group.setLayout(caracteristica_layout)
        self.visual_hlayout.addWidget(caract_group)

        self.scatter_group = QGroupBox("Scatter Plot Caracteristicas")
        self.scatter_group.setMinimumHeight(300)
        self.scatter_group.setMaximumHeight(300)
        self.scatter_layout = QVBoxLayout()
        self.scatter_canvas = None

        self.scatter_group.setLayout(self.scatter_layout)
        self.visual_hlayout.addWidget(self.scatter_group)

        layout.addLayout(self.visual_hlayout)

        self.frontera_group = QGroupBox("Frontera de Decision")
        self.frontera_group.setMinimumHeight(400)
        self.frontera_layout = QVBoxLayout()
        self.frontera_canvas = None

        self.frontera_group.setLayout(self.frontera_layout)
        layout.addWidget(self.frontera_group)

        self.errores_group = QGroupBox("Errores vs Epocas")
        self.errores_group.setMinimumHeight(400)
        self.errores_layout = QVBoxLayout()
        self.errores_group.setLayout(self.errores_layout)
        layout.addWidget(self.errores_group)

        self.setLayout(layout)

    def start_training(self):
        try:
            Logger.instance().clear()
            tasa_aprendizaje = float(self.tasa_aprendizaje_edit.text())
            num_max_epochs = int(self.num_max_epochs_edit.text())
            porcentaje_entrenamiento = float(self.porcentaje_entrenamiento_edit.text())

            nombre_caract_1 = self.combo_x.currentText()
            nombre_caract_2 = self.combo_y.currentText()
        except ValueError:
            QMessageBox.critical(self, "Error", "Ingrese valores numéricos válidos.")
            return

        self.run_button.setEnabled(False)
        self.worker = AWorker(self.ambiente, nombre_caract_1, nombre_caract_2, num_max_epochs, tasa_aprendizaje, porcentaje_entrenamiento)
        self.worker.result_signal.connect(self.display_result)
        self.worker.frontera_signal.connect(self.actualizar_frontera)
        self.worker.start()

    def show_scatter(self):
        feature_x = self.combo_x.currentText()
        feature_y = self.combo_y.currentText()

        try:
            idx_x = self.ambiente.feature_names.index(feature_x)
            idx_y = self.ambiente.feature_names.index(feature_y)
        except ValueError:
            QMessageBox.warning(self, "Error", "Error al buscar las caracteristicas seleccionadas")
            return

        x_data = self.ambiente.x[:, idx_x]
        y_data = self.ambiente.x[:, idx_y]
        labels = self.ambiente.y

        if self.scatter_canvas:
            self.scatter_layout.removeWidget(self.scatter_canvas)
            self.scatter_canvas.setParent(None)
            self.scatter_canvas.deleteLater()
            self.scatter_canvas = None

        # Create a new figure and canvas
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(x_data, y_data, c=labels, cmap='coolwarm', edgecolors='k')
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title("Scatter Plot")

        self.scatter_canvas = FigureCanvas(fig)
        self.scatter_layout.addWidget(self.scatter_canvas)

    def display_result(self, emit_input: dict):
        if hasattr(self, 'errores_canvas') and self.errores_canvas:
            self.errores_layout.removeWidget(self.errores_canvas)
            self.errores_canvas.setParent(None)
            self.errores_canvas.deleteLater()
            self.errores_canvas = None

        errors = emit_input.get('errores', [])

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-')
        ax.set_title("Errores vs Epocas")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Error")
        ax.grid(True)

        self.errores_canvas = FigureCanvas(fig)
        self.errores_layout.addWidget(self.errores_canvas)

        self.run_button.setEnabled(True)

    def actualizar_frontera(self, pesos, X_data, y_data):
        if self.frontera_canvas:
            self.frontera_layout.removeWidget(self.frontera_canvas)
            self.frontera_canvas.setParent(None)
            self.frontera_canvas.deleteLater()
            self.frontera_canvas = None

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        # fondo
        x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
        y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        mesh_points = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
        Z = self.ambiente.sigmoide(np.dot(mesh_points, pesos))
        Z = Z.reshape(xx.shape)

        # color de fondo
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['#AAAAFF', '#FFAAAA'])

        # datos
        ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='coolwarm', edgecolors='k')

        # linea de frontera
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(pesos[0] + pesos[1] * x_vals) / pesos[2]
        ax.plot(x_vals, y_vals, 'g--', linewidth=2, color='#000000')

        ax.set_xlabel(self.combo_x.currentText())
        ax.set_ylabel(self.combo_y.currentText())
        ax.set_title("Frontera de Decisión")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        self.frontera_canvas = FigureCanvas(fig)
        self.frontera_layout.addWidget(self.frontera_canvas)

# QThread para poder ejecutar el algoritmo dentro de la interfaz
class AWorker(QThread):
    # signal para enviar resultado del algoritmo
    result_signal = pyqtSignal(dict)
    # signal para enviar actualizacion de la frontera
    frontera_signal = pyqtSignal(object, object, object, int, int)

    def __init__(self, ambiente, primer_caract, segunda_caract, num_epocas, tasa_aprendizaje, porcentaje_entrenamiento, parent=None):
        self.ambiente: AmbienteAlgoritmo = ambiente
        self.primer_caract: str = primer_caract
        self.segunda_caract: str = segunda_caract
        self.num_epocas: int = num_epocas
        self.tasa_aprendizaje: float = tasa_aprendizaje
        self.porcentaje_entrenamiento: float = porcentaje_entrenamiento
        super().__init__(parent)

    def run(self):
        idx_x = self.ambiente.feature_names.index(self.primer_caract)
        idx_y = self.ambiente.feature_names.index(self.segunda_caract)

        def callback(weights, X_data, y_data, idx_x, idx_y):
            self.frontera_signal.emit(weights.copy(), X_data.copy(), y_data.copy(), idx_x, idx_y)


        result = self.ambiente.entrenar(idx_x, idx_y, self.num_epocas, self.tasa_aprendizaje, self.porcentaje_entrenamiento, callback)
        self.result_signal.emit(result)
