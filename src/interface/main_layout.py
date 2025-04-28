from PyQt5.QtWidgets import (
    QLayout, QMainWindow, QScrollArea, QWidget, QTabWidget, QLabel, QLineEdit, QPushButton,
    QTextEdit, QVBoxLayout, QGridLayout, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal
from utils.algoritmo import *
from interface.algoritmo_layout import ALayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aprendizaje No Supervisado - Redes Neuronales")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
      # Directamente usar el layout de GALayout sin tabs
        self.main_widget = ALayout()  # asumiendo que GALayout hereda de QWidget

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.main_widget)

        container = QWidget()
        container.setLayout(main_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)

        self.setCentralWidget(scroll_area)
