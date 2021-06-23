import sys

from qtpy.QtWidgets import QApplication
from gui import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()