import sys

from qtpy.QtWidgets import QApplication
from gui import MainWindow

if 'dev' in sys.argv:
    dev_mode=True
else:
    dev_mode=False

app = QApplication(sys.argv)
window = MainWindow(dev_mode)
window.show()
app.exec()