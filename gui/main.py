import sys
import os
import threading
os.system("color")
import inspect

#from qtpy.QtCore import QThread,Signal,Qt
from qtpy.QtWidgets import (QApplication,QMainWindow,QVBoxLayout,QWidget,
                            QAction,QListWidget,QFormLayout,QComboBox,QLineEdit,
                            QTextEdit,QPushButton,QFileDialog,QMenu)
from qtpy.QtGui import QIcon,QIntValidator,QDoubleValidator,QColor,QFont

from . import qrc_resources
from .holo_container import get_holo_container
from .networking.client import PyClient
from .strtypes import error, warning, info

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import holograms as hg
from slm import SLM

hologram_functions = {'grating':hg.gratings.grating,'vertical grating':hg.gratings.vert,'vertical grating (gradient)':hg.gratings.vert_gradient,'horizontal grating':hg.gratings.hori,'horizontal grating (gradient)':hg.gratings.hori_gradient,'lens':hg.lenses.lens,'lens (focal shift)':hg.lenses.focal_plane_shift,
                      'zernike polynomial':hg.zernike,'array':hg.arrays.aags}
aperture_functions = {'circular aperture':hg.apertures.circ,'vertical aperture':hg.apertures.vert,'horizontal aperture':hg.apertures.hori}
cam_functions = {'LG superposition':hg.complex_amp_mod.superposition}

red = QColor(255,0,0)

def get_holo_type_function(name):
    if name in list(aperture_functions.keys()):
        holo_type = 'aperture'
        function = aperture_functions[name]
    elif name in list(cam_functions.keys()):
        holo_type = 'cam'
        function = cam_functions[name]
    elif name in list(hologram_functions.keys()):
        holo_type = 'holo'
        function = hologram_functions[name]
    else:
        raise NameError('Hologram \'{}\' not recognised'.format(name))
    return holo_type,function

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self,dev_mode=False):
        super().__init__()
        if dev_mode:
            self.tcp_client = PyClient(host='localhost',port=8627,name='SLM')
        else:
            self.tcp_client = PyClient(host='129.234.190.164',port=8627,name='SLM')
        self.tcp_client.start()

        self.setWindowTitle("SLM control")
        layout = QVBoxLayout()

        self.holoList = QListWidget()
        self.setCentralWidget(self.holoList)

        self._createActions()
        self._createMenuBar()
        self._createToolBars()
        # self._createContextMenu()
        self._connectActions()

        self.holos = []
        self.slm_settings = {'orientation':'horizontal',
                             'x size':512,
                             'y size':512,
                             'pixel size (m)':15e-6,
                             'monitor':1,
                             'beam x0':265,
                             'beam y0':251,
                             'beam waist (pixels)':215,
                             'wavelength':1064e-9}

        self.load_holo_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),"default_gui_state.txt"))
        if dev_mode:
            self.update_slm_settings({'monitor':0})

        self.slm = SLM(monitor=self.slm_settings['monitor'],gui=self)

        self.update_global_holo_params()
        self.update_holo_list()

        self.load_holo_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),"default_gui_state.txt"))

    def _createActions(self):
        self.addHoloAction = QAction(self)
        self.addHoloAction.setText("Add")
        self.addHoloAction.setIcon(QIcon(":add.svg"))

        self.removeHoloAction = QAction(self)
        self.removeHoloAction.setText("Remove")
        self.removeHoloAction.setIcon(QIcon(":subtract.svg"))

        self.editHoloAction = QAction(self)
        self.editHoloAction.setText("Edit")
        self.editHoloAction.setIcon(QIcon(":edit.svg"))

        self.upHoloAction = QAction(self)
        self.upHoloAction.setText("Up")
        self.upHoloAction.setIcon(QIcon(":up.svg"))

        self.downHoloAction = QAction(self)
        self.downHoloAction.setText("Down")
        self.downHoloAction.setIcon(QIcon(":down.svg"))

        self.loadHoloFileAction = QAction(self)
        self.loadHoloFileAction.setText("Load state")

        self.saveHoloFileAction = QAction(self)
        self.saveHoloFileAction.setText("Save state")

        self.slmSettingsAction = QAction(self)
        self.slmSettingsAction.setText("SLM settings")

    def _createMenuBar(self):
        menuBar = self.menuBar()
        mainMenu = menuBar.addMenu("Menu")
        mainMenu.addAction(self.loadHoloFileAction)
        mainMenu.addAction(self.saveHoloFileAction)
        mainMenu.addAction(self.slmSettingsAction)
    
    def _createToolBars(self):
        holoToolBar = self.addToolBar("Hologram creation")
        holoToolBar.addAction(self.addHoloAction)
        holoToolBar.addAction(self.removeHoloAction)
        holoToolBar.addAction(self.editHoloAction)

        orderToolBar = self.addToolBar("Hologram ordering")
        orderToolBar.addAction(self.upHoloAction)
        orderToolBar.addAction(self.downHoloAction)
    
    def _createContextMenu(self):
        contextMenu = QMenu(self)
        contextMenu.addAction(self.addHoloAction)
        contextMenu.addAction(self.removeHoloAction)
        contextMenu.addAction(self.editHoloAction)
        contextMenu.addAction(self.upHoloAction)
        contextMenu.addAction(self.downHoloAction)

    def _connectActions(self):
        self.addHoloAction.triggered.connect(self.open_new_holo_window)
        self.removeHoloAction.triggered.connect(self.remove_holo)
        self.editHoloAction.triggered.connect(self.edit_holo)
        self.upHoloAction.triggered.connect(self.up_holo)
        self.downHoloAction.triggered.connect(self.down_holo)
        self.loadHoloFileAction.triggered.connect(self.load_holo_file_dialogue)
        self.saveHoloFileAction.triggered.connect(self.save_holo_file_dialogue)
        self.slmSettingsAction.triggered.connect(self.open_slm_settings_window)
        self.holoList.itemDoubleClicked.connect(self.edit_holo)
        self.tcp_client.textin.connect(self.recieved_tcp_msg)

    def open_new_holo_window(self):
        self.w = HoloCreationWindow(self)
        self.w.show()

    def up_holo(self):
        currentRow = self.holoList.currentRow()
        if currentRow != 0:
            self.holos[currentRow],self.holos[currentRow-1] = self.holos[currentRow-1],self.holos[currentRow]
            self.update_holo_list()
            self.holoList.setCurrentRow(currentRow-1)

    def down_holo(self):
        currentRow = self.holoList.currentRow()
        if currentRow != self.holoList.count()-1:
            self.holos[currentRow],self.holos[currentRow+1] = self.holos[currentRow+1],self.holos[currentRow]
            self.update_holo_list()
            self.holoList.setCurrentRow(currentRow+1)

    def open_slm_settings_window(self):
        self.slm_settings_window = SLMSettingsWindow(self,self.slm_settings)
        self.slm_settings_window.show()

    def get_slm_settings(self):
        return self.slm_settings
    
    def update_slm_settings(self,slm_settings):
        old_slm_settings = self.slm_settings
        new_slm_settings = {**self.slm_settings,**slm_settings}
        for setting in slm_settings.keys():
            old_value = old_slm_settings[setting]
            new_value = new_slm_settings[setting]
            if new_value != old_value:
                warning('Changed global SLM setting {} from {} to {}'.format(setting,old_value,new_value))
                if setting == 'monitor':
                    try:
                        self.slm
                    except AttributeError:
                        pass
                    else:
                        error('SLM monitor cannot be updated once the display has been created.\n\t'
                              'Change the monitor in gui/default_gui_state.txt and restart the program.\n\t'
                              'Resetting monitor back to {}'.format(old_value))
                        new_slm_settings[setting] = old_value
            if setting == 'orientation':
                if new_value == 'horizontal':
                    aperture_functions['horizontal aperture'] = hg.apertures.vert
                    aperture_functions['vertical aperture'] = hg.apertures.hori
                else:
                    aperture_functions['horizontal aperture'] = hg.apertures.hori
                    aperture_functions['vertical aperture'] = hg.apertures.vert
        self.slm_settings = new_slm_settings
        self.update_global_holo_params()
        self.slm_settings_window = None

    def update_global_holo_params(self):
        shape = (self.slm_settings['x size'],self.slm_settings['y size'])
        beam_center = (self.slm_settings['beam x0'],self.slm_settings['beam y0'])
        pixel_size = self.slm_settings['pixel size (m)']
        self.global_holo_params = {'shape':shape,
                                   'beam_center':beam_center,
                                   'beam_waist':self.slm_settings['beam waist (pixels)'],
                                   'pixel_size':pixel_size,
                                   'wavelength':self.slm_settings['wavelength']}
    
    def get_global_holo_params(self):
        return self.global_holo_params

    def add_holo(self,holo_params):
        holo_params = {**holo_params,**self.global_holo_params}
        try:
            holo = get_holo_container(holo_params,self.global_holo_params)
            self.holos.append(holo)
            self.holoList.addItem(holo.get_label())
            self.w = None
            self.update_holo_list()
        except ValueError as e:
            error('Error when generating {} hologram:'.format(holo_params['name']),e)

    def edit_holo(self):
        currentRow = self.holoList.currentRow()
        self.w = HoloCreationWindow(self,currentRow)
        self.w.show()

    def remove_holo(self):
        currentRow = self.holoList.currentRow()
        del self.holos[currentRow]
        self.update_holo_list()
    
    def update_holo_list(self):
        currentRow = self.holoList.currentRow()
        labels = []
        types = []
        for i,holo in enumerate(self.holos):
            labels.append(str(i)+': '+holo.get_label())
            types.append(holo.get_type())
        for i in range(self.holoList.count()):
            self.holoList.takeItem(0)
        self.holoList.addItems(labels)
        # for i in range(self.holoList.count()):
        #     self.holoList.item(i).setForeground(red)
        try:
            last_holo = types[::-1].index('holo')
            last_aperture = types[::-1].index('aperture')
            if last_aperture > last_holo:
                warning('A hologram is applied after the final aperture')
        except ValueError:
            pass
        if currentRow >= self.holoList.count():
            currentRow = self.holoList.count()-1
        self.holoList.setCurrentRow(currentRow)
        self.calculate_total_holo()
        # print(self.holos)

    def calculate_total_holo(self):
        self.total_holo = hg.blank(phase=0,shape=self.global_holo_params['shape'])
        for holo in self.holos:
            if holo.get_type() == 'aperture':
                self.total_holo = holo.apply_aperture(self.total_holo)
            elif holo.get_type() == 'cam':
                self.total_holo = holo.get_cam_holo(self.total_holo)
            else:
                self.total_holo += holo.get_holo()
        # print(self.total_holo)
        self.slm.apply_hologram(self.total_holo)

    def set_holos_from_list(self,holo_list):
        """
        Set holograms from a list.

        Parameters
        ----------
        holos : list
            Should be a list of sublists containing the holo name and a dict 
            containing the holo arguments in the form [[holo1_name,{holo1_args}],...]
        """
        self.holos = []
        for name,args in holo_list:
            holo_params = {'name':name}
            holo_params['type'],holo_params['function'] = get_holo_type_function(name)
            holo_params = {**holo_params,**args}
            holo = get_holo_container(holo_params,self.global_holo_params)
            self.holos.append(holo)
        self.update_holo_list()
        self.w = None

    def generate_holo_list(self):
        holo_list = []
        for holo in self.holos:
            name = holo.get_name()
            args = holo.get_local_args()
            holo_list.append([name,args])
        return holo_list

    def save_holo_file(self,filename):
        holo_list = self.generate_holo_list()
        msg = [self.slm_settings,holo_list]
        with open(filename, 'w') as f:
            f.write(str(msg))
        info('SLM settings and holograms saved to "{}"'.format(filename))
    
    def save_holo_file_dialogue(self):
        filename = QFileDialog.getSaveFileName(self, 'Save file','.',"Text documents (*.txt)")[0]
        if filename != '':
            self.save_holo_file(filename)

    def recieved_tcp_msg(self,msg):
        info('TCP message recieved: "'+msg+'"')
        split_msg = msg.split('=')
        command = split_msg[0]
        arg = split_msg[1]
        if command == 'save':
            pass
        elif command == 'save_all':
            self.save_holo_file(arg)
        elif command == 'load_all':
            self.load_holo_file(arg)
        elif command == 'set_data':
            for update in eval(arg):
                try:
                    ind,arg_name,arg_val = update
                    info('Updating Hologram {} with {} = {}'.format(ind,arg_name,arg_val))
                    holo = self.holos[ind]
                    holo.update_arg(arg_name,arg_val)
                    self.holos[ind] = holo
                except NameError as e: 
                    error('{} is an invalid argument for Hologram {}\n'.format(arg_name,ind))
                except IndexError as e: 
                    error('Hologram {} does not exist\n'.format(ind))
        self.update_holo_list()
    
    def load_holo_file_dialogue(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file','.',"Text documents (*.txt)")[0]
        if filename != '':
            self.load_holo_file(filename)

    def load_holo_file(self,filename):
        try:
            with open(filename, 'r') as f:
                msg = f.read()
        except FileNotFoundError:
            error('"{}" does not exist'.format(filename))
            return
        try:
            msg = eval(msg)
            slm_settings = msg[0]
            holo_list = msg[1]
            self.update_slm_settings(slm_settings)
            try:
                self.slm
            except AttributeError:
                # info('SLM settings loaded from "{}"'.format(filename))
                pass
            else:
                self.set_holos_from_list(holo_list)
                info('SLM settings and holograms loaded from "{}"'.format(filename))
        except (SyntaxError, IndexError) as e:
            error('Failed to evaluate file "{}". Is the format correct?'.format(filename),e)

class SLMSettingsWindow(QWidget):
    def __init__(self,mainWindow,slm_settings):
        super().__init__()
        self.mainWindow = mainWindow
        self.slm_settings = slm_settings.copy()
        self.setWindowTitle("SLM settings")

        layout = QVBoxLayout()

        self.slmParamsLayout = QFormLayout()
        for key in list(self.slm_settings.keys()):
            if key == 'orientation':
                widget = QComboBox()
                widget.addItems(['horizontal','vertical'])
                widget.setCurrentText(self.slm_settings[key])
            else:
                widget = QLineEdit()
                widget.setText(str(self.slm_settings[key]))
                if key == (key == 'pixel size (m)') or (key == 'wavelength'):
                    widget.setValidator(QDoubleValidator())
                if key == 'monitor':
                    widget.setReadOnly(True)
                else:
                    widget.setValidator(QIntValidator())
            self.slmParamsLayout.addRow(key, widget)
        layout.addLayout(self.slmParamsLayout)

        self.saveButton = QPushButton("Save")
        layout.addWidget(self.saveButton)

        self.setLayout(layout)

        self._createActions()
        self._connectActions()

    def _createActions(self):
        self.saveAction = QAction(self)
        self.saveAction.setText("Save")

    def _connectActions(self):
        self.saveButton.clicked.connect(self.saveAction.trigger)
        self.saveAction.triggered.connect(self.update_slm_settings)
    
    def update_slm_settings(self):
        for row in range(self.slmParamsLayout.rowCount()):
            key = self.slmParamsLayout.itemAt(row,0).widget().text()
            widget = self.slmParamsLayout.itemAt(row,1).widget()
            if key == 'orientation':
                value = widget.currentText()
            elif (key == 'pixel size (m)') or (key == 'wavelength'):
                value = float(widget.text())
            else:
                value = int(widget.text())
            self.slm_settings[key] = value
        self.mainWindow.update_slm_settings(self.slm_settings)

    def get_slm_settings(self):
        return self.slm_settings

class HoloCreationWindow(QWidget):
    def __init__(self,mainWindow,edit_holo=None):
        super().__init__()
        self.mainWindow = mainWindow
        if edit_holo is None:
            self.setWindowTitle("New Hologram")
            self.editing = False
        else:
            self.setWindowTitle("Edit Hologram {}".format(edit_holo))
            self.editing = True
            self.edit_holo = edit_holo
            self.current_holo_list = self.mainWindow.generate_holo_list()
            self.current_name = self.current_holo_list[edit_holo][0]
            self.current_params = self.current_holo_list[edit_holo][1]
        layout = QVBoxLayout()
        
        self.holoSelector = QComboBox()
        self.holoSelector.addItems(list(hologram_functions.keys()))
        self.holoSelector.addItems(list(aperture_functions.keys()))
        self.holoSelector.addItems(list(cam_functions.keys()))
        layout.addWidget(self.holoSelector)

        if self.editing == True:
            self.holoSelector.setCurrentText(self.current_name)

        self.holoParamsLayout = QFormLayout()
        layout.addLayout(self.holoParamsLayout)

        if self.editing == False:
            self.holoAddButton = QPushButton("Add")
        else:
            self.holoAddButton = QPushButton("Edit")
        layout.addWidget(self.holoAddButton)

        self.holoDocBox = QTextEdit()
        self.holoDocBox.setReadOnly(True)
        self.holoDocBox.setLineWrapMode(False)
        # self.holoDocBox.setCurrentFont(QFont("Courier",4))
        layout.addWidget(self.holoDocBox)
        self.setLayout(layout)

        self._connectActions()
        self.update_holo_arguments()

    def _connectActions(self):
        self.holoAddButton.clicked.connect(self.return_holo_params)
        self.holoSelector.currentTextChanged.connect(self.update_holo_arguments)

    def update_holo_arguments(self):
        self.clear_holo_params()
        slm_settings = self.mainWindow.get_slm_settings()
        current = self.holoSelector.currentText()
        self.type,self.function = get_holo_type_function(current)
        arguments,_,_,defaults = inspect.getfullargspec(self.function)[:4]
        if len(arguments) != len(defaults):
            pad = ['']*(len(arguments)-len(defaults))
            defaults = pad + list(defaults)
        global_holo_params = self.mainWindow.get_global_holo_params()
        slm_settings = self.mainWindow.get_slm_settings()
        for argument,default in zip(arguments,defaults):
            if default != '':
                if argument not in global_holo_params.keys():
                    self.holoParamsLayout.addRow(argument, QLineEdit())
                    text_box = self.holoParamsLayout.itemAt(self.holoParamsLayout.rowCount()-1, 1).widget()
                    text_box.returnPressed.connect(self.return_holo_params)
                    if (self.editing == True) and (current == self.current_name):
                        text_box.setText(str(self.current_params[argument]))
                    else:
                        if argument == 'x0':
                            text_box.setText(str(slm_settings['beam x0']))
                        elif argument == 'y0':
                            text_box.setText(str(slm_settings['beam y0']))
                        elif argument == 'radius':
                            radius = min([slm_settings['x size']-slm_settings['beam x0'],
                                        slm_settings['y size']-slm_settings['beam y0'],
                                        slm_settings['beam x0'],slm_settings['beam y0']])
                            text_box.setText(str(radius))
                        else:
                            text_box.setText(str(default))
        self.holoDocBox.setText(self.function.__doc__.split('Returns')[0])

    def clear_holo_params(self):
        for i in range(self.holoParamsLayout.rowCount()):
            # print(i)
            self.holoParamsLayout.removeRow(0)
    
    def return_holo_params(self):
        holo_params = {'name':self.holoSelector.currentText()}
        holo_params['function'] = self.function
        holo_params['type'] = self.type
        if self.editing == True:
            holo_params = {}
        for row in range(self.holoParamsLayout.rowCount()):
            key = self.holoParamsLayout.itemAt(row,0).widget().text()
            widget = self.holoParamsLayout.itemAt(row,1).widget()
            value = widget.text()
            if (value.lower() == 'none') or (value == ''):
                value = None
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            holo_params[key] = value
        if self.editing == True:
            holo_list = self.current_holo_list.copy()
            holo_list[self.edit_holo] = [self.holoSelector.currentText(),holo_params]
            # print(holo_list)
            self.mainWindow.set_holos_from_list(holo_list)
        else:
            self.mainWindow.add_holo(holo_params)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()