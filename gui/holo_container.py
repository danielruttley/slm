import sys
import inspect
from numpy import array_equal

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import holograms as hg
import time

def get_holo_container(holo_params,global_holo_params):
        if holo_params['type'] == 'aperture':
            holo = ApertureContainer(holo_params,global_holo_params)
        elif holo_params['type'] == 'cam':
            holo = CAMContainer(holo_params,global_holo_params)
        elif holo_params['type'] == 'holo':
            holo = HoloContainer(holo_params,global_holo_params)
        else:
            raise NameError('Hologram type \'{}\' not recognised'.format(holo_params['type']))
        return holo

class Container():
    def get_label(self):
        label = self.name+' ('
        for key in self.local_args.keys():
            label += key+'='+str(self.args[key])+', '
        label = label[:-2]
        label += ')'
        return label

    def get_type(self):
        return self.type
    
    def get_name(self):
        return self.name
    
    def get_arg_names(self):
        return inspect.getargspec(self.function)[0]

    def get_args(self):
        return self.args
    
    def get_local_args(self):
        return self.local_args

class HoloContainer(Container):
    def __init__(self,holo_params,global_holo_params):
        self.name = holo_params['name']
        self.function = holo_params['function']
        self.type='holo'
        self.args = {k: holo_params[k] for k in inspect.getfullargspec(self.function)[0] if k in holo_params}
        self.local_args = self.args.copy()
        for key in global_holo_params:
            self.local_args.pop(key, None)

        self.calculate_holo()
    
    def calculate_holo(self):
        self.holo = self.function(**self.args)

    def update_arg(self,arg_name,arg_value):
        if arg_name in inspect.getfullargspec(self.function)[0]:
            self.args[arg_name] = arg_value
            self.calculate_holo()
        else:
            raise NameError('{} is not a parameter for {} hologram'.format(arg_name,self.name))

    def get_holo(self):
        return self.holo

class ApertureContainer(Container):
    def __init__(self,holo_params,global_holo_params):
        self.name = holo_params['name']
        self.function = holo_params['function']
        self.type='aperture'
        self.args = {k: holo_params[k] for k in inspect.getfullargspec(self.function)[0] if k in holo_params}
        self.local_args = self.args.copy()
        for key in global_holo_params:
            self.local_args.pop(key, None)
    
    def apply_aperture(self,holo):
        holo = self.function(holo,**self.args)
        return holo

    def update_arg(self,arg_name,arg_value):
        if arg_name in inspect.getfullargspec(self.function)[0]:
            self.args[arg_name] = arg_value
        else:
            raise NameError('{} is not a parameter for {}'.format(arg_name,self.name))

class CAMContainer(Container):
    def __init__(self,holo_params,global_holo_params):
        self.name = holo_params['name']
        self.function = holo_params['function']
        self.type='cam'
        self.args = {k: holo_params[k] for k in inspect.getfullargspec(self.function)[0] if k in holo_params}
        self.local_args = self.args.copy()
        self.prev_holos = None
        self.holo = None
        for key in global_holo_params:
            self.local_args.pop(key, None)
    
    def get_cam_holo(self,prev_holos):
        if not array_equal(self.prev_holos,prev_holos):
            self.holo = self.function(prev_holos,**self.args)
            self.prev_holos = prev_holos.copy()
        return self.holo

    def update_arg(self,arg_name,arg_value):
        if arg_name in inspect.getfullargspec(self.function)[0]:
            self.args[arg_name] = arg_value
        else:
            raise NameError('{} is not a parameter for {}'.format(arg_name,self.name))