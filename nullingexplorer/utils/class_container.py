import logging
import torch.nn as nn

class ModuleRegister(nn.Module):
    def __init_subclass__(cls, cls_type = None, cls_name = None):
        ClassContainer.add_class(cls, cls_type, cls_name)

    def config(self, config):
        pass


class ClassContainer(object):
    '''
    Singleton, contain all the user defined classes, like instruments, spectra and targets.
    '''

    _instance = None
    _class_list = \
        {   
            "Amplitude"         : {},
            "Instrument"        : {}, 
            "Spectrum"          : {},
            "Transmission"      : {},
            "Electronics"       : {}
        }
    def __new__(cls, *args, **kwds):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwds)
        return cls._instance

    def __init__(self):
        pass

    @classmethod
    def get_types(cls):
        return cls._class_list.keys()

    @classmethod
    def get_class(cls, cls_type, cls_name):
        if cls_type not in cls._class_list.keys():
            raise ValueError(f"Module type {cls_type} not found!")

        if cls_name in cls._class_list[cls_type].keys():
            return cls._class_list[cls_type][cls_name]
        else:
            raise ValueError(f"Class {cls_name} not found!")

    @classmethod
    def add_class(cls, add_cls: object, cls_type, cls_name):
        if cls_type == None:
            raise TypeError("Please assign the module cls_type!")
        elif cls_type not in cls._class_list.keys():
            raise ValueError(f"Module type {cls_type} not found!")
    
        if cls_name == None:
            cls_name = add_cls.__name__
        if cls_name in cls._class_list[cls_type].keys():
            raise TypeError(f"Class {cls_name} already exist in {cls_type} class list!")

        if cls_name != 'base':
            if not issubclass(add_cls, cls._class_list[cls_type]['base']):
                raise TypeError(f"Class {cls_name} must be a subclasses inheriting from Base{cls_type}!")

        cls._class_list[cls_type][cls_name] = add_cls

        #print(f"Register {cls_type} class {cls_name}")        

    @classmethod
    def print_available_class(cls):
        print('All available modules:')
        for type_name, atype in cls._class_list.items():
            print(f"Module {type_name}:\t{list(atype.keys())}")

def register_class(cls_type = None, cls_name = None):
    def decorator(cls):
        ClassContainer.add_class(cls, cls_type, cls_name)
        return cls
    return decorator

def get_class(cls_type = None, cls_name = None):
    return ClassContainer.get_class(cls_type, cls_name)

def get_amplitude(cls_name = None):
    return ClassContainer.get_class('Amplitude', cls_name)

def get_instrument(cls_name = None):
    return ClassContainer.get_class('Instrument', cls_name)

def get_spectrum(cls_name = None):
    return ClassContainer.get_class('Spectrum', cls_name)

def get_transmission(cls_name = None):
    return ClassContainer.get_class('Transmission', cls_name)

def get_electronics(cls_name = None):
    return ClassContainer.get_class('Electronics', cls_name)

# test
def main():
    ClassContainer.print_available_class()
    overallinst = ClassContainer.get_class("instrument", "overall")()
    overallinst.test()

if __name__ == "__main__":
    main()