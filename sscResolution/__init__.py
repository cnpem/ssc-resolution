try:
    import pkg_resources 
    __version__ = pkg_resources.require("sscResolution")[0].version
except:
    pass


from .resolutiontypes import *
from .frc import *
from .fsc import *
from .statistics import *
from .misc import *

if __name__ == "__main__":
   pass


