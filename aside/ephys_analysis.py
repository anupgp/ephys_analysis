import numpy as np
from neo.io import AxonIO
import os
import logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

def get_fileformat(fb):
    """
    This function returns 1 or 2 if the ABF file is v1 or v2.    
    This function returns False if the file is not an ABF file.
    The first few characters of an ABF file tell you its format.
    Storage of this variable is superior to reading the ABF header because
    the file format is required before a version can even be extracted.
    """
    fb.seek(0)
    code = fb.read(4)
    code = code.decode("ascii", errors='ignore')
    if code == "ABF ":
        return 1
    elif code == "ABF2":
        return 2
    else:
        return False
    
def get_protocol_name(fname):
    """Determine the protocol used to record an ABF file"""
    f=open(fname,'rb')
    print(get_fileformat(f))
    
    raw=f.read(30*1000) #it should be in the first 30k of the file
    f.close()
    raw=raw.decode("utf-8","ignore")
    raw=raw.split("Clampex")[1].split(".pro")[0]+".pro"
    return raw


