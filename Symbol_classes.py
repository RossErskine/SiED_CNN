from collections import namedtuple

Point = namedtuple('x', 'y')

class Symbol_info:
    """ to store detected symbol information """
    
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.text =""
        self.box = ((x,y),( x+100, y +100))
    
    
    def setText(self, text):
        self.text = text
        
class Lines:
    """ to store detected line information 
        sx = start x coordinate
        sy = start y coordinate 
        ex = end x coordinate
        ey = end y coordinate
    """
    label_names = []  
    
    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        
     
    
    