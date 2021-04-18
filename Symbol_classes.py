
class Symbol_info:
    """ to store detected symbol information """
    
    def __init__(self, x, y, label, score, scale, power):
        x = int(x * (scale ** power)) # moves position depending scale for final image
        y = int(y * (scale ** power))
        self.box = (x ,y,(x+ int(100 / (scale ** power))),(y+ int(100 / (scale ** power))))
        self.label = label
        self.score = score
        self.text =""
        
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
        self.line = ((sx, sy),(ex,ey))
        
    def addText(self, label):
         self.label_names.append(label)
    
    