class Logger:
    def __init__(self, debug=True):
        self.debug = debug
        
    def log_forward(self, layer_name, weights, inputs, Z, Y):
        if not self.debug:
            return