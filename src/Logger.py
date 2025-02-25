class Logger:
    _instance = None
    
    @staticmethod
    def getInstance():
        if Logger._instance == None:
            Logger._instance = Logger()
        return Logger._instance

    def __init__(self, debug=False):
        if Logger._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Logger._instance = self
        self.debug = debug
        
    def logForward(self,Name, A, nbOfNeurons, weights, Z, Y):
        if not self.debug:
            return
        print("=" * 60)
        print(f"🟢 {Name} of {nbOfNeurons} Neurons")
        print("=" * 60)

        print(f"🔹 A: \n{repr(A)}\n")
        print(f"🔹 Weight : \n{repr(weights)}\n")
        print(f"🔹 Z: \n{repr(Z)}\n")
        print(f"🔹 Y: \n{repr(Y)}\n")
    
    def logBackward(self, Name, dE_dz, dE_dw, old_weights, new_weights):
        if not self.debug:
            return
        print("\n" + "=" * 60)
        print(f"🔴 Backward {Name}")
        print("=" * 60)
        
        print(f"🔹 Gradient dE_dz: ou yR -yP \n{repr(dE_dz)}\n")
        print(f"🔹 Gradient dE_dw: ajustement theorique \n{repr(dE_dw)}\n")
        print(f"🔹 Old weights: \n{repr(old_weights)}\n")
        print(f"🔹 Updated weights: \n{repr(new_weights)}\n")

    def printShape(self, name, matrix):
        print(f"{name}.shape = {matrix.shape}")

