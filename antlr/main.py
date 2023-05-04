from antlr4 import *

from antlr.interpreter import MatrixInterpreter

import numpy as np


if __name__ == "__main__":
    prog =  InputStream(input(">>> "))
    interpreter = MatrixInterpreter(np.zeros((3, 3)))
    print(interpreter(prog))
