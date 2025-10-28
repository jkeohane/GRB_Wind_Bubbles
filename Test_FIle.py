import sys, matplotlib
print("Python:", sys.executable)
print("matplotlib:", matplotlib.__version__)
print("backend:", matplotlib.get_backend())

import matplotlib.pyplot as plt
plt.plot([0,1,2],[0,1,0])
plt.title("Backend test")
plt.show()

