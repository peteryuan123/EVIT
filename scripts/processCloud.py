import numpy as np
import sys

path = sys.argv[1]
folder = path[:path.rfind("/")]

cloud = np.loadtxt(path)
cloud[:, 3:] = cloud[:, 3:] - cloud[:, :3]
for i in range(cloud.shape[0]):
    cloud[i, 3:] /= np.linalg.norm(cloud[i, 3:])
np.savetxt(folder + "/modified.txt", cloud)