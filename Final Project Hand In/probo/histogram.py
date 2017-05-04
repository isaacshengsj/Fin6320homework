import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from probo import test_naivemc

# Randomly generate some normal data
#z = np.random.normal(size = 100000)



# the histogram of the data
plt.hist(z, 50, normed=1, facecolor = 'green')
#, alpha=0.75)

# save the figure to file
plt.savefig('jike2.png')

