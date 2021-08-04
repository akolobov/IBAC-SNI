import numpy as np
import pickle
from sklearn.manifold import TSNE
import glob
import matplotlib.pyplot as plt

pkls = glob.glob('../../*.pkl')
sorted(pkls)

for i,pkl in enumerate(pkls):
    data = pickle.load(open(pkl,"rb"))
    step_id = pkl.split('_')[-1].split('.')[0]
    tsne = TSNE(2)
    dots = tsne.fit_transform(data[1])
    plt.scatter(dots[64:,0],dots[64:,1],label=str(int((i+1)*1e5)))

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('%s_tsne..png'%step_id)
# plt.clf()
