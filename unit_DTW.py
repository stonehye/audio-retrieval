import os
import time

import librosa
import matplotlib.pyplot as plt

from dtw import dtw
from numpy.linalg import norm


total_start = time.time()
path1 = os.path.join('/hdd/stonehye/VCDB/core_dataset/audios/beautiful_mind_game_theory/46f2e964ae16f5c27fad70d6849c76616fad7502.wav')
path2 = os.path.join('/hdd/stonehye/VCDB/core_dataset/audios/baggio_penalty_1994/bb604f57a18455867544e79c2e32bf5583c358d4.wav')
y1, sr1 = librosa.load(path1)
y2, sr2 = librosa.load(path2)

feature1_start = time.time()
mfcc1 = librosa.feature.mfcc(y1, sr1)
print("feature1 time: ", time.time() - feature1_start)
feature2_start = time.time()
mfcc2 = librosa.feature.mfcc(y2, sr2)
print("feature2 time: ", time.time() - feature2_start)

dtw_start = time.time()
dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print("dtw time: ", time.time() - dtw_start)

print("total time: ", time.time() - total_start)

plt.imshow(cost.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))

plt.savefig('result.png')
