import os
import glob
import csv

import librosa
import matplotlib.pyplot as plt

from dtw import dtw
from numpy.linalg import norm

category = 'bill_clinton_apology_speech'
annotation = open(category+'.csv', 'w', encoding='utf-8')
wr = csv.writer(annotation)
wr.writerow(['Name', 'audio 1', 'audio 2', 'dist'])

path = os.path.join('/hdd/stonehye/VCDB/core_dataset/audios/', category, '*')
audio_list = glob.glob(path)
audio_list2 = audio_list.copy()

idx = 1
for idx1, aud1 in enumerate(audio_list):
	y1, sr1 = librosa.load(aud1)
	mfcc1 = librosa.feature.mfcc(y1, sr1)
	for idx2, aud2 in enumerate(audio_list2):
		if (idx1<=idx2):
			name = 'result'+str(idx)

			y2, sr2 = librosa.load(aud2)
			mfcc2 = librosa.feature.mfcc(y2, sr2)

			dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
			wr.writerow([name, aud1.split('.')[0], aud2.split('.')[0], dist])
			print([name, aud1.split('.')[0], aud2.split('.')[0], dist])
			# print ('Normalized distance between the two sounds:', dist)
			plt.imshow(cost.T, origin='lower', cmap='gray', interpolation='nearest')
			plt.plot(path[0],path[1], 'w')
			plt.xlim((-0.5, cost.shape[0]-0.5))
			plt.ylim((-0.5, cost.shape[1]-0.5))

			plt.savefig(name+'.png')
			plt.clf()

			idx += 1

annotation.close()