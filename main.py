import os
import glob
import sys

DB_path = os.path.join('/hdd', 'stonehye', 'VCDB', 'core_dataset', 'videos', '*')
dir_list = glob.glob(DB_path)

for dir in dir_list:
	category = dir.replace('videos', 'audios')
	if not (os.path.isdir(category)):
		os.makedirs(category)

	file_list = glob.glob(dir+'/*')
	file_list_video = [file for file in file_list if file.endswith(".mp4") or file.endswith(".flv")]

	for video in file_list_video:
		audio = video.replace('videos', 'audios')
		audio = audio.split('.')[0] + '.wav'
		os.system('ffmpeg -i '+video+' -f wav -vn '+audio)





