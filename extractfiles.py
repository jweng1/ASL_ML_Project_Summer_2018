
import csv
import glob
import os
from subprocess import call
import os



path = './asl_data'
data_file = ['train, test']

#ffmpeg -i input.mp4 -r 1 -f image2 image-%2d.png

for filename in os.listdir(path):
    print ('yielding')
    if (filename.endswith(".mp4")):
        os.system("ffmpeg -i {} -r 40 -f image2 {}_%3d.jpg -hide_banner".format(os.path.join(path, filename), filename))
    else:
        continue
