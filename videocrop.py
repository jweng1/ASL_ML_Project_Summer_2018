import os

#ffmpeg -i clip.mp4 -vf "crop=360:360" cropped_clip.mp4

path = './Naomi/'

for filename in os.listdir(path):
    print ('yielding')
    if (filename.endswith(".mov")):
        os.system("ffmpeg -i {} -vf crop=360:360 {}".format(os.path.join(path, filename), filename))
    else:
        continue
