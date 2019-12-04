fo = open("train.txt",'w')

img_start = 2500
img_end = 12500

for i in range(img_start, img_end):
	fo.write("dataset/train/cat." + str(i) + ".jpg" + " 0\n")
	fo.write("dataset/train/dog." + str(i) + ".jpg" + " 1\n")

fo.close()
