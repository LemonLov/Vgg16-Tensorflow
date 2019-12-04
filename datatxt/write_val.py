fo = open("val.txt",'w')

img_num = 1250

for i in range(img_num):
	fo.write("dataset/val/cat." + str(i) + ".jpg" + " 0\n")
	fo.write("dataset/val/dog." + str(i) + ".jpg" + " 1\n")

fo.close()
