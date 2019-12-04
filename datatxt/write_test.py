fo = open("test.txt",'w')

img_start = 1250
img_end = 2500

for i in range(img_start, img_end):
	fo.write("dataset/test/cat." + str(i) + ".jpg" + " 0\n")
	fo.write("dataset/test/dog." + str(i) + ".jpg" + " 1\n")

fo.close()
