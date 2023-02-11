import os.path
import glob
import sys


path = "/home/student/Downloads/FRC2023test3/labels"
#savepath = "/home/student/Downloads/FRC2023test3/test_new"
savepath = path+"_new"
filelist = (glob.glob(path+"/*.txt"))
for f in filelist:
	#assuming all the files have the same number of numbers
	#filedata = open(f,'r')
	file_new = open(savepath+"/n_"+os.path.basename(f),'w')
	i = True
	with open(f, 'r') as openfileobject:
		for line in openfileobject:
			linesplit = line.split()
			if int(line[0]) >= 2:
				linesplit[0]="2"
			newline = " ".join(linesplit)+"\n"
			file_new.write(newline)
		openfileobject.close()
	file_new.close()
	

		
		
	#print (f.open())



#dirPath = r"/home/student/Downloads/FRC2023test3/labels"
#result = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
#print(result)

#for labels in os.listdir ("labels"):
	#with open(os.path.join("labels", labels), 'r') as f
	#text = f.read()
	#print(text)
	
