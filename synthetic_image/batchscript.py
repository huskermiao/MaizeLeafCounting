import os
import subprocess as sp

meshdir = "/media/jschnable/Seagate Backup Plus Drive/e-onsoftware/Objects/"

myfiles = os.listdir(meshdir)

for f in myfiles:
	if not "3ds" in f: continue
	mynum = f.split('.')[0].replace('maize mesh ','')
	for s in range(1,21):
		fh = open("configfile",'w')
		fh.write(meshdir+f+"\n"+"mesh{0}_angle{1}.png".format(str(int(mynum)).zfill(4),s)+"\n"+"/media/jschnable/Seagate Backup Plus Drive/chamberangles/angle{0}.blend".format(s) + "\n")
		fh.close()
		proc = sp.Popen(['blender',"/media/jschnable/Seagate Backup Plus Drive/chamberangles/angle{0}.blend".format(s),'--background','--python','render_script_v3.py'])
		proc.wait()
