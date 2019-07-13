import bpy
import fnmatch
import sys

import numpy
import colorsys
#myfile = sys.argv[5]
#myout = sys.argv[6]
#print(sys.argv)

#This part reads in the the variables from a config file
fh = open("configfile")
#Path to Mesh
myfile = fh.readline().strip()
myout = fh.readline().strip()
myscene = fh.readline().strip()
#1/0
print(myfile)
print(myout)
#1/0


mypath = ""
#myscene = "chamber.blend"

def random_green():
	
	h = numpy.random.normal(loc=88,scale=8)
	if h < 60: h = 60
	if h > 120: h = 120
	s = numpy.random.normal(0.529,.09)
	if s < 0.15: s = 0.15
	if s > 0.98: s = 0.98

	v = numpy.random.normal(0.22,0.06)
	if v < 0.0235: v = 0.0235
	if v > 0.6: v = 0.6

	r,g,b = colorsys.hsv_to_rgb(h/360.0,s,v)
	return (r,g,b)
#print(h,s,v)
#print(g,r,b)

#1/0

#bpy.ops.wm.open_mainfile(filepath=mypath+myscene)

#use_apply_transform

bpy.ops.import_scene.autodesk_3ds(filepath = myfile)
x,y,z = bpy.data.objects['Realms_A'].dimensions
bpy.data.objects['Realms_A'].dimensions = (x*1.3, y*1.3, z*.8)
bpy.context.scene.objects.active = bpy.data.objects['Realms_A']
for o in bpy.context.object.material_slots:
	if "Leaf" in o.name:
		o.material.diffuse_color = random_green()
	if "Stalk" in o.name:
		o.material.diffuse_color = random_green()
	o.material.specular_intensity = 0.05
"""
	if 'Ear' in o.name or 'Silk' in o.name:
		print(o.name)
		o.material.alpha=0
		o.material.transparency_method = 'Z_TRANSPARENCY'
		o.material.use_transparency = True
#		bpy.context.object.show_transparent = True
		o.material.volume.density = 0
		o.material.translucency = 1
#    sys.stderr.write(o.name + "\n")
"""

object = bpy.data.objects['Realms_A']
object.select = True
bpy.ops.mesh.separate(type='MATERIAL')
bpy.ops.object.mode_set( mode = 'OBJECT' )
for o in bpy.context.selected_objects:
	if not 'Ear' in o.active_material.name and not 'Silk' in o.active_material.name:
		o.select = False
	else:
		o.select = True

bpy.ops.object.delete()

"""
scene = bpy.context.scene
for s in scene.objects:
	sys.stderr.write(s.name + "\n")
	sys.stderr.write(",".join(list(s.children)))

foo_objs = [obj for obj in scene.objects if fnmatch.fnmatchcase(obj.name, "leaf*")]
sys.stderr.write("Number of leafs:" + str(len(foo_objs)))
1/0


for l in foo_objs:
	l.active_material.diffuse_color = (r,g,b)
"""
bpy.data.scenes['Scene'].render.filepath = "/media/jschnable/Seagate Backup Plus Drive/New_Renders/Angle{0}/".format(myscene.split("chamberangles/angle")[-1].split('.')[0]) + myout
bpy.ops.render.render( write_still=True ) 
