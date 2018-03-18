"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import math
import xml.etree.ElementTree as ET


def get_clean_int(xml,tag):
	
	return int(math.ceil(float(xml.find(tag).text)))

def get_str(xml,tag):
	return xml.find(tag).text

def pascal_voc_clean_xml(ANN, pick, exclusive = False):

	print('Parsing for {} {}'.format(
			pick, 'exclusively' * int(exclusive)))
	def pp(l): # pretty printing 
		for i in l: print('{}: {}'.format(i,l[i]))

	def parse(line): # exclude the xml tag
		x = line.decode().split('>')[1].decode().split('<')[0]
		try: r = int(x)
		except: r = x
		return r

	def _int(literal): # for literals supposed to be int 
		return int(float(input(literal)))

	dumps = list()
	cur_dir = os.getcwd()
	os.chdir(ANN)
	annotations = os.listdir('.')
	annotations = [file for file in annotations if '.xml' in file]
	size = len(os.listdir('.'))

	for i, file in enumerate(annotations):

		# progress bar		
		sys.stdout.write('\r')
		percentage = 1. * (i+1) / size
		progress = int(percentage * 20)
		bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
		bar_arg += [file]
		sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
		sys.stdout.flush()
		
		xml=ET.parse(file)
		size_xml=xml.find('size')
		h=get_clean_int(size_xml, 'height')
		w=get_clean_int(size_xml, 'width')
		jpg=get_str(xml, 'filename')
		all=[]
		for object_item in xml.findall('object'):
			current= get_str(object_item, 'name')
			bndbox=object_item.find('bndboox')
			xmin=get_clean_int(bndbox, 'xmin')
			ymin=get_clean_int(bndbox, 'ymin')
			xmax=get_clean_int(bndbox, 'xmax')
			ymax=get_clean_int(bndbox, 'ymax')
			new_object=[current, xmin, ymin, xmax, ymax ]
			all.append(new_object)

		add = [[jpg, [w, h, all]]]
		dumps += add

	# gather all stats
	#stat = dict()
	#for dump in dumps:
	#	all = dump[1][2]
	#	for current in all:
	#		if current[0] in pick:
	#			if current[0] in stat:
	#				stat[current[0]]+=1
	#			else:
	#				stat[current[0]] =1

	print() 
	print('Statistics:')
	#pp(stat)
	print('Dataset size: {}'.format(len(dumps)))
	print dumps[5]
	os.chdir(cur_dir)
	return dumps


if __name__=="__main__":
	pascal_voc_clean_xml('/media/milton/Research1/dataset/traffic_sign/tsinghua_traffic_sign/train_annotation/', [1,5],  False)
	