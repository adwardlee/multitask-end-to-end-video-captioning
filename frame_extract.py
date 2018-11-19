import os
import numpy as np
import glob
import argparse
import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)



parser = argparse.ArgumentParser(description="read frames in folder")
parser.add_argument("-src_dir",type = str, default='/media/llj/storage/microsoft-corpus/youtube_frame_flow')
parser.add_argument("-outfile", type = str, default = './video_list.txt')
args = parser.parse_args()

src_dir = args.src_dir
outfile = args.outfile
#train_num = [1:]


with open(outfile,'w') as output_file:
	for root,subfolders, filename in os.walk(src_dir):
		subfolders = natural_sort(subfolders)
		for folders in subfolders:
			files = os.listdir(os.path.join(root,folders))
			#count = 0
			for filenames in sorted(files):
				#if count%20 ==0:
					if 'frame' in filenames:
						output_file.write('%s,%s\n'%(folders,os.path.join(root,folders,filenames)))
				#count += 1
			#count = 0
			##for filen in sorted(filename):
				#print 'filename ',filename
				#count += 1
				#if count % 10 == 0:
					#output_file.write('%s,%s\n'%(folders,os.path.join(root,folders,filen)))
				#output_file.write('%s\n'%(os.path.join(root,filen)))
