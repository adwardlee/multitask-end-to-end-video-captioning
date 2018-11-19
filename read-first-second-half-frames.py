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
parser.add_argument("-rgboutfile", type = str, default = './5mix_list.txt')
parser.add_argument("-flowxoutfile", type = str, default = './25flowx_list.txt')
parser.add_argument("-flowyoutfile", type = str, default = './25flowy_list.txt')
parser.add_argument("-rgb_prefix", type = str, default = 'frame_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument("-modality", type = str, default = 'rgb')
parser.add_argument("-framenum", type = int, default = 5)
args = parser.parse_args()

src_dir = args.src_dir
#rgboutfile = args.rgboutfile
#train_num = [1:]
stack_depth = 0
if args.modality == 'rgb':
    stack_depth = 1
    rgboutfile = open(args.rgboutfile,'w')
elif args.modality == 'flow':
    stack_depth = 5
    flowxoutfile = open(args.flowxoutfile,'w')
    flowyoutfile = open(args.flowyoutfile,'w')

for root,subfolders, filename in os.walk(src_dir):
	subfolders = natural_sort(subfolders)
	for folders in subfolders:
		files = os.listdir(os.path.join(root,folders))
		frame_cnt = 0
		for filenames in sorted(files):				
			if 'frame' in filenames:
				frame_cnt += 1
		#frame_ticks = map(lambda x: x+1, xrange(10))
                quarter = frame_cnt/4
                half_num = frame_cnt/2
                step = (quarter-1)/(args.framenum)


                if step >0:
                   #frame_ticks_1 = range(step,min((step + step * (args.framenum-1)+1),quarter),step)
                   #frame_ticks_2 = range(quarter,min( quarter + step *(args.framenum-1)+1,half_num),step)
                   #frame_ticks_3 = range(half_num,min( half_num + step *(args.framenum-1)+1,half_num+quarter),step)
                   #frame_ticks_4 = range(half_num+quarter,min( half_num+quarter + step *(args.framenum-1)+1,frame_cnt),step)
                   #frame_ticks_1.extend(frame_ticks_2)
                   frame1 = quarter/3
                   frame2 = 2*(quarter/3)
                   frame3 = quarter + quarter/2
                   frame4 = half_num + (quarter)/2
                   frame5 = half_num + quarter + quarter/2
                   frame_ticks = []
                   frame_ticks.append(frame1)
                   frame_ticks.append(frame2)
                   frame_ticks.append(frame3)
                   frame_ticks.append(frame4)
                   frame_ticks.append(frame5)
		#step = (frame_cnt - stack_depth) / args.framenum
		#if step > 0:
    		# 	frame_ticks = range(1, min((2 + step * (args.framenum-1)), frame_cnt+1), step)
		else:
    		 	frame_ticks = range(frame_cnt)
		 	frame_ticks = map(lambda x: x+1, frame_ticks)
		 	frame_ticks.extend([1]*(args.framenum - frame_cnt))
                        print('dsadsda :"::')

		for tick in frame_ticks:
			if args.modality == 'rgb':
				name = '{}{:06d}.jpg'.format(args.rgb_prefix, tick)	
				rgboutfile.write('%s,%s\n'%(folders,os.path.join(root,folders,name)))
			if args.modality == 'flow':
				frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
				for idx in frame_idx:
					flowxname = '{}{:06d}.jpg'.format(args.flow_x_prefix, idx)
					flowxoutfile.write('%s,%s\n'%(folders,os.path.join(root,folders,flowxname)))
					flowyname = '{}{:06d}.jpg'.format(args.flow_y_prefix, idx)
					flowyoutfile.write('%s,%s\n'%(folders,os.path.join(root,folders,flowyname)))
