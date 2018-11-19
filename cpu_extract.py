import os
import glob
import sys
from multiprocessing import Pool, current_process

import argparse
out_path = ''


def dump_frames(vid_path):
    import cv2
    video = cv2.VideoCapture(vid_path[0])
    vid_name = vid_path[0].split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        resize_frame = cv2.resize(frame, (new_size[0], new_size[1])) 
        assert ret
        cv2.imwrite('{}/frame_{:06d}.jpg'.format(out_full_path, i), resize_frame)
        access_path = '{}/frame_{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print '{} done'.format(vid_name)
    sys.stdout.flush()
    return file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--flow_type", type=str, default='frame', help = 'frame')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=224, help='resize image width')
    parser.add_argument("--new_height", type=int, default=224, help='resize image height')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    flow_type = args.flow_type

    ext = args.ext
    new_size = (args.new_width, args.new_height)

    if not os.path.isdir(out_path):
        print "creating folder: "+out_path
        os.makedirs(out_path)
    
    #vid_list = glob.glob(src_path+'/*/*.'+ext)
    vid_list = glob.glob(src_path+'/*.'+ext)
    for i in xrange(len(vid_list)):    
	print vid_list[i]

    print ' len : ',len(vid_list)
    print '\(sss\)'    

    print len(vid_list)
    pool = Pool(num_worker)
    a = pool.map(dump_frames,zip(vid_list,xrange(len(vid_list))))
    print "done"
