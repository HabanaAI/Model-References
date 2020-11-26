import argparse
from os.path import join

from me_jobs_runner import Task, dispatch_tasks, NETBATCH

SCRIPT_PATH='/mobileye/algo_STEREO2/stereo/car_hood_mask/get_car_hood_mask.py'


def parse_args():
	parser = argparse.ArgumentParser(description='Get a mask of car hood for each camera in session')
	parser.add_argument('-c', metavar='clip_names', dest='clip_names', required=True,
                    help='Clip name or path to file contains list of clips')
	parser.add_argument('-o', metavar='output_path', dest='output_path', type=str, required=True,
                    help='output path for masks (and images, if -s is specified)')
	parser.add_argument('-s', dest='save_imgs', action='store_true', default=False,
                    help='save also images (default - True)')
	parser.add_argument('-f', dest='from_file', action='store_true', default=False,
                    help='Whether the clip_names are read from file')
	parser.add_argument('-n', dest='netbatch_run', action='store_true', default=False,
                    help='Should the script run on netbatch')
	parser.add_argument('-t', metavar='thresh' , dest='thresh', type=float , default=0.7,
                    help='Threshold that varies for each car')

	return parser.parse_args()

def get_clip_names_from_file(file_path):
	with open(file_path, 'rb') as f:
		return f.read().splitlines()


def single_run(clip_name, output_path, save_imgs, thresh):
	sections_cameras = {
		'Left' : ['frontCornerLeft', 'rearCornerLeft'],
		'Right' : ['frontCornerRight', 'rearCornerRight']
	}
	all_imgs = get_session_imgs(clip_name, sections_cameras, output_path, save_imgs)
	masks = get_cameras_masks(all_imgs, thresh)
	save_cameras_masks(masks, output_path, clip_name, as_obj=True)


def multi_run(clip_names_path, output_path, save_imgs, thresh):
	clip_names = get_clip_names_from_file(clip_names_path) 
	print('Number of clips: %s' % len(clip_names))
	for clip_name in clip_names:
		print('Getting mask of clip: %s' %clip_name)
		single_run(clip_name, output_path, save_imgs, thresh)

def netbatch_run(args):
	clip_names_path, output_path, save_imgs = args.clip_names, args.output_path, args.save_imgs
	clip_names = get_clip_names_from_file(clip_names_path)
	nb_task = 'get_car_hood_mask'
	cmd_lines = []
	for clip_name in clip_names:
		cmd_line = 'python %s -c %s -o %s' %(SCRIPT_PATH, clip_name, output_path)
		if save_imgs:
			cmd_line += ' -s'
		cmd_lines.append(cmd_line)
	task = Task(task_name=nb_task, commands=cmd_lines, input_folders_or_clips=clip_names,
				output_folders=output_path, timeout_minutes=0, memory_gb=8)
	dispatch_tasks([task], task_name=nb_task, log_dir=join(output_path, 'nb_logs'), qslot='ALGO_STEREO', dispatch_type=NETBATCH)


def main():
	args = parse_args()
	if args.netbatch_run:
		netbatch_run(args)
	elif args.from_file:
		multi_run(args.clip_names, args.output_path, args.save_imgs, args.thresh)
	else:
		single_run(args.clip_names, args.output_path, args.save_imgs, args.thresh)


if __name__ == '__main__':
	main()