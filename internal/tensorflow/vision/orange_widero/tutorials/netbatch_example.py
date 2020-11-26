import argparse
import os
import cv2
from devkit.clip import MeClip
from me_jobs_runner import Task, dispatch_tasks, NETBATCH


SCRIPT_PATH = '/mobileye/algo_STEREO3/ofers/code/tutorials/netbatch_example.py'


def parse_args():
	
	parser = argparse.ArgumentParser(description='Save the 13th image of each clip')
	clip = parser.add_mutually_exclusive_group(required=True)
	clip.add_argument('-c', metavar='clip_name', dest='clip_name', help='clip name')
	clip.add_argument('-C', metavar='clip_names_path', dest='clip_names_path', help='path to file contains list of clips')
	parser.add_argument('-o', metavar='output_path', dest='output_path', type=str, required=True,
					help='output path for save_somthing')

	return parser.parse_args()


def get_clip_names_from_file(file_path):
	with open(file_path, 'rb') as f:
		return f.read().splitlines()


def do_something(clip_name):
	clip = MeClip(clip_name)
	frame = clip.get_frame(frame=13, camera_name='main', tone_map='ltm')
	return frame[0]['pyr'][-2].im


def save_somthing(clip_name, output_path, something):
	output_dir = os.path.join(output_path, clip_name)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	cv2.imwrite(output_dir + '/image.png', something)


def single_run(clip_name, output_path):
	something = do_something(clip_name)
	save_somthing(clip_name, output_path, something)


def netbatch_run(clip_names_path, output_path):
	nb_task = 'example_task'
	cmd_lines = []
	clip_names = get_clip_names_from_file(clip_names_path)
	for clip_name in clip_names:
		cmd_line = 'python %s -c %s -o %s' %(SCRIPT_PATH, clip_name, output_path)
		cmd_lines.append(cmd_line)
	task = Task(task_name=nb_task, commands=cmd_lines, input_folders_or_clips=clip_names,
				output_folders=output_path, timeout_minutes=0, memory_gb=8)
	dispatch_tasks([task], task_name=nb_task, log_dir=os.path.join(output_path, 'nb_logs'), qslot='ALGO_STEREO', dispatch_type=NETBATCH)


def main():
	args = parse_args()
	if args.clip_names_path:
		netbatch_run(args.clip_names_path, args.output_path)
	elif args.clip_name:
		single_run(args.clip_name, args.output_path)

if __name__ == '__main__':
	main()
