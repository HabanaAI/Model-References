import os
import s3fs, boto3
import numpy as np
import subprocess

class Probe(object):

    #############################################################################################
    ################### YOU MUST IMPLEMENT THESE FUNCTIONS IN YOUR CHILD CLASS ##################
    #############################################################################################

    def my_init(self):
        # Sets up any member variables you'll need to use in your update method
        raise NotImplementedError("Must implement my_init method for class inheriting from Probe")

    def my_update(self, *args):
        # Calculates statistics or saves info/images for each batch into the class'
        # member variables so they can be summarized later
        '''
        :param loss: float (batch,1)
        :param clip_name: str (batch,?)
        :param gi: int (batch,1)
        :param output: float (batch,310,720,?) **INVERSE DEPTH IS AT INDEX 0
        :param center_im: float (batch,310,720,1)
        :param inp_ims: float (batch,310,720,?) **INPUT IMAGES THAT ARE NOT CENTER_IM
        :param ground_truth: float (batch,310,720,1) **DEPTH
        :param x: int (batch,310,720,1) **X MESH GRID
        '''
        raise NotImplementedError("Must implement my_update method for class inheriting from Probe")

    def my_summarize(self):
        # Summarizes all the data this instance has seen in the form of a dict
        '''
        :return: A dictionary containing whatever you care about from this instance
        '''
        raise NotImplementedError("Must implement my_summarize method for class inheriting from Probe")

    def my_concatenate(self, dict_lst):
        # Concatenates the dicts produced from each individual instance into one final dict
        '''
        :param dict_lst: A list of dictionaries of format you specified in my_summarize
        :return: A dictionary that contains the compiled info from the dictionaries it received
        '''
        raise NotImplementedError("Must implement my_concatenate method for class inheriting from Probe")

    #############################################################################################



    def __init__(self, json_path, out_dir, restore_iter, idx, num_instances, cam, loss_name, dataset_name, blacklist,
                 suffix, debug, batch_size=12, local=False):
        self.json_path = json_path

        if out_dir == '/mobileye/algo_STEREO3/old_stereo/eval':
            if debug:
                self.out_dir = os.path.join(out_dir, 'debug')
            else:
                self.out_dir = os.path.join(out_dir, cam)
        else:
            self.out_dir = out_dir
        self.s3_dir = os.path.join('eval', cam)
        self.restore_iter = restore_iter
        self.idx = idx
        self.num_instances = num_instances
        self.cam = cam
        self.loss_name = loss_name
        self.dataset_name = dataset_name
        if blacklist is None or len(blacklist) == 0:
            print ("No blacklist")
            self.blacklist = []
        else:
            try:
                s3 = s3fs.S3FileSystem()
                self.blacklist = s3.open(blacklist).read().split('\n')[:-1]
                print ("Blacklist includes %d clips" % (len(self.blacklist)))
            except:
                raise ValueError("Could not open Blacklist")
        self.suffix = suffix
        self.debug = debug
        self.batch_size = batch_size

        self.model_name = os.path.splitext(os.path.split(json_path)[1])[0]
        if suffix != '':
            suffix = '_' + suffix
        self.save_name = self.model_name + '_' + str(restore_iter) + suffix
        self.local = local
        self.my_init()

    def update(self, out_lst):
        # This will be called after every iteration (batch) of inference on each instance individually
        self.my_update(*out_lst)

    def summarize(self):
        # This will be called at the end of all the iterations to sum up the results from ONE instance
        dict = self.my_summarize()
        self.save_summary(dict)

    def save_summary(self, dict):
        # Method to handle the saving once you've summarized your stats/images on an individual instance
        local_path = self.save_name + '.npz'
        np.savez(local_path, **dict)
        s3_client = boto3.client('s3')
        save_path = os.path.join(self.s3_dir, self.save_name + '_' + str(self.idx) + '.npz')
        s3_client.upload_file(local_path, 'mobileye-habana/mobileye-team-stereo', save_path)
        os.remove(local_path)
        print ("Summary saved at " + os.path.join('s3://mobileye-habana/mobileye-team-stereo', save_path))

    def concatenate(self):
        # This will be called after all the instances have finished to sum up the results of the entire dataset
        npz_list = []
        for i in range(self.num_instances):
            temp_path = os.path.join('s3://mobileye-habana/mobileye-team-stereo', self.s3_dir, self.save_name + '_' + str(i) + '.npz')
            dest_path = os.path.join('/tmp', self.save_name + '_' + str(i) + '.npz')
            command = ["aws", "s3", "cp",
                       temp_path,
                       dest_path]

            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print("Executing: {}".format(" ".join(command)))
            _ = p.communicate()
            eval_npz = np.load(dest_path, allow_pickle=True)
            npz_list.append(eval_npz)
        final_dict = self.my_concatenate(npz_list)
        self.save_concatenate_and_cleanup_s3(final_dict)

    def save_concatenate_and_cleanup_s3(self, dict):

        # Save in out_dir
        local_path = os.path.join(self.out_dir, self.save_name + '.npz')
        np.savez(local_path, **dict)
        print ("Summary saved at " + local_path)

        # Cleanup partial files
        s3 = s3fs.S3FileSystem()
        for i in range(self.num_instances):
            temp_path = os.path.join('s3://mobileye-habana/mobileye-team-stereo', self.s3_dir, self.save_name + '_' + str(i) + '.npz')
            s3.rm(temp_path)
            dest_path = os.path.join('/tmp', self.save_name + '_' + str(i) + '.npz')
            os.remove(dest_path)











