import subprocess
import tensorflow as tf

from stereo.common.s3_utils import my_open
from stereo.models.conf_utils import latest_model
import boto3
import os
import fnmatch


class SyncCheckpointListener(tf.estimator.CheckpointSaverListener):
    """A saver listener to sync checkpoints with s3"""

    def __init__(self, src, dst):
        super(tf.estimator.CheckpointSaverListener, self).__init__()
        self._src = src
        self._dst = dst
        # self._synced = False
        self._p = None  # subprocess

    def begin(self):
        # You can add ops to the graph here.
        # print('Starting the session.')
        try:
            # try to restore existing checkpoint in model_dir
            restore_iter = latest_model('/'.join(self._dst.split('/')[3:])+'/')
            with my_open(os.path.join(self._dst, 'checkpoint'), 'w') as f:
                f.write(u'model_checkpoint_path: "model.ckpt-%d"\n' % restore_iter)
                f.write(u'all_model_checkpoint_paths: "model.ckpt-%d"\n' % restore_iter)
            command = ["aws", "s3", "cp",
                       self._dst,
                       self._src,
                       "--recursive",
                       "--exclude", "*",
                       "--include", "model.ckpt-%d.*" % restore_iter,
                       "--include", "checkpoint"
                       ]
            print("Executing: {}".format(" ".join(command)))
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print('Restoring existing checkpoint! Waiting for checkpoint download to machine to finish')
            _ = p.communicate()
        except:
            print('No existing checkpoint')
            pass

    def before_save(self, session, global_step_value):
        if self._p:
            print('Wait for upload to s3 to finish')
            _ = self._p.communicate()
            # out, _ = self._p.communicate()
            # print(out)

    def after_save(self, session, global_step_value):
        print('Done writing checkpoint {}'.format(global_step_value))
        command = ["aws", "s3", "sync",
                   self._src,
                   self._dst,
                   "--exclude", "*",
                   "--include", "graph.pbtxt",
                   "--include", "model.ckpt*",
                   "--include", "checkpoint",
                   "--include", "events*"
                   ]
        print("Executing: {}".format(" ".join(command)))
        self._p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def end(self, session, global_step_value):
        # print('Done with the session.')
        pass


def make_name_aws_friendly(name):

    name = name.replace('.', '-').replace('_', '-').replace('+', '-')
    return name


def check_job_completed(job_name):
    client = boto3.client('sagemaker')

    response = client.search(
        Resource='TrainingJob',
        SearchExpression={
            'Filters': [
                {
                    'Name': 'TrainingJobName',
                    'Operator': 'Contains',
                    'Value': job_name
                }
            ]
        },
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    res = response['Results']
    if len(res) == 0:
        return False
    res = res[0]['TrainingJob']
    return res['TrainingJobStatus'] != 'InProgress'


def get_training_jobs(start_time, min_job_name=''):
    client = boto3.client('sagemaker')
    username = os.getenv('USERNAME')
    if username is None:
        username = os.getenv('USER')
    params = {'Resource': 'TrainingJob',
              'SearchExpression':
              {
                  'Filters': [
                      {
                          'Name': 'Tags.username',
                          'Operator': 'Equals',
                          'Value': username
                      },
                      {
                          'Name': 'TrainingJobName',
                          'Operator': 'Contains',
                          'Value': min_job_name
                      }
                  ]
              },
              'SortBy': 'CreationTime',
              'SortOrder': 'Descending',
              'MaxResults': 100
             }
    response = client.search(
        **params
    )
    out_results = []
    for job in response['Results']:
        if 'TrainingStartTime' not in job['TrainingJob']:
            continue
        if (job['TrainingJob']['TrainingStartTime'].replace(tzinfo=None) - start_time).total_seconds() > 0:
            out_results.append(job['TrainingJob'])
    return out_results


def kill_instances(start_time, min_job_name):
    client = boto3.client('sagemaker')
    username = os.getenv('USERNAME')
    if username is None:
        username = os.getenv('USER')
    params = {'Resource': 'TrainingJob',
              'SearchExpression':
                  {
                      'Filters': [
                          {
                              'Name': 'Tags.username',
                              'Operator': 'Equals',
                              'Value': username
                          },
                          {
                              'Name': 'TrainingJobName',
                              'Operator': 'Contains',
                              'Value': min_job_name
                          }
                      ]
                  },
              'SortBy': 'CreationTime',
              'SortOrder': 'Descending',
              'MaxResults': 100
              }
    response = client.search(
        **params
    )
    for job in response['Results']:
        if 'TrainingJobName' not in job['TrainingJob'] or 'TrainingStartTime' not in job['TrainingJob']:
            continue
        if (job['TrainingJob']['TrainingStartTime'].replace(tzinfo=None) - start_time).total_seconds() > 0:
            job_name = job['TrainingJob']['TrainingJobName']
            print ("Stopping job %s" % (job_name))
            client.stop_training_job(TrainingJobName=job_name)

    print("All jobs killed")


def get_checkpoint_path(checkpoint_dir, restore_iter=-1):

    if not checkpoint_dir.endswith('/'):  # This is here to fix bug of two models with contained names
        checkpoint_dir += '/'
    if restore_iter < 0:
        restore_iter = latest_model(checkpoint_dir)

    s3 = boto3.resource('s3')
    sm_bucket = s3.Bucket('mobileye-habana/mobileye-team-stereo')
    ckpt_keys = [obj.key for obj in sm_bucket.objects.filter(Prefix=checkpoint_dir) if
                 ('ckpt-' + str(restore_iter) + '.') in obj.key]
    checkpoint_path = ''
    for key in ckpt_keys:
        if '.meta' in key:
            checkpoint_path = 's3://mobileye-habana/mobileye-team-stereo/%s' % key.split('.meta')[0]
            break
    return checkpoint_path


def get_all_tensors_names():
    return [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]


def get_tensors_by_regex(regex):
    all_tensors_names = get_all_tensors_names()
    filtered_tensors_names = fnmatch.filter(all_tensors_names, regex)
    return filtered_tensors_names


def create_valid_job_name(job_name):
    sm_job_name = job_name.replace('.', '-').replace('_', '-').replace('+', '-')
    return sm_job_name
