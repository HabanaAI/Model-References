import os
import re
import subprocess
from time import time, sleep
from datetime import timedelta
import tempfile
import shutil
import fnmatch
from s3fs import S3FileSystem


def find_synced_files(sync_log):
    synced = re.findall('to (.*?)\s', sync_log)
    synced = [os.path.abspath(f) for f in synced]
    return synced


def find_models(model_dir, models):
    if model_dir.startswith("s3"):
        fs = S3FileSystem()
        all_models = fs.ls(model_dir)
        model_dir = model_dir.replace("s3://", "")
    else:
        all_models = os.listdir(model_dir)

    matched_models = []
    for m in models:
        matched_models.extend(
            fnmatch.filter(all_models, os.path.join(model_dir, m))
        )
    matched_models = map(os.path.basename, matched_models)
    return list(set(matched_models))


def s3_sync(local_path, s3_path, include_items, log_dir, delete=True):
    t0 = time()
    if not isinstance(include_items, list):
        raise TypeError("include_items must be a list")

    log_file = os.path.join(log_dir, "s3_sync.log")
    print("Writing s3 log to: {}".format(log_file))

    includes = [s
                for item in include_items
                for s in ["--include", item]]

    cmd = ["aws", "s3", "sync", s3_path, local_path,
           "--exclude", "*"] + includes
    if delete:
        cmd.append("--delete")
    stdout = ""
    with open(log_file, 'w+') as f:
        _ = subprocess.call(cmd,
                            stdout=f,
                            stderr=subprocess.STDOUT)
        f.seek(0)
        stdout = f.read()

    # Find synced files in log
    synced = find_synced_files(stdout)

    elapsed_seconds = time() - t0

    print("AWS sync is Finished. Time elapsed: {}".format(timedelta(seconds=elapsed_seconds)))
    print("{} synced files".format(len(synced)))
    print(os.linesep.join(synced))
    return synced


def sync_tfevents(local_model_dir, s3_model_dir, models, log_dir=None, use_only_train_test=True):
    print("Syncing models:")
    print(os.linesep.join(models))
    if use_only_train_test:
        print("Showing only train and test directories")
        tf_events = [m + "/train/*tfevents*" for m in models] + \
                    [m + "/test/*tfevents*" for m in models]
    else:
        tf_events = [m + "/*tfevents*" for m in models]
    _ = s3_sync(local_model_dir, s3_model_dir, tf_events, log_dir=log_dir)


def clean_model_dir(local_model_dir, models, use_only_train_test=True, reset_dir=False):
    for m in models:
        m = os.path.join(local_model_dir, m)
        if os.path.isdir(m):
            if reset_dir:
                import shutil
                shutil.rmtree(m)
            elif use_only_train_test:
                to_remove = [os.path.join(m, f) for f in os.listdir(m)
                             if os.path.isfile(os.path.join(m, f)) and "tfevents" in f]
                for path in to_remove:
                    os.remove(path)
                    print("Removed: {}".format(path))


def launch_tb(model_dir, models, port=6006):
    import tensorboard
    log_dirs = "".join(["{}:{},".format(m, os.path.join(model_dir, m))
                        for m in models])[:-1]

    tb = tensorboard.program.TensorBoard()
    tb.configure(logdir=log_dirs,
                 host="0.0.0.0",
                 port=port)
    url = tb.launch()
    print("TensorBoard %s started at %s" % (tensorboard.version.VERSION, url))
    pid = os.getpid()
    print("PID = %d; use 'kill %d' to quit" % (pid, pid))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run TensorBoard and sync s3 tfevents files to a local directory")
    parser.add_argument('-m', '--models', nargs='+', required=True,
                        help="Space separated model names. "
                             "The names can be written with glob syntax, e.g. -m model_1.1 \"model_2.*\"")
    parser.add_argument('-d', '--local_dir', default="/mobileye/algo_STEREO3/stereo/cache/tfevents",
                        help="Local models directory. default: %(default)s")
    parser.add_argument('-s', '--s3_dir', default="s3://mobileye-habana/mobileye-team-stereo/models",
                        help="s3 models directory. default: %(default)s")
    parser.add_argument('-p', '--port', type=int, default=6006,
                        help="Port to serve TensorBoard on. default: %(default)s")
    parser.add_argument('-t', '--sync_time', type=int, default=10,
                        help="Sync interval with s3 in minutes. default: %(default)s")
    parser.add_argument('-l', '--s3_log', type=str,
                        help="Path for s3 sync logs. "
                             "If not specified a tmp directory is created")
    parser.add_argument('-n', '--not_local', action='store_true',
                        help="Run TB on s3 files, without syncing locally. much slower!")
    parser.add_argument('-g', '--not_only_train_test', action='store_true',
                        help="Show also tfevents that are not in train and test directories. "
                             "Using this flag will add the global step/sec for each run")
    parser.add_argument('-r', '--reset', action='store_true',
                        help="Reset the local dir of the selected models")

    args = parser.parse_args()

    tmp_log = args.s3_log is None
    if tmp_log:
        args.s3_log = tempfile.mkdtemp()

    local_model_dir = args.local_dir
    s3_model_dir = args.s3_dir
    models = find_models(model_dir=s3_model_dir, models=args.models)
    if len(models) == 0:
        raise ValueError("Could not find models that match: {}".format(args.models))
    clean_model_dir(local_model_dir, models,
                    use_only_train_test=not args.not_only_train_test,
                    reset_dir=args.reset)

    def sync():
        sync_tfevents(local_model_dir, s3_model_dir, models, args.s3_log,
                      use_only_train_test=not args.not_only_train_test)

    launch_tb(s3_model_dir if args.not_local else local_model_dir,
              models=models, port=args.port)

    while True:
        try:
            if not args.not_local:
                sync()
            sleep(args.sync_time * 60)
        except KeyboardInterrupt:
            break

    # delete tmp log dir
    if tmp_log:
        shutil.rmtree(args.s3_log)

    print()
    print("Shutting down")
