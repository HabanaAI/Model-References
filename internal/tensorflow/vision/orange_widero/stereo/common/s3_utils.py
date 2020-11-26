import os
import pickle


def my_exists(name, *args, **kwargs):
    if name.startswith("s3"):
        import s3fs
        fs = s3fs.S3FileSystem()
        return fs.exists(name, *args, **kwargs)
    else:
        return os.path.exists(name, *args, **kwargs)


def my_glob(pathname, *args, **kwargs):
    if pathname.startswith("s3"):
        cache_path = os.path.join('_cache',pathname[5:])
        from glob import glob
        glob_cache =  glob(cache_path, *args, **kwargs)
        if glob_cache:
            return glob_cache
        import s3fs
        fs = s3fs.S3FileSystem()
        # return fs.glob(pathname, *args, **kwargs)
        return ["s3://{}".format(f) for f in fs.glob(pathname, *args, **kwargs)]
    else:
        from glob import glob
        return glob(pathname, *args, **kwargs)


def my_open(name, *args, **kwargs):
    if name.startswith("s3"):
        cache_path = os.path.join('_cache',name[5:])
        if os.path.exists(cache_path):
            return open(cache_path, *args, **kwargs)
        import s3fs
        fs = s3fs.S3FileSystem()
        return fs.open(name, *args, **kwargs)
    else:
        return open(name, *args, **kwargs)


def my_listdir(path):
    if path.startswith("s3"):
        import s3fs
        fs = s3fs.S3FileSystem()
        return [s.split('/')[-1] for s in fs.ls(path)]
    else:
        return os.listdir(path)


def s3_copy(src, dst, extra_args=None, recursive=True):
    import subprocess
    command = ["aws", "s3", "cp",
               src,
               dst]
    if recursive:
        command.append("--recursive")
    if extra_args is not None:
        command.extend(extra_args)
    print("Executing: {}".format(" ".join(command)))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _ = p.communicate()


def my_save(path, data, is_obj=True, *args, **kwargs):
    if path.startswith("s3"):
        import s3fs
        s3 = s3fs.S3FileSystem()
        open_func = s3.open
    else:
        open_func = open
    with open_func(path, "wb", *args, **kwargs) as f:
        if is_obj:
            pickle.dump(data, f, protocol=2)
        else:
            f.write(data)


def my_read(path, is_obj=True, *args, **kwargs):
    if path.startswith("s3"):
        import s3fs
        s3 = s3fs.S3FileSystem()
        open_func = s3.open
    else:
        open_func = open
    with open_func(path, "rb", *args, **kwargs) as f:
        if is_obj:
            return pickle.load(f)
        else:
            return f.read()
