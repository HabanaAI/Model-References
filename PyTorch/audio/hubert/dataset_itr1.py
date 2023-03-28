# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
import argparse, requests, os, shutil, tarfile
from tqdm import tqdm


def get_train_links(key):
    return {'100': ['https://www.openslr.org/resources/12/train-clean-100.tar.gz'],
            '360': ['https://www.openslr.org/resources/12/train-clean-360.tar.gz'],
            '500': ['https://www.openslr.org/resources/12/train-other-500.tar.gz'],
            '960': ['https://www.openslr.org/resources/12/train-clean-100.tar.gz',
                    'https://www.openslr.org/resources/12/train-clean-360.tar.gz',
                    'https://www.openslr.org/resources/12/train-other-500.tar.gz'],
            'dev': ['https://www.openslr.org/resources/12/dev-other.tar.gz',
                    'https://www.openslr.org/resources/12/dev-clean.tar.gz'],
            'test': ['https://www.openslr.org/resources/12/test-clean.tar.gz',
                    'https://www.openslr.org/resources/12/test-other.tar.gz']}[key]

def exists_so_skip(f):
    def helper(path, *args, **kwargs):
        if os.path.exists(path):
            print(f'{path} exists, so skipping')
            return
        return f(path, *args, **kwargs)
    return helper

@exists_so_skip
def download_link(download_loc, url):
    with requests.get(url, stream=True) as r:
        total_length = int(r.headers.get("Content-Length"))
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
            with open(download_loc, 'wb')as output:
                shutil.copyfileobj(raw, output)

@exists_so_skip
def mkdir(path):
    os.mkdir(path)

@exists_so_skip
def extract(outloc, tar_fl_name):
    with tarfile.open(tar_fl_name) as tar:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            tar.extract(member, path=outloc)

@exists_so_skip
def move(dst, src):
    shutil.move(src, dst)



def download(download_dir, librispeech_split, cleanup):
    list_links = get_train_links(librispeech_split)
    mkdir(download_dir)
    raw_dl_path = f'{download_dir}/downloaded_raw/'
    mkdir(raw_dl_path)
    tar_fl_names = [f"{raw_dl_path}/{link.split('/')[-1]}" for link in list_links]
    [download_link(loc, link) for loc, link in zip(tar_fl_names, list_links)]
    extract_loc = f'{download_dir}/{librispeech_split}_extract/'
    mkdir(extract_loc)
    extract_locs = [f"{extract_loc}/{tar_fl_name.split('/')[-1].split('.')[0]}/" for tar_fl_name in tar_fl_names]
    for ext_loc, tar_fl_name in zip(extract_locs, tar_fl_names):
        extract(ext_loc, tar_fl_name)


    if args.cleanup:
        for tar_fl in tar_fl_names:
            print(f'Deleting {tar_fl}')
            os.remove(tar_fl)
        print(f'Deleting {extract_loc}')
        shutil.rmtree(raw_dl_path)

    print(f'Extracted data here: {extract_loc}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--download_dir", required=True, type=str)
    parser.add_argument("--librispeech_split", required=False, choices=['100','300','500','960','dev','test'], type=str,
                        default='960',
                        help='Which section of librispeech to download')
    parser.add_argument('--cleanup', action='store_true', help='Delete tar.gz etc')
    args = parser.parse_args()
    if args.librispeech_split != '960':
        print(f'You chose {args.librispeech_split} as your split. 960 is the full split')
    download(args.download_dir, args.librispeech_split, args.cleanup)

#python dataset_itr1.py --download_dir=/scratch2/hubert/itr1_data --librispeech_split 960 --cleanup
