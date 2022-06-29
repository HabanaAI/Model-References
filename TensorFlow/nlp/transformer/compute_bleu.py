###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import subprocess
from argparse import ArgumentParser
from TensorFlow.common.tb_utils import TBSummary


parser = ArgumentParser()
parser.add_argument('--decoded_file', '-df', type=str, default='wmt14.tgt.tok',
                    help='Decoded file produced by t2t-decode command.')
parser.add_argument('--log_dir', '-ld', type=str, default=None,
                    help='Where to store TensorBoard summary file, '
                         'if None summary will not be saved.')
args = parser.parse_args()

def get_sacremoses_version():
    ver_line = subprocess.run(['sacremoses', '--version'], stdout=subprocess.PIPE).stdout.decode()
    return tuple(map(int, ver_line.split()[-1].split('.')))

def get_sacremoses_cmd(version):
    if version >= (0, 0, 42):
        return ['sacremoses', '-l', 'de', 'detokenize']
    else:
        return ['sacremoses', 'detokenize', '-l', 'de']

def main():
    detok = subprocess.run(get_sacremoses_cmd(get_sacremoses_version()),
                           stdin=open(args.decoded_file, 'r'),
                           stdout=subprocess.PIPE)
    bleu = subprocess.run(['sacrebleu', '-t', 'wmt14', '-l', 'en-de', '-b'],
                          input=detok.stdout, stdout=subprocess.PIPE)
    score = bleu.stdout.decode()
    print('BLEU:', score)

    if args.log_dir is not None:
        with TBSummary(args.log_dir) as tb:
            tb.add_scalar('accuracy', float(score), 0)


if __name__ == '__main__':
    main()
