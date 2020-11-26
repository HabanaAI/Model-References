import argparse
from papermill import execute_notebook
from jupytext import readf, write
from stereo.common.general_utils import tree_base
from nbconvert import export, HTMLExporter, writers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-c', '--conf', type=str, help='')
    parser.add_argument('-o', '--out_dir', help='output directory for eval results',  default='/tmp/')
    parser.add_argument('-i', '--restore_iter',
                        help='number iterations of relevant checkpoint. If -1, then most recent',
                        type=int,
                        default=-1)
    parser.add_argument('-m', '--clim_minZ',
                        help='clim_minZ',
                        type=float,
                        default=4.0)
    parser.add_argument('-x', '--clim_maxZ',
                        help='clim_maxZ',
                        type=float,
                        default=40.0)
    parser.add_argument('--cam', help='on which inference cam to run ',
                        default='main', choices={"main", "frontCornerLeft", "frontCornerRight", "rearCornerLeft",
                                                 "rearCornerRight", "rear"})
    args = vars(parser.parse_args())

    write(readf('%s/notebooks/canonical_examples.py' % tree_base()),  '%s/canonical_examples.ipynb' % args['out_dir'])
    iter = 0 if args['restore_iter'] < 0 else args['restore_iter']
    out_name = '%s.%s.%07d' % (args['conf'].split('/')[-1].split('.json')[0], args['cam'], iter)
    executed = '%s/%s.ipynb' % (args['out_dir'], out_name)
    execute_notebook(input_path='%s/canonical_examples.ipynb' % args['out_dir'],
                    output_path=executed,
                    parameters=args)
    exporter = HTMLExporter()
    exporter.exclude_input = True
    output, resources = export(nb=executed,
                                exporter=exporter)
    fw = writers.FilesWriter()
    fw.build_directory = args['out_dir']
    fw.write(output, resources, notebook_name=out_name)
    print('Done! output can be found at %s/%s.html' % (args['out_dir'], out_name))