import os
import re
import argparse
import glob
import torch
import pandas as pd


DF_COLUMNS = ['iter', 'micro_iter', 'log_class', 'module_name', 'out_tensor_index', 'pearson', 't1', 't2']
REPORT_FILENAME = 'tlog_report.xlsx'


def parse_args():
    parser = argparse.ArgumentParser(description='tensor logger files compare utility')
    parser.add_argument('-1', '--path1', type=str, help='path to 1st directory that contains tlog files', required=True)
    parser.add_argument('-2', '--path2', type=str, help='path to 2nd directory that contains tlog files', required=True)
    parser.add_argument('-o', '--out-path', type=str, help='output report path', required=True)
    args = parser.parse_args()
    return args


def get_tlog_files(path):
    files = glob.glob(os.path.join(path, 'tensor_logger_rank_*.pt'))
    return files


def unwrap(val):
    unwrapped = []

    def _unwrap(v):
        if isinstance(v, list) or isinstance(v, tuple):
            for e in v:
                _unwrap(e)
        elif isinstance(v, torch.Tensor):
            unwrapped.append(v)
        else:
            assert 'unknown type {}'.format(type(v))

    _unwrap(val)
    return unwrapped


def validate_compatible_files(files1, files2):
    def get_sorted_ranks(files):
        ranks = []
        for f in files:
            res = re.search('.*/.*_(.*).pt', f)
            if res:
                ranks.append(res.group(1))
        ranks.sort()
        return ranks

    # verify same number of files
    n_files1, n_files2 = len(files1), len(files2)
    if n_files1 != n_files2:
        raise Exception('Different number of partitions ({}, {}) - aborting'.format(n_files1, n_files2))
    elif n_files1 == 0:
        raise Exception('No tensor logger files found')

    # verify same ranks
    ranks1 = get_sorted_ranks(files1)
    ranks2 = get_sorted_ranks(files2)
    if ranks1 != ranks2:
        raise Exception('Different partition ranks (ranks1={}, ranks2={})'.format(ranks1, ranks2))


def align_model_inputs(i, model_inputs):
    # convert from:
    #   dict with keys 'inputs', 'kwargs' where each is a list of micro-batches
    # to:
    #   dict of input names where each has a list of micro batches
    keys = sorted(list(model_inputs.keys()))
    if keys != ['inputs', 'kwargs']:
        raise Exception('Missing keys in model_inputs for iteration={}: keys={}'.format(i, keys))

    inputs = {}
    for i, t in enumerate(model_inputs['inputs']):
        for k, v in enumerate(t):
            input_name = 'inputs_' + str(k)
            if input_name not in inputs:
                inputs[input_name] = []
            inputs[input_name].append(v)

    for i, d in enumerate(model_inputs['kwargs']):
        for k, v in d.items():
            kwarg_name = 'kwargs_' + k
            if kwarg_name not in inputs:
                inputs[kwarg_name] = []
            inputs['kwargs_' + k].append(v)

    return inputs


def pearson(x, y):
    with torch.no_grad():
        if torch.all(torch.eq(x, y)):
            p = torch.tensor(0.)
        else:
            vx = x - torch.mean(x.float())
            vy = y - torch.mean(y.float())
            if torch.count_nonzero(vx) == 0. and torch.count_nonzero(vy) == 0.:
                p = torch.tensor(0.)
            else:
                p = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return p


def compare_log_class(df, i, log_class, data1, data2):
    keys1 = sorted(data1.keys())
    keys2 = sorted(data2.keys())
    if keys1 != keys2:
        raise Exception('Different keys in iteration iter={} log_class={}: keys1={} keys2={}'.format(
            i, log_class, keys1, keys2))

    # run on all modules
    for module_name in keys1:
        n_micro_iter1 = len(data1[module_name])
        n_micro_iter2 = len(data2[module_name])
        if n_micro_iter1 != n_micro_iter2:
            raise Exception(
                'Different number of micro iterations key={} in iteration={} log_class={}: n1={} n2={}'.format(
                    module_name, i, log_class, n_micro_iter1, n_micro_iter2))

        # run on micro batch iterations
        for ui in range(n_micro_iter1):
            tensors1 = unwrap(data1[module_name][ui])
            tensors2 = unwrap(data2[module_name][ui])
            n_tensors1 = len(tensors1)
            n_tensors2 = len(tensors2)
            if n_tensors1 != n_tensors2:
                raise Exception(
                    'Different number of output tensors for module={} in iteration={} log_class={}: n1={} n2={}'.format(
                        module_name, i, log_class, n_tensors1, n_tensors2))

            # run over module outputs (can be more than 1)
            for ti in range(n_tensors1):
                t1 = tensors1[ti]
                t2 = tensors2[ti]
                p = pearson(t1, t2)
                row = {
                    'iter': i,
                    'micro_iter': ui,
                    'log_class': log_class,
                    'module_name': module_name,
                    'out_tensor_index': ti,
                    'pearson': p.numpy(),
                    't1': t1.numpy(),
                    't2': t2.numpy(),
                }
                df = df.append(row, ignore_index=True)

    return df


def compare_iter(df, i, data1, data2):
    log_classes = ['model_inputs', 'fwd_act', 'bwd_grad_out', 'bwd_grad_in']
    sorted_log_classes = sorted(log_classes)

    # verify that both data dicts contain exactly above keys
    keys1 = sorted(data1.keys())
    keys2 = sorted(data2.keys())
    if keys1 != sorted_log_classes or keys2 != sorted_log_classes:
        raise Exception('Invalid keys in iteration iter={}: keys1={} keys2={}'.format(i, keys1, keys2))

    data1['model_inputs'] = align_model_inputs(i, data1['model_inputs'])
    data2['model_inputs'] = align_model_inputs(i, data2['model_inputs'])

    for log_class in log_classes:
        keys1 = sorted(data1[log_class].keys())
        keys2 = sorted(data2[log_class].keys())
        if keys1 != keys2:
            raise Exception('Different tensor names for iter={} log_class={} keys1={} keys2={}'.format(
                i, log_class, keys1, keys2))
        df = compare_log_class(df, i, log_class, data1[log_class], data2[log_class])

    return df


def compare_files(f1, f2):
    data1 = torch.load(f1)
    data2 = torch.load(f2)

    n_iter1 = len(data1)
    n_iter2 = len(data2)
    if n_iter1 != n_iter2:
        raise Exception('Different number of mini-batch iterations (n_iter1={}, n_iter2={}'.format(n_iter1, n_iter2))

    df = pd.DataFrame(data=None, columns=DF_COLUMNS)
    for i in range(n_iter1):
        data1_iter, data2_iter = data1[i], data2[i]
        df = compare_iter(df, i, data1_iter, data2_iter)
    return df


def save_to_xls(args, df):
    filename = os.path.join(args.out_path, REPORT_FILENAME)
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='report', index=False)

    # set autofit column width
    worksheet = writer.sheets['report']

    def get_col_widths():
        return [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]

    for i, width in enumerate(get_col_widths()):
        worksheet.set_column(i, i, width)

    # limit the column length for last two columns (tensor values)
    n_columns = len(df.columns)
    worksheet.set_column(n_columns-2, n_columns-1, 20)
    worksheet.set_column(n_columns-1, n_columns-1, 20)

    writer.save()


def main():
    args = parse_args()
    path1 = args.path1
    path2 = args.path2

    files1 = get_tlog_files(path1)
    files2 = get_tlog_files(path2)

    # perform basic validation on compatibility of captured tlog files from both paths
    validate_compatible_files(files1, files2)

    # compare all files
    files1.sort()
    files2.sort()
    agg_df = pd.DataFrame(data=None, columns=['partition']+DF_COLUMNS)

    for i in range(len(files1)):
        f1, f2 = files1[i], files2[i]
        df = compare_files(f1, f2)
        df['partition'] = i
        agg_df = agg_df.append(df, ignore_index=True)

    # save report
    agg_df = agg_df.astype({'pearson': float})
    save_to_xls(args, agg_df)

    # check if all is identical
    pearson_vals = agg_df['pearson'].to_numpy()
    all_zeroes = (pearson_vals[0] == 0) and (pearson_vals[0] == pearson_vals).all()   # noqa
    if all_zeroes:
        print('ALL IDENTICAL')
    else:
        print('NOT IDENTICAL')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    main()
