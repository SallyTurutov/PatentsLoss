import argparse
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Dataset Argumants'
    )
    # dataset settings (the default is SureChEMBL)
    parser.add_argument('--dataset_name', type=str, default='SureChEMBL', help='get dataset name')
    parser.add_argument('--dataset_full_name', type=str, default='SureChEMBL_DATASET',
                        help='get dataset full name')
    parser.add_argument('--dataset_size', type=int, default=25,
                        help='the number of files in which the information is stored')
    parser.add_argument('--counter_suffix', type=str, default='0000', help='the suffix of the file countering')
    parser.add_argument('--old_file_suffix', type=str, default='csv', help='the suffix of the current data files')
    parser.add_argument('--new_file_suffix', type=str, default='txt', help='the suffix of the new data files')
    parser.add_argument('--data_col', type=int, default=4, help='the column in which the information is stored')
    parser.add_argument('--seed', type=int, default=50, help='base seed')

    args = parser.parse_args()
    return args


def get_old_path(args):
    file_suffix = args.old_file_suffix
    counter_suffix = args.counter_suffix
    data_path = 'dataset/' + args.dataset_name + '/' + args.dataset_full_name + '_' + str(
        i) + counter_suffix + '-' + str(i + 1) + counter_suffix + '.' + file_suffix

    print(' ')
    print('Intermediate_analyse: Loading data from file   => ' + data_path)

    return data_path


def get_new_path(args):
    file_suffix = args.new_file_suffix
    data_path = 'dataset/' + args.dataset_name + '/' + args.dataset_name + '.' + file_suffix

    print(' ')
    print('Intermediate_analyse: Saving data to file   => ' + data_path)

    return data_path


if __name__ == "__main__":
    # parse arguments
    args = parse_arguments()

    for i in range(args.dataset_size):
        # get data
        old_data_path = get_old_path(args)

        data_col = args.data_col
        df = pd.read_csv(old_data_path, header=None, sep='\s+', dtype=str)[[data_col]]

        # save data
        new_data_path = get_new_path(args)
        df.to_csv(new_data_path, header=None, index=False, sep='\n', mode='a')
