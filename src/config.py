import argparse
from distutils.util import strtobool
from utils.file_handlers import *

parser = argparse.ArgumentParser("Setting configurations")

# if type=strtobool, True -> 't' or 'True', False -> 'f' or 'False'
parser.add_argument('--comparator', default='dot', type=str, help="[dot, cos, l2, squared_l2]")  # Only 'dot' is here!
parser.add_argument('--dataset_path', default='fit2022v2', type=str, help="csvfile name of dataset")
parser.add_argument('--emb_model', default='embeddings_all_0_v50.pickle', type=str, help="picklefile name of trained embedding model")
parser.add_argument('--get_date', default=None, help='date_time for path name')
parser.add_argument('--pages_tsv', default='jawiki-20220601-page.sql.tsv', type=str, help="tsvfile name of wikidata page")
parser.add_argument('--summary_th', default=3, type=int, help="extract number of summary")
parser.add_argument('--threshold', default=5, type=int, help="threshold of association")
parser.add_argument('--top_k', default=150, type=int, help="extract top_k's scores")
parser.add_argument('--use_cuda', default='f', type=strtobool, help="using GPU or not")  # Only CPU works!
parser.add_argument('--with_vec', default='f', type=strtobool, help="outputing results with vectors or not. True is NOT reccomended")

args = parser.parse_args()


class KeyWords:
    if args.with_vec: _vec = 'withV'
    else: _vec = 'WOV'

    COMPARED_MODEL = f'{args.dataset_path}_{args.comparator}_{args.top_k}_{_vec}'
    COMPARED_MODEL_PATH = f'models/{COMPARED_MODEL}.json'
    ALL_EMBEDDINGS_PATH = f'models/{args.emb_model}'  # path of a trained graph embedding model
    TITLE2ID_PATH = 'data/jawiki-20220601-title2id.pickle'
    DATASET_PATH = f'data/dataset/{args.dataset_path}.csv'
    IS_LINKED_PATH = 'data/jawiki-20220601_fit2022v2_islinked.pickle'

    DIR_NAME, DATE_TIME = dir_name_getter(args)
    RESULTS_JSON_PATH = f'{DIR_NAME}/results_{DATE_TIME}_{file_name_getter(args)}.json'
    SUMMARY_JSON_PATH = f'{DIR_NAME}/summary_{DATE_TIME}_{file_name_getter(args)}.json'