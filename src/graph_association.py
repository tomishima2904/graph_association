import numpy as np
import time
from tqdm import tqdm
import sys, os

# original modules
from utils.comparators import Comparators
from utils.converters import *
from utils.file_handlers import *
from config import args, KeyWords


class GraphAssociation:
    """
    b: batch size. total num of tasks. default b is approximately 80.
    n: num of stimulations. default n is 5.
    d: dimension. default d is 400.
    r: ranks. default r is 150.
    v: vocabulary(verticies) size. all titles in wikidata. v is approximately 2M.
    """

    def __init__(self, args) -> None:

        self.start_time = time.time()
        self.args = args  # set options
        self.comparator = Comparators(self.args)  # define dist func
        self.ranks = [i for i in range(self.args.top_k)]
        self.compared_stims_path = KeyWords.COMPARED_MODEL_PATH

        t = time.time()
        print("Setting models...")

        # Load a dict (id:vec)
        self.emb_path = KeyWords.ALL_EMBEDDINGS_PATH
        self.id2vector = pickle_reader(self.emb_path)
        self.all_indicies = list(self.id2vector.keys())
        self.all_vecs = list(self.id2vector.values())

        # Load a dict (id:title)
        title2id_dict_path = KeyWords.TITLE2ID_PATH
        self.title2id = pickle_reader(title2id_dict_path)
        self.num_v = len(self.title2id)

        # Set a dataset
        self.dataset_path = KeyWords.DATASET_PATH
        self.csv_data = csv_reader(self.dataset_path)
        self.num_b = len(self.csv_data)

        print(f"Setted models in {time.time()-t}s")


    # Compare stims with all titles using trained graph embeddings
    def compare(self) -> None:
        t = time.time()
        print("Comparing...")
        all_compared_stims = {}

        for b, row in tqdm(enumerate(self.csv_data.itertuples()), total=self.num_b):
            stims = [stim for stim in eval(row.stims)]
            for n, stim in enumerate(stims):
                print(f"{b:2d}-{n}: {stim}")
                if stim in all_compared_stims.keys(): continue  # continue if stim is already compared
                stim_id = self.title2id[stim]
                all_compared_vecs = self.comparator(self.all_vecs, self.id2vector[stim_id])  # compare a stim's vec with all title's vecs
                sorted_args = [np.argsort(all_compared_vecs)[::-1][r].item() for r in self.ranks]
                compared_vecs = [np.sort(all_compared_vecs)[::-1][r].tolist() for r in self.ranks]  # extract top k's vecs associated with a stim
                compared_indicies = [self.all_indicies[idx] for idx in sorted_args]
                compared_titles = [self._get_keys_from_value(self.title2id, str(idx)) for idx in compared_indicies]
                if self.args.with_vec:
                    compared_raw_vecs = [self.id2vector[str(idx)].tolist() for idx in compared_indicies]
                    compared_results = [
                        {'id':idx, 'title': title, 'score': score, 'vec': vec}
                        for idx, title, score, vec
                        in zip(compared_indicies, compared_titles, compared_vecs, compared_raw_vecs)
                    ]
                else:
                    compared_results = [
                        {'id':idx, 'title': title, 'score': score}
                        for idx, title, score
                        in zip(compared_indicies, compared_titles, compared_vecs)
                    ]
                all_compared_stims[stim] = compared_results  # shape==[r, 3(4)]

        print(f'Compared in {time.time()-t}s')
        json_writer(self.compared_stims_path, all_compared_stims)

        self.is_linked()  # connection check


    # Are there connections between stim and titles ?
    def is_linked(self) -> None:
        t = time.time()
        print("Connection checking...")
        all_compared_stims = json_reader(self.compared_stims_path)
        islinked_list = pd.DataFrame(pickle_reader(KeyWords.IS_LINKED_PATH))
        for stim in all_compared_stims.keys():
            for r, asso in enumerate(all_compared_stims[stim]):
                extract_islinked_list = islinked_list[(islinked_list[0]==int(self.title2id[stim])) & (islinked_list[2]==int(self.title2id[asso['title']]))]
                if len(extract_islinked_list) > 0:
                    all_compared_stims[stim][r]['islinked'] = 1
                else:
                    all_compared_stims[stim][r]['islinked'] = 0
        print(f'Checked in {time.time()-t}s')
        json_writer(self.compared_stims_path, all_compared_stims)
        print(f'Checking done in {time.time()-t}s')


    # Associaton !!
    def associate(self) -> dict:

        # Associate titles with stimulations using trained graph embeddings
        t = time.time()
        print("Associating...")
        all_compared_stims = json_reader(self.compared_stims_path)
        all_info = {}  # <- THIS HAS ALL INFORMATION
        summary_info = {}
        islinked_list = pd.DataFrame(pickle_reader(KeyWords.IS_LINKED_PATH))

        for b, row in tqdm(enumerate(self.csv_data.itertuples()), total=self.num_b):
            #  Make a dict which has all information
            stims = [stim for stim in eval(row.stims)]
            stims_ids = [self.title2id[stim] for stim in stims]
            stims_scores = [np.dot(self.id2vector[idx], self.id2vector[self.title2id[row.answer]]).item() for idx in stims_ids]
            extract_islinked_list = [
                len(islinked_list[
                    (islinked_list[0]==int(self.title2id[row.answer])) & (islinked_list[2]==int(self.title2id[stim]))
                ])
                for stim in stims
            ]
            extract_islinked_list = [1 if islinked_num >0 else 0 for islinked_num in extract_islinked_list]
            stims_dict = [
                {'id': idx, 'stim': stim, 'score': score, 'islinked': islinked, 'associated': all_compared_stims[stim]}
                for idx, stim, score, islinked in zip(stims_ids, stims, stims_scores, extract_islinked_list)
            ]
            summary_stims_dict = [{stim: islinked} for stim, islinked in zip(stims, extract_islinked_list)]
            all_info[b] = {'cat':row.category, 'ans': row.answer, 'stims': stims_dict}
            summary_info[b] = {'cat': row.category, 'ans': row.answer, 'stims': summary_stims_dict}

            # Predict answer(s) based on stim["associated"]
            research_dict = {}
            found_flag = False
            for r in self.ranks:
                # search titles associated with stims every rank
                for n, stim in enumerate(all_info[b]['stims']):
                    # remove stims from associated words
                    if stim['associated'][r]['title'] in stims:
                        continue
                    elif stim['associated'][r]['title'] in research_dict.keys():
                        research_dict[stim['associated'][r]['title']] += 1
                    else:
                        research_dict[stim['associated'][r]['title']] = 1

                # if found the title that is associated with all stims
                if self.args.threshold in research_dict.values():
                    found_flag = True
                    break

            # record research results
            all_info[b]['results'] = {}
            all_info[b]['results']['predictions'] = research_dict
            summary_info[b]['results'] = {}
            research_dict_summary = {k: v for k, v in research_dict.items() if v >= self.args.summary_th}
            if found_flag:
                all_info[b]['results']['rank'] = r + 1
                summary_info[b]['results']['rank'] = r + 1
                summary_info[b]['results']['predictions'] = research_dict_summary
            else:
                all_info[b]['results']['rank'] = 0
                summary_info[b]['results']['rank'] = 0
                summary_info[b]['results']['predictions'] = 'Not found'

        print(f'Associated in {time.time()-t}s')
        return all_info, summary_info


    # Get key(s) from a value
    def _get_keys_from_value(self, d:dict, val:str) -> list:
        keys = [k for k, v in d.items() if v == val]
        if len(keys) == 1:
            return keys[0]
        else:
            print(val)
            sys.exit(1)


if __name__ == "__main__":
    start_time = time.time()
    graph_association = GraphAssociation(args)
    # graph_association.compare()
    results, summary = graph_association.associate()
    os.makedirs(KeyWords.DIR_NAME, exist_ok=True)  # make dir if not exists
    json_writer(KeyWords.RESULTS_JSON_PATH, results)
    json_writer(KeyWords.SUMMARY_JSON_PATH, summary)
    print(f'Done in {time.time()-start_time}')