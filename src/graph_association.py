import numpy as np
import time
from tqdm import tqdm
import sys

# original modules
from utils.comparators import Comparators
from utils.converters import *
from utils.file_handlers import *
from config import args


class GraphAssociation:
    """
    b: batch size. total num of tasks. default b is approximately 80.
    n: num of stimulations. default n is 5.
    d: dimension. default d is 400.
    r: ranks. default r is 150.
    v: vocabulary(verticies) size. all titles in wikidata. v is approximately 2M.
    """

    def __init__(self, args) -> None:

        self.args = args  # set options
        self.comparator = Comparators(self.args)  # define dist func
        self.start_time = time.time()

        t = time.time()
        print("Setting models...")

        # Load a dict (id:vec)
        self.emb_path = f'models/{self.args.emb_model}'  # path of a trained graph embedding model
        self.id2vector = pickle_reader(self.emb_path)
        self.all_indicies = list(self.id2vector.keys())
        self.all_vecs = list(self.id2vector.values())

        # Load a dict (id:title)
        id2_title_dict_path = 'data/jawiki-20220601-id2title.pickle'
        self.id2title = pickle_reader(id2_title_dict_path)
        self.num_v = len(self.id2title)

        # Get ready for a dict which has all information
        self.dataset_path = f'data/dataset/{self.args.dataset_path}'
        csv_data = csv_reader(self.dataset_path)
        self.all_info = {}  # <- THIS HAS ALL INFORMATION
        self.all_stims_vecs = []
        self.num_b = len(csv_data)
        for b, row in tqdm(enumerate(csv_data.itertuples()), total=self.num_b):
            stims = [stim for stim in eval(row.stims)]
            stims_ids = [self._get_keys_from_value(self.id2title, stim) for stim in stims]
            if self.args.with_vec:
                stims_vecs = [self.id2vector[idx].tolist() for idx in stims_ids]  # embed all stims
                self.all_stims_vecs.append(stims_vecs)  # [b,n,d]
                stims_dict = [{'id': idx, 'stim': stim, 'vec': vec} for idx, stim, vec, in zip(stims_ids, stims, stims_vecs)]
            else:
                stims_dict = [{'id': idx, 'stim': stim} for idx, stim in zip(stims_ids, stims)]
            self.all_info[b] = {'cat':row.category, 'ans': row.answer, 'stims': stims_dict}

        print(f"Setted models in {time.time()-t}s")


    # Associaton !!
    def __call__(self) -> dict:

        # Associate titles with stimulations using trained graph embeddings
        t = time.time()
        print("Associating...")
        ranks = [i for i in range(self.args.top_k)]
        for b in tqdm(range(self.num_b), total=self.num_b):
            for n, stim in enumerate(self.all_info[b]['stims']):
                ts = time.time()
                all_compared_vecs = self.comparator(self.all_vecs, self.id2vector[stim['id']])  # compare a stim's vec with all title's vecs
                print(f'Compared a stim with all titles in {time.time()-ts}')
                ts = time.time()
                sorted_args = [np.argsort(all_compared_vecs)[::-1][r].item() for r in ranks]
                compared_vecs = [np.sort(all_compared_vecs)[::-1][r].tolist() for r in ranks]  # extract top k's vecs associated with a stim
                print(f'Sorted all titles order by score in {time.time()-ts}')
                compared_indicies = [self.all_indicies[idx] for idx in sorted_args]
                compared_titles = [self.id2title[str(idx)] for idx in compared_indicies]
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
                stim['associated'] = compared_results  # shape==[r, 4]
                if n==0: break  ####################################################################  THIS LINE HAS TO BE REMOVED!!!

            # Predict answer(s) based on stim["results"]
            research_dict = {}
            found_flag = False
            for r in ranks:
                # search titles associated with stims every rank
                for n, stim in enumerate(self.all_info[b]['stims']):
                    if stim['associated'][r]['title'] in research_dict.keys():
                        research_dict[stim['associated'][r]['title']] += 1
                    else:
                        research_dict[stim['associated'][r]['title']] = 1
                    if n == 0: break ####################################################################  THIS LINE HAS TO BE REMOVED!!!
                # if found the title that is associated with all stims
                if self.args.threshold in research_dict.values():
                    found_flag = True
                    break

            # record research results
            self.all_info[b]['results'] = {}
            self.all_info[b]['results']['predictions'] = research_dict
            if found_flag: self.all_info[b]['results']['rank'] = r + 1
            else: self.all_info[b]['results']['rank'] = 0
            if b==0: break  ####################################################################  THIS LINE HAS TO BE REMOVED!!!

        print(f'Associated in {time.time()-t}s')
        return self.all_info


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
    results = graph_association()
    output_json_path, _ = save_path_getter(args, name='results', filetype='json')
    json_writer(output_json_path, results)
    print(f'Done in {time.time()-start_time}')