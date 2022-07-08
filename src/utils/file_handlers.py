import os
import pandas as pd
import datetime
import json, csv, pickle
from typing import Union


# decide a file name
def file_name_getter(args) -> str:
    if args.with_vec:
        vec = 'withV'
    else:
        vec = 'WOV'
    return f'{vec}_th{args.threshold}'


# get a directory name
def dir_name_getter(args) -> str:
    # get date
    if type(args.get_date) is str:
        date_time = args.get_date
    else:
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        date_time = now.strftime('%y%m%d_%H%M%S')

    dataset_name = args.dataset_path.replace('.csv', '')
    save_dir = f"results/{date_time}_{dataset_name}_{file_name_getter(args)}"

    os.makedirs(save_dir, exist_ok=True)  # make dir if not exists
    return save_dir, date_time


# return a path and date_time for saving results
def save_path_getter(args, name:str, filetype:str) -> str:
    dir_name, date_time = dir_name_getter(args)
    return f'{dir_name}/{name}_{date_time}_{file_name_getter(args)}.{filetype}', date_time


# read .csv
def csv_reader(path:str, encoding='utf-8') -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=0, encoding=encoding, engine="python")
    except:
        df = pd.read_csv(path, header=0, engine="python")
    return df


# write .csv with header
def csv_writer(path:str, results:list, header:list, encoding='utf-8') -> None:
    with open(path, 'w', newline="",  encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"Successfully output {path}!")


# read .json
def json_reader(path:str, encoding='utf-8') -> dict:
    with open(path, encoding=encoding) as f:
        return json.load(f)


# write .json
def json_writer(path:str, results:dict, encoding='utf-8') -> None:
    with open(path, 'w', encoding=encoding) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Successfully output {path}!")


# read .pickle
def pickle_reader(path:str) -> Union[dict, list, pd.DataFrame]:
    with open(path, 'rb') as f:
        any_type_instance = pickle.load(f)
    return any_type_instance


# write .pickle
def pickle_writer(path:str, results:Union[dict, list]) -> None:
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Successfully output {path}!")