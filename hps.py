import os
import itertools
import csv
import datetime
from train import train
from src.options import Options

CONFIGURATIONS = [
    [
        ["--iterations"], [2],
        ["--lr"], [0.1, 0.01, 0.001, 0.0001, 0.00001],
        ["--ncf"], [[30, 50, 64], [30, 50, 64, 64]]
    ]
]

CSV_COLUMNS = ["config", "iterations", "lr", "ncf", "loss"]

def create_env():
    timestamp = "{:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    out_dir = "hps_results/" + timestamp
    os.makedirs(out_dir)

    csv_path = "{0}/results.csv".format(out_dir)

    return out_dir, csv_path

def get_cases_for_config(config):
    return list(itertools.product(*config))

def create_args_list(case):
    args_list = []
    for x in case:
        if type(x) is list:
            args_list.extend([str(y) for y in x])
        else:
            args_list.append(str(x))

    return args_list

def print_config(index, csv_writer):
    print("\n@@@@@@@@@@@@@@@ Configuration #{0} @@@@@@@@@@@@@@@".format(index))
    csv_writer.writerow({"config" : index})

def print_case(case, csv_writer):
    print("\n########## Hyperparams ##########")

    case_dict = {}
    for i in range(0, len(case), 2):
        hp, val = case[i].split("--")[1], case[i+1]
        print("\t{0}={1}".format(hp, val))
        case_dict[hp] = val

    csv_writer.writerow(case_dict)
    print("#################################\n")

def print_loss(loss, csv_writer):
    print("---------- Loss = {0} ----------".format(loss))
    csv_writer.writerow({"loss" : loss})

def run_cases(cases, csv_writer):
    parser = Options.create_parser()
    for case in cases:
        print_case(case, csv_writer)
        args_list = create_args_list(case)
        args = parser.parse_args(args_list)
        loss = train(args)
        print_loss(loss, csv_writer)

def main():
    out_dir, csv_path = create_env()
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        csv_writer.writeheader()
        
        for index, config in enumerate(CONFIGURATIONS):
            print_config(index, csv_writer)
            cases = get_cases_for_config(config)
            run_cases(cases, csv_writer)
            

if __name__ == "__main__":
    main()