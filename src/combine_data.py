import argparse
import simplification
import csv
import pandas
from tqdm import tqdm
import os
from shutil import copyfile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--add_file",
                        type=str,
                        help="Input path for added train.tsv.")
    parser.add_argument("--isMR",
                        type = int,
                        default = 1)
    parser.add_argument("--change_label",
                        type = int,
                        default = 0, 
                        help="how much to change label, +2 or +4 or -2 or -4")
    parser.add_argument("--original_dataset",
                        type=str,
                        required=True,
                        help="Input path for original dataset.")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Output path for newly generated train.tsv.")
    args = parser.parse_args()


    if args.isMR == 0:
        sents = []
        labels = []

        with open(os.path.join(args.original_dataset, 'train.tsv'), "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter="\t")
            for row in data:
                if row[0] == 'sentence' or row[0] == 'query':
                    continue
                sents.append(row[0])
                labels.append(row[1])
        with open(args.add_file, "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter="\t")
            for row in data:
                if row[0] == 'sentence' or row[0] == 'query':
                    continue
                sents.append(row[0])
                labels.append(str(int(row[1])+args.change_label))
        dataframe = pandas.DataFrame({'sentence': sents, 'label': labels})
    else:
        IDs = []
        sents = []
        labels = []

        with open(os.path.join(args.original_dataset, 'train.tsv'), "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter="\t")
            for row in data:
                if row[1] == 'sentence':
                    continue
                IDs.append(row[0])
                sents.append(row[1])
                labels.append(row[2])

        with open(args.add_file, "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter="\t")
            for row in data:
                if row[1] == 'sentence':
                    continue
                IDs.append(row[0])
                sents.append(row[1])
                labels.append(str(int(row[2])+args.change_label))
        dataframe = pandas.DataFrame({'': IDs, 'sentence': sents, 'label': labels})

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    dataframe.to_csv(os.path.join(args.output_path, 'train_ordered.tsv'), sep='\t', index=False)
    print("train_ordered.tsv built")

    import random
    out = open(os.path.join(args.output_path, 'train.tsv'), 'w', encoding='utf-8')
    lines = []
    with open(os.path.join(args.output_path, 'train_ordered.tsv'), 'r', encoding='utf-8') as infile:
        for (i,line) in enumerate(infile):
            if i == 0:
                out.write(line)
                continue
            lines.append(line)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    for line in lines:
        out.write(line)
    out.close()
    if args.isMR == 1:
        copyfile(os.path.join(args.original_dataset, 'test.tsv'), os.path.join(args.output_path, 'test.tsv'))
    else:
        copyfile(os.path.join(args.original_dataset, 'dev.tsv'), os.path.join(args.output_path, 'dev.tsv'))

if __name__ == "__main__":
    main()