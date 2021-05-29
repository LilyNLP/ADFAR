import argparse
import simplification
import csv
import pandas
from tqdm import tqdm
import os
from shutil import copyfile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--adversaries_path",
                        type=str,
                        help="Input path for adversaries.txt.")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Output path for newly generated train.tsv.")
    parser.add_argument("--times",
                        type=int,
                        help="times")
    parser.add_argument("--change",
                        type=int,
                        help="change or not")
    parser.add_argument("--txtortsv",
                        type=str,
                        help="output is txt or tsv.")
    parser.add_argument("--datasize",
                        type=int,
                        help="datasize for txt output")
    args = parser.parse_args()

    import pandas
    labels = []
    sentences = []

    with open(args.adversaries_path) as f:
        lines = f.readlines()

    if args.change == 1:
        for ii in range(args.times):
            i = 0
            for line in tqdm(lines):
                if i>args.datasize*3:
                    break
                line = line.strip()
                line = line.replace('\n ', ' ').replace('\t', ' ')
                if line == '':
                    continue
                if line[:12] == 'adv sent (0)':
                    labels.append(3)
                    sentences.append(line.replace('adv sent (0):', ''))
                if line[:12] == 'adv sent (1)':
                    labels.append(2)
                    sentences.append(line.replace('adv sent (1):', ''))
                i += 1

    if args.change == 0:
        for ii in range(args.times):
            i = 0
            for line in tqdm(lines):
                if i>args.datasize*3:
                    break
                line = line.strip()
                line = line.replace('\n ', ' ').replace('\t', ' ')
                if line == '':
                    continue
                if line[:12] == 'adv sent (0)':
                    labels.append(1)
                    sentences.append(line.replace('adv sent (0):', ''))
                if line[:12] == 'adv sent (1)':
                    labels.append(0)
                    sentences.append(line.replace('adv sent (1):', ''))
                i += 1

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    if args.txtortsv == 'tsv':
        result_dataframe = pandas.DataFrame({'sentence': sentences, 'label': labels})
        result_dataframe.to_csv(os.path.join(args.output_path, 'pure_adversaries_ordered.tsv'), sep='\t', index=False)

        import random
        out = open(os.path.join(args.output_path, 'pure_adversaries.tsv'), 'w', encoding='utf-8')
        lines = []
        with open(os.path.join(args.output_path, 'pure_adversaries_ordered.tsv'), 'r', encoding='utf-8') as infile:
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

    if args.txtortsv == 'txt':
        out = open(os.path.join(args.output_path, 'adversaries_for_detection.txt'), 'w')
        for (i, (sentence, label)) in enumerate(zip(sentences,labels)):
            if i >= args.datasize:
                break
            out.write(str(label) + ' ' + sentence + '\n')
        out.close

if __name__ == "__main__":
    main()