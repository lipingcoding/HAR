import codecs
from tqdm import tqdm

from path_def import ori_semmed_path, simplified_semmed_path


def purify(in_path, out_path):
    # purify original semmed csv file
    with codecs.open(in_path, 'r', encoding='utf-8',
             errors='ignore') as fin:
        with open(out_path, 'w') as fout:
            for i, line in enumerate(tqdm(fin)):
                line = eval(line.strip()[:-9])
                line = [line[3], line[4], line[8]]
                line = ','.join(line)
                print(line, file=fout)



if __name__ == '__main__':
    purify(ori_semmed_path, simplified_semmed_path)
