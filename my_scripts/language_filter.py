import fasttext
import argparse


bin_path = 'xxx/fasttext/lid.176.bin'

try:
    model = fasttext.load_model(bin_path)
except ValueError:
    model = fasttext.load_model(bin_path.replace('data', 'data04'))


def get_lang_info(line,relax):
    if relax:
        labels, probas = model.predict(line.strip().lower(),k=10)
    else:
        labels, probas = model.predict(line.strip().lower())
    return labels, probas


parser = argparse.ArgumentParser()
parser.add_argument('--in-src-file', help='input source file path')
parser.add_argument('--in-tgt-file', help='input target file path')
parser.add_argument('--src-lang', help='source language')
parser.add_argument('--tgt-lang', help='target language')
parser.add_argument('--src-threshold', default=0.3,help='source language filter threshold')
parser.add_argument('--tgt-threshold', default=0.3,help='target language filter threshold')
parser.add_argument('--output-dir', help='output directory')
parser.add_argument('--stage-id', help='Stage id for identifiers of the generated files')
parser.add_argument('--relax', default=False, action='store_true',
                    help='relax mode. In this mode, only the pairs that are highly probably from another language'
                         'will be removed')
# parser.add_argument('--relax-threshold', default=0.95, help='threshold for the relax mode')
args = parser.parse_args()

seen = set()
src_proba_threshold = float(args.src_threshold)
tgt_proba_threshold = float(args.tgt_threshold)
# relax_threshold = float(args.relax_threshold)
src_lang, tgt_lang = args.src_lang, args.tgt_lang

with open(args.in_src_file, 'r', encoding='utf-8') as in_src_file, \
        open(args.in_tgt_file, 'r', encoding='utf-8') as in_tgt_file, \
        open(args.output_dir + f'/{args.stage_id}.{args.src_lang}-{args.tgt_lang}.{args.src_lang}', 'w+', encoding='utf-8') as out_src_file, \
        open(args.output_dir + f'/{args.stage_id}.{args.src_lang}-{args.tgt_lang}.{args.tgt_lang}', 'w+', encoding='utf-8') as out_tgt_file, \
        open(args.output_dir + f'/{args.stage_id}.{args.src_lang}-{args.tgt_lang}.{args.src_lang}.trash',
             'w+', encoding='utf-8') as src_trash_file, \
        open(args.output_dir + f'/{args.stage_id}.{args.src_lang}-{args.tgt_lang}.{args.tgt_lang}.trash',
             'w+', encoding='utf-8') as tgt_trash_file:
    if args.relax:
        print('*****Relax mode*****')
    for src_line, tgt_line in zip(in_src_file, in_tgt_file):
        src_line, tgt_line = src_line.strip(), tgt_line.strip()
        if not src_line or not tgt_line:
            continue
        if (src_line, tgt_line) in seen:
            continue
        seen.add((src_line, tgt_line))
        model_src_lang, src_proba = get_lang_info(src_line,args.relax)
        model_tgt_lang, tgt_proba = get_lang_info(tgt_line,args.relax)
        if args.relax:
            reject_cond = (f'__label__{src_lang}' not in model_src_lang ) or \
                            (f'__label__{tgt_lang}' not in model_tgt_lang)
        else:
            model_src_lang = model_src_lang[0]
            model_tgt_lang = model_tgt_lang[0]
            reject_cond = not ((model_src_lang == f'__label__{src_lang}' and src_proba >= src_proba_threshold) and
                               (model_tgt_lang == f'__label__{tgt_lang}' and tgt_proba >= tgt_proba_threshold))
        if reject_cond:
            src_trash_file.write(f'{model_src_lang}\t{src_proba}\t{src_line}\n')
            tgt_trash_file.write(f'{model_tgt_lang}\t{tgt_proba}\t{tgt_line}\n')
        else:
            out_src_file.write(src_line + '\n')
            out_tgt_file.write(tgt_line + '\n')
