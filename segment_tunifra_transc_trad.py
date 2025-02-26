import os
import re
import sys
import torch
import argparse
import torchaudio
import torchaudio.transforms as T
from pyarabic.araby import strip_tashkeel, strip_harakat, normalize_hamza, normalize_ligature

def load_wav(file_path, sr=None):
    waveform, sampling_rate = torchaudio.load(file_path)
    if sr is not None and sr != sampling_rate:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=sr)
        waveform = resampler(waveform)
        sampling_rate = sr
    return waveform.squeeze().numpy(), sampling_rate

def save_wav(waveform, sample_rate, output_path):
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add the channel dimension                                                                                                                                                                                                                              
    torchaudio.save(output_path, waveform, sample_rate)

def normalize(text):
    ### Étape 1 : Supprimer toute la ponctuation sauf '%' et '@'
    text = re.sub(r"[^\w\s%@]-", "", text, flags=re.UNICODE)
    ### Étape 2 : Supprimer les diacritiques, Hamzas et Maddas
    text = strip_harakat(text)  # Supprime les diacritiques (Tashkeel)
    text = normalize_hamza(text)  # Normalise ou supprime les Hamzas
    text = re.sub(r"[ـ]", "", text)  # Supprime les Maddas spécifiquement
    ### Étape 3 : Translitérer les chiffres arabes orientaux en occidentaux
    arabic_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    text = text.translate(arabic_to_western)    
    return text

def clean_transcription(l, norm=False):
    l = l.strip() #remove initial final spaces
    if norm:
        l = normalize(l)
    l = l.replace('...','')
    l = l.replace('&amp;','')
    l = l.replace('#','')
    l = l.replace('*','')
    l = l.replace('+','')
    l = l.replace('_',' ')
    l = re.sub(r"\s{2,}", " ", l) #replace multiple consecutive spaces (spaces, tabs, newlines) with a single space
    l = l.strip()
    return l

def clean_translation(l):
    l = l.strip() #remove initial final spaces
    l = l.replace('...','')
#    l = l.replace('&amp;','')
#    l = l.replace('#','')
#    l = l.replace('*','')
#    l = l.replace('+','')
#    l = l.replace('_',' ')
    l = l.replace('=','')
    l = l.replace('%pw',' ')
    l = re.sub(r"\s{2,}", " ", l) #replace multiple consecutive spaces (spaces, tabs, newlines) with a single space
    l = l.strip()
    return l

def get_segments(ftrs, no_overlap=False, norm=False):
    segments = []
    with open(ftrs, 'r') as fd:
        prev_str = ''
        for l in fd:
            if no_overlap and '...' in l: 
                continue
            match = re.search(r'<Sync time="([\d.]+)"/>', prev_str)
            if match:
                time_value = f"{float(match.group(1)):.3f}" ### force to have 3 decimals
                sentence = clean_transcription(l, norm)
                if len(sentence):
                    segments.append({'start':time_value, 'sentence':sentence})
            prev_str = l
    print(f"\tfound {len(segments)} segments in {ftrs}")
    return segments

def find_time_value_in_segments(time_value, time2i):
    if time_value in time2i:
        return time2i[time_value]
    return None

def get_translations(fstm, segments, no_overlap=False):
    time2i = {d['start']:i for i,d in enumerate(segments)}
    with open(fstm, 'r') as fd:
        nfound = 0
        for l in fd: 
            if no_overlap and '...' in l: 
                continue
            tok = l.strip().split(' ')
            if len(tok) >= 7:
                time_value = f"{float(tok[3]):.3f}" ### force to have 3 decimals
                i = find_time_value_in_segments(time_value, time2i)
                if i is not None:
                    translation = clean_translation(' '.join(tok[6:]))
                    if len(translation):
                        segments[i]['translation'] = translation
                        nfound += 1
    print(f"\tfound {nfound} translations in {fstm}")
    return segments

def split_in_segments(name, ftrs, fwav, out_dir, fdo, no_overlap=False, norm=False):
    s = 0
    wav, sr = load_wav(fwav, sr=16000)
    segments = get_segments(ftrs, no_overlap, norm)
    segments = get_translations(ftrs.replace('Tun_transcription_Elyadata','Tun2Fr_Translation').replace('.trs','.fr.stm'), segments, no_overlap)

    total_duration = 0.
    for i in range(len(segments)):
        if 'translation' in segments[i]:
            sentence = segments[i]['sentence']
            translation = segments[i]['translation']
            start = int(float(segments[i]['start'])*sr)
            stop = int(float(segments[i+1]['start'])*sr) if len(segments) > i+1 else len(wav)
            out_wav = os.path.join(out_dir, f"{name}_seg{s}.wav")
            #print(f"saving {out_wav} [{start}, {stop}) sec={1.0*(stop-start)/sr}")
            total_duration += 1.0*(stop-start)/sr
            save_wav(wav[start:stop], sr, out_wav)
            fdo.write(out_wav + '\t' + sentence + '\t' + translation + '\n')
            fdo.flush()
            s += 1
#        else:
#            print(f"\tdiscarded {segments[i]['start']} {segments[i]['sentence']}")
    return s, total_duration


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parse TuniFra corpus and build segments with corresponding transcriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_dir', type=str, required=False, default='/nfs/RESEARCH/crego/projects/COMMUTE/PartageIWSLT/Tun_transcription_Elyadata', help='Pakita corpus directory.')
    parser.add_argument('--out_dir', type=str, required=True, default=None, help='Directory (absolute path) where segments are left. [out_dir].map is also built.')
    parser.add_argument('--no_overlap', action='store_true', help='Discard overlapped speech (marked with \'...\')')
    parser.add_argument('--norm', action='store_true', help='Apply normalization rules over Tunisian Arabic transcriptions.')
    args = parser.parse_args()

    if not os.path.isabs(args.out_dir):
        raise ValueError(f"Directory '{args.out_dir}' must be an absolute path")

    if os.path.exists(args.out_dir):
        raise ValueError(f"Directory '{args.out_dir}' already exists.")

    names = [f[:-4] for f in os.listdir(args.base_dir) if f.endswith(".wav")]
    print(f'found {len(names)} ids', file=sys.stderr, flush=True)
    
    print(f"Building file {args.out_dir}.map and directory {args.out_dir}", file=sys.stderr, flush=True)
    os.makedirs(args.out_dir)
    fdo = open(args.out_dir+'.map', 'w')    
    S = 0
    for i, name in enumerate(names):
        s, total_duration = split_in_segments(name, args.base_dir+'/'+name+'.trs', args.base_dir+'/'+name+'.wav', args.out_dir, fdo, args.no_overlap, args.norm)
        S += s
        print(f'{i}:{name}, {s} avg duration = {total_duration/s:.2f} => {S} segments', file=sys.stderr, flush=True)
    fdo.close()
    print('Done', flush=True)
