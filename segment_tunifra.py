import os
import re
import sys
import torch
import argparse
import torchaudio
import torchaudio.transforms as T

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

def clean_text(l):
    l = l.strip() #remove initial final spaces
    l = l.replace('...','')
    l = l.replace('&amp;','')
    l = l.replace('#','')
    l = l.replace('*','')
    l = l.replace('+','')
    l = l.replace('_',' ')
    l = re.sub(r"\s{2,}", " ", l) #replace multiple consecutive spaces (spaces, tabs, newlines) with a single space
    return l

def get_segments(ftrs):
    segments = []
    with open(ftrs, 'r') as fd:
        prev_str = ''
        for l in fd:
            match = re.search(r'<Sync time="([\d.]+)"/>', prev_str)
            if match:
                time_value = float(match.group(1))
                sentence = clean_text(l)
                segments.append({'start':time_value, 'sentence':sentence})
            prev_str = l
    return segments

def split_in_segments(name, ftrs, fwav, out_dir, fdo):
    s = 0
    wav, sr = load_wav(fwav, sr=16000)
    segments = get_segments(ftrs)
    #print(f"found {len(segments)} segments in {ftrs}")
    total_duration = 0.
    for i in range(len(segments)):
        sentence = segments[i]['sentence']
        if len(sentence):
            start = int(segments[i]['start']*sr)
            stop = int(segments[i+1]['start']*sr) if len(segments) > i+1 else len(wav)
            out_wav = os.path.join(out_dir, f"{name}_seg{s}.wav")
            #print(f"saving {out_wav} [{start}, {stop}) sec={1.0*(stop-start)/sr}")
            total_duration += 1.0*(stop-start)/sr
            save_wav(wav[start:stop], sr, out_wav)
            fdo.write(out_wav + '\t' + sentence + '\n')
            fdo.flush()
            s += 1
    return s, total_duration


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parse TuniFra corpus and build segments with corresponding transcriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_dir', type=str, required=False, default='/nfs/RESEARCH/crego/projects/COMMUTE/PartageIWSLT/Tun_transcription_Elyadata', help='Pakita corpus directory.')
    parser.add_argument('--out_dir', type=str, required=True, default=None, help='Directory (absolute path) where segments are left. [out_dir].map is also built.')
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
        s, total_duration = split_in_segments(name, args.base_dir+'/'+name+'.trs', args.base_dir+'/'+name+'.wav', args.out_dir, fdo)
        S += s
        print(f'{i}:{name}, {s} avg duration = {total_duration/s:.2f} => {S} segments', file=sys.stderr, flush=True)
    fdo.close()
    print('Done', flush=True)
