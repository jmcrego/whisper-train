import os
import sys
import torch
import argparse
import torchaudio
import torchaudio.transforms as T

#############################       PARSE ARGUMENTS       #####################################

def load_wav(file_path, sr=None):
    waveform, sampling_rate = torchaudio.load(file_path)
    if waveform.shape[0] != 2:
        raise ValueError(f"The WAV file '{file_path}' does not have 2 channels.")
    if sr is not None and sr != sampling_rate:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=sr)
        waveform = resampler(waveform)
        sampling_rate = sr
    return waveform.numpy(), sampling_rate

def save_wav(waveform, sample_rate, output_path):
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add the channel dimension                                                                                                                                                                                                                              
    torchaudio.save(output_path, waveform, sample_rate)

def find_files_with_extension(base_dir, extension):
    valid_files = {}
    for root, dirs, files in os.walk(base_dir): # Walk through the directory recursively
        for file_name in files:
            if file_name.endswith(extension):
                path = os.path.join(root, file_name)
                desc = file_name[:-len(extension)]
                valid_files[desc] = path
    print(f'found {len(valid_files)} {extension} files', file=sys.stderr)
    return valid_files

def split_in_segments(name, fstm, fwav, out_dir, fdo):
    s = 0
    wav, sr = load_wav(fwav, sr=16000)
    with open(fstm, 'r') as fd:
        for l in fd:
            if l.startswith(name):
                toks = l.strip().split(' ')
                if len(toks) > 6 and toks[2].startswith(name):
                    channel = int(toks[1]) - 1
                    start = float(toks[3])
                    stop = float(toks[4])
                    sentence = ' '.join(toks[6:])
                    #print(name, channel, start, stop, sentence)
                    out_wav = os.path.join(out_dir, f"{name}_ch{channel}_seg{s}.wav")
                    save_wav(wav[channel,int(start*sr):int(stop*sr)], sr, out_wav)
                    fdo.write(out_wav + '\t' + sentence + '\n')
                    fdo.flush()                    
                    s += 1
    return s

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parse pakita corpus and build segments with corresponding transcriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_dir', type=str, required=False, default='/nfs/RESEARCH/crego/projects/pakita_fra-cts/corpus/pakita', help='Pakita corpus directory.')
    parser.add_argument('--out_dir', type=str, required=True, default=None, help='Directory (absolute path) where segments are left. [out_dir].map is also built.')
    args = parser.parse_args()

    if not os.path.isabs(args.out_dir):
        raise ValueError(f"Directory '{args.out_dir}' must be an absolute path")
    
    if os.path.exists(args.out_dir):
        raise ValueError(f"Directory '{args.out_dir}' already exists.")
    
    ids = set()
    for l in sys.stdin:
        ids.add(l.strip())
    print(f'found {len(ids)} ids', file=sys.stderr, flush=True)
    
    stm2path = find_files_with_extension(args.base_dir, '.stm')
    wav2path = find_files_with_extension(args.base_dir, '.wav')
    names = set(stm2path.keys()) & set(wav2path.keys())
    print(f'found {len(names)} wav & stm files', file=sys.stderr, flush=True)
    names = names & ids
    print(f'found {len(names)} wav&stm & ids files', file=sys.stderr, flush=True)
    
    print(f"Building '{args.out_dir}.map' and directory {args.out_dir}", file=sys.stderr, flush=True)
    os.makedirs(args.out_dir)
    fdo = open(args.out_dir+'.map', 'w')    
    S = 0
    for i, name in enumerate(names):
        s = split_in_segments(name, stm2path[name], wav2path[name], args.out_dir, fdo)
        S += s
        print(f'{i}:{name}, {s} => {S} segments', file=sys.stderr, flush=True)
    fdo.close()
    print('Done', flush=True)
