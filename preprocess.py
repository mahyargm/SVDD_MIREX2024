import soundfile as sf
import os
from pathlib import Path
import shutil
from pathlib import Path
import argparse

def main():
    sr = 44100
    segment_size = 4 #seconds
    min_segment_size = 1 #second

    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('input_type')
    ap.add_argument('output_dir')

    
    args = ap.parse_args()
    data_dir = Path(args.dataset_dir)
    input_type = args.input_type
    output_dir = Path(args.output_dir)
    
    (output_dir / input_type).mkdir(exist_ok = True, parents = True)
    
    for root, _, files in os.walk(data_dir / input_type):
        for file in files:
            path = os.path.join(root, file)
            audio_len = len(sf.SoundFile(path))
            path = Path(path)

            if audio_len/sr >= segment_size + min_segment_size:
                remaining = audio_len
                segment_num = 0
                while remaining >= min_segment_size * sr:
                    try:
                        x, sr = sf.read(path, start=segment_num*segment_size*sr, stop=min((segment_num+1)*segment_size*sr, audio_len))
                    except:
                        prob_dir = Path(data_dir.parent / f'problematic_files/{data_dir.name}/{input_type}')
                        prob_dir.mkdir(exist_ok = True, parents = True)
                        shutil.copyfile(path, (prob_dir / path.name))
                        break
                    if sr!= 44100: 
                        print('Error: sr in not valid!', path)
                        sr = 44100
                        break
                    output_path = output_dir / path.parent.name /(path.name[:-5] + f'_{segment_num}.flac')
                    sf.write(output_path, x, sr)
                    remaining -= segment_size * sr
                    segment_num+=1
            else:
                shutil.copyfile(path, output_dir / path.parent.name / (path.name[:-5] + f'_0.flac'))

if __name__=='__main__':
    main()