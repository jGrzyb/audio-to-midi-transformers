import mido
import numpy as np
import os
import pandas as pd
import librosa
import numpy as np
import cv2


def getSpectogram(wave_path, sample_rate=12_800, n_fft=2048, hop_length=256, n_mels=512):
    samples, sr = librosa.load(wave_path, sr=sample_rate)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db, sr


def getMessageDf(file_path):
    mid = mido.MidiFile(file_path)
    # note_msg = filter(lambda y: y.type == 'note_on', mid.tracks[1])
    note_msg = map(lambda x: (x, x.time), mid.tracks[1])
    note_df = pd.DataFrame(note_msg, columns=['other', 'time'])

    note_df['time'] = note_df['time'].cumsum()
    note_df = note_df[note_df['other'].apply(lambda x: x.type == 'note_on')]
    note_df['note'] = note_df['other'].apply(lambda x: x.note)
    note_df['velocity'] = note_df['other'].apply(lambda x: x.velocity)
    note_df = note_df.drop(columns=['other']).reset_index(drop=True)
    return note_df


def makeChunks(midi_path, wave_path, name, output_dir, chunk_size=128, step_size=64, clear=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if clear:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    image, sr = getSpectogram(wave_path)
    midi = getMessageDf(midi_path)
    time_per_frame = 256 / sr

    image = (image - image.min()) / (image.max() - image.min()) * 255
    length = int((image.shape[1] - chunk_size) / step_size)

    files = []

    for i in range(length):
        im_out_path = f'{name}_{i}.png'
        csv_out_path = f'{name}_{i}.csv'

        tmp = image[:, i * step_size:i *step_size + chunk_size].astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, im_out_path), tmp)

        start_time = i * step_size * time_per_frame * 1000
        end_time = (i * step_size + chunk_size) * time_per_frame * 1000
        midi_tmp = midi[(midi['time'] >= start_time)& (midi['time'] <= end_time)]
        midi_tmp = midi_tmp.copy()
        midi_tmp['time'] = midi_tmp['time'] - int(start_time)
        midi_tmp.to_csv(os.path.join(output_dir, csv_out_path), index=False)

        files.append((im_out_path, csv_out_path))
    return files


def createDataset(wav_midi_list: list[tuple], output_dir: str, chunk_size=128, step_size=64):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = []

    for i, (wave_path, midi_path) in enumerate(wav_midi_list):
        name = wave_path.split('/')[-1].split('.')[0]
        chunks_files = makeChunks(midi_path, wave_path, name, output_dir, chunk_size, step_size)
        files.extend(chunks_files)
    return files


if __name__ == '__main__':
    output = 'waves'
    wav_midi = pd.read_csv('wav_midi.csv').apply(lambda x: (os.path.join(output, x['wav']), os.path.join(output, x['midi'])), axis=1).tolist()
    wav_midi[:3]

    files = createDataset(wav_midi, 'dataset')
    files = pd.DataFrame(files, columns=['image', 'midi'])
    files.to_csv('dataset.csv', index=False, header=['image', 'midi'])
