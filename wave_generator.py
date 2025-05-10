import os
import pandas as pd
from pydub import AudioSegment
from midi2audio import FluidSynth
from mido import MidiFile


def generateWave(midi_path, name, output_dir):
    path = os.path.join(output_dir, name)
    mid = MidiFile(midi_path)
    FluidSynth(
        '/usr/share/sounds/sf2/FluidR3_GM.sf2').midi_to_audio(midi_path, path)

    audio = AudioSegment.from_file(path, format="wav")
    audio = audio.set_frame_rate(12_800).set_channels(1).set_sample_width(2)
    audio.export(path, format="wav")


def getMessageDf(file_path):
    mid = MidiFile(file_path)
    # note_msg = filter(lambda y: y.type == 'note_on', mid.tracks[1])
    note_msg = map(lambda x: (x, x.time), mid.tracks[1])
    note_df = pd.DataFrame(note_msg, columns=['other', 'time'])

    note_df['time'] = note_df['time'].cumsum()
    note_df = note_df[note_df['other'].apply(lambda x: x.type == 'note_on')]
    note_df['note'] = note_df['other'].apply(lambda x: x.note)
    note_df['velocity'] = note_df['other'].apply(lambda x: x.velocity)
    note_df = note_df.drop(columns=['other']).reset_index(drop=True)
    return note_df


if __name__ == '__main__':
    root = 'maestro-v3.0.0'
    df = pd.read_csv(os.path.join(root, 'maestro-v3.0.0.csv'))
    df = df.sort_values(by=['duration'], ascending=True)
    df = df.iloc[:300]

    if not os.path.exists('waves'):
        os.makedirs('waves')

    file_rows = []

    for i, (idx, row) in enumerate(df.iterrows()):
        wav_name = f'{idx}.wav'
        midi_name = f'{idx}.midi'
        generateWave(os.path.join(root, row['midi_filename']), wav_name, 'waves')
        with open(os.path.join(root, row['midi_filename']), 'rb') as f1, open(os.path.join('waves', midi_name), 'wb') as f2:
            f2.write(f1.read())
        file_rows.append((wav_name, midi_name))
        print(f'{str(i).rjust(3)}. {str(idx).rjust(5)}    {row["midi_filename"]}')
    
    file_df = pd.DataFrame(file_rows, columns=['wav', 'midi'])
    file_df.to_csv('wav_midi.csv', index=False)