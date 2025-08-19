from pathlib import Path

# Get the current working directory
# current_directory = Path(r'/home/luis/Dropbox/DATASETS_AUDIO/Dvectors/TTS4_easy_40-200/input_LONG_wavs')
current_directory = Path(r'/home/luis/Dropbox/DATASETS_AUDIO/Dvectors/TTS4_easy_40-200/input_wavs')

# Iterate over all the WAV files in the current directory
for file_path in current_directory.glob('*.wav'):
    # new_stem = file_path.stem.replace('preComp', '')
    new_stem = file_path.stem.replace('D1_', 'D')
    new_file_path = current_directory / (new_stem + file_path.suffix)
    
    # Rename the file
    file_path.rename(new_file_path)
    print(f'Renamed "{file_path.name}" to "{new_file_path.name}"')


print('File renaming completed.')