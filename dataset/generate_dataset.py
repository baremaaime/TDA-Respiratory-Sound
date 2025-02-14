import os

import kagglehub
from pydub import AudioSegment

def generate_dataset():

    # Download latest version
    path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
    
    print("Path to dataset files:", path)

    path = os.path.join(path,
                        r"Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files")

    fileNames = [f[:-4] for f in os.listdir(path) if f.endswith('.wav')]

    wavDir = "wav"
    os.makedirs(wavDir, exist_ok=True)

    for fileName in fileNames:
        audio = AudioSegment.from_wav(os.path.join(path, fileName + ".wav"))
        
        textFile = os.path.join(path, fileName + ".txt")
        with open(textFile, "r") as file:
            for i, line in enumerate(file):
                audioInfo = line.strip().split() 
    
                if audioInfo[2] == '1':
                    continue
        
                startTime = float(audioInfo[0]) * 1e3 
                endTime = float(audioInfo[1]) * 1e3  
                
                slicedAudio = audio[startTime:endTime]
    
                slicedAudio.export(wavDir + '/' + fileName + f"_{i}.wav", format="wav")
    
                with open("annotations.txt", "a") as file:  
                    file.write(f"{fileName}_{i}\t{audioInfo[-1]}\n")

if __name__ == "__main__":
    generate_dataset()