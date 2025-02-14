import os
import numpy as np
import wave

def data_loader(wavDir = "dataset/wav", annotations = "dataset/annotations.txt"):
    signal0, signal1 = [], []
    label0, label1 = [], []
    
    with open(annotations, "r") as file:
        for line in file:
            audioInfo = line.strip().split()
         
            file_path = os.path.join(wavDir, audioInfo[0]+'.wav')
            signal = read_wav_file(file_path)
    
            if audioInfo[-1] == '0':
                signal0.append(signal)
                label0.append(audioInfo[-1])
            else:
                signal1.append(signal)
                label1.append(audioInfo[-1])
    
    signals = signal0[:100]
    signals.extend(signal1[:100])

    labels = label0[:100]
    labels.extend(label1[:100])

    signals = np.array(signals, dtype=object)  
    labels = np.array(labels)

    return signals, labels

def read_wav_file(file_path):
    with wave.open(file_path, "r") as spf:
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, dtype=np.int16)
    return signal

if __name__ == "__main__":
    data_loader()