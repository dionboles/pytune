import numpy as np
from librosa import load
import librosa
import soundfile as sf 
import psola
import scipy.signal as sig
from scipy.io.wavfile import write
import sounddevice as sd
def correct(f0):
    if np.isnan(f0):
        return np.nan
    
    c_minor_degress = librosa.key_to_degrees("C:min")
    c_minor_degress = np.concatenate((c_minor_degress, [c_minor_degress[0]+ 12]))
    midi_note = librosa.hz_to_midi(f0)
    degress = midi_note % 12
    closet_degree_id = np.argmin(np.abs(c_minor_degress - degress))
    degress_difference = degress - c_minor_degress[closet_degree_id]
    midi_note -= degress_difference
    return librosa.midi_to_hz(midi_note)
def correct_pitch(f0):
    corrected_f0 = np.zeros_like(f0)
    for i in range(f0.shape[0]):
        corrected_f0[i] = correct(f0[i])
    smoothed_correced_f0 = sig.medfilt(corrected_f0,kernel_size=11)
    smoothed_correced_f0[np.isnan(smoothed_correced_f0)] = corrected_f0[np.isnan(smoothed_correced_f0)]

    return smoothed_correced_f0
def autotune(y,sr):
    frame_length = 2048
    hop_length = frame_length // 4

    fmin = librosa.note_to_hz("c2") 
    fmax = librosa.note_to_hz("c7")
    f0, _, _ = librosa.pyin(y,frame_length=frame_length,
                 hop_length=hop_length,fmax=fmax,fmin=fmin)
    correct_pitch_f0 = correct_pitch(f0)
    return psola.vocode(y,sample_rate=int(sr), target_pitch=correct_pitch_f0,fmin=fmin,fmax=fmax)
def main(): 

    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    print("rec")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    sd.Stream
    filename = 'output.wav'
    write(filename, fs, myrecording)  # Save as WAV file 
    
    y, sr = load(filename)
    if y.ndim > 1:
        y = y[0,:]

    pitch_corected_y = autotune(y,sr)
    sf.write("test.wav",pitch_corected_y,sr)

        


if __name__ == "__main__":
    main()