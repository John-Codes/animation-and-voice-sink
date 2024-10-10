from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file, wav_file):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file)
    
    # Export as WAV
    audio.export(wav_file, format="wav")
    
    print(f"Converted {mp3_file} to {wav_file}")

# Example usage
mp3_file = "A.mp3"
wav_file = "A.wav"

convert_mp3_to_wav(mp3_file, wav_file)