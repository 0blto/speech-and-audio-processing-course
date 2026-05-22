from pydub import AudioSegment
from pydub.playback import play

audio = AudioSegment.from_file(r"vits_vctk_A_20260424_175444_April_24_2026_05+54PM_cae8682_synthesis.wav")

faster_audio = audio._spawn(audio.raw_data, overrides={
    "frame_rate": int(audio.frame_rate * 2.2)
}).set_frame_rate(audio.frame_rate)

play(faster_audio)