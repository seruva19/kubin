# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky4_video2audio/utils.py)
"""


import os
import uuid
from pathlib import Path

import numpy as np
import pydub
import torchvision
from PIL import Image
from moviepy.editor import AudioFileClip, ImageSequenceClip
from transformers import AutoModelForCausalLM, AutoTokenizer

# from IPython.display import Video, clear_output, display

from .riffusion.spectrogram_image_converter import SpectrogramImageConverter
from .riffusion.spectrogram_params import SpectrogramParams


def set_to_target_level(sound, target_level):
    difference = target_level - sound.dBFS
    return sound.apply_gain(difference)


def image_to_audio(pil_image, target_dBFS=-30.0, device="cuda"):
    """
    Reconstruct an audio clip from a spectrogram image.
    """
    # Get parameters from image exif
    img_exif = pil_image.getexif()
    assert img_exif is not None

    try:
        params = SpectrogramParams.from_exif(exif=img_exif)
    except (KeyError, AttributeError):
        print(
            "WARNING: Could not find spectrogram parameters in exif data. Using defaults."
        )
        params = SpectrogramParams()

    converter = SpectrogramImageConverter(params=params, device=device)
    segment = converter.audio_from_spectrogram_image(pil_image)

    if target_dBFS is not None:
        segment = set_to_target_level(segment, target_dBFS)

    save_path = f"./{uuid.uuid4()}.mp3"
    segment.export(save_path, format="mp3")

    return save_path


def create_video(
    spectogram,
    video,
    target_dBFS=-30.0,
    save_path=None,
    display_video=False,
    device="cuda",
):
    spectogram = Image.fromarray(np.array(spectogram))
    audio_temp_path = image_to_audio(spectogram, target_dBFS, device)
    try:
        audioclip = AudioFileClip(audio_temp_path)

        images_list = [video[i].numpy() for i in range(video.shape[0])]
        fps = video.shape[0] / audioclip.duration
        videoclip = ImageSequenceClip(images_list, fps=fps)

        video_with_new_audio = videoclip.set_audio(audioclip)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            video_with_new_audio.write_videofile(save_path)

        if display_video:
            video_with_new_audio.write_videofile("temp.mp4")
            # clear_output(wait=True)
            # display(Video("temp.mp4"))

        os.remove(audio_temp_path)
    except Exception as ex:
        os.remove(audio_temp_path)
        raise ex


def load_video(video, fps, num_frames=38, max_duration_sec=12):
    duration_sec = int(video.shape[0] / fps)
    if duration_sec > max_duration_sec:
        duration_sec = max_duration_sec
    total_frames = int(duration_sec * fps)

    video = video[:total_frames]
    frame_id_list = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    video_input = video[frame_id_list].permute(3, 0, 1, 2)
    return video_input, video, duration_sec
