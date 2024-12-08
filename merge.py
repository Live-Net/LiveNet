from moviepy.editor import *
import os

L = []

for root, dirs, files in os.walk("animations/doorway_scenario_suite_5"):
    files.sort()
    for file in files:
        if '1_faster' in file:
            continue
        if os.path.splitext(file)[1] == '.mp4':
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("merged2.mp4", fps=24, remove_temp=False)