#!/bin/bash

# Grabs all .mp4 videos in dir and split it into frames according to $FPS using ffmpeg version n7.0.1
# Put the frames into subdir called <original video name>_frames
# Must specify number of digits ($DIGIT_NUMBER) the files should have per video (eg. 1000 frames should have 4 digits)
DIGIT_NUMBER=15
FPS=24

for input in *.mp4; do
  if [ -f "$input" ]; then
    no_extension_name=$(basename "$input" .mp4)
    output_dir="${no_extension_name}_frames"
    output_filename="${output_dir}/${no_extension_name}_%${DIGIT_NUMBER}d.png"

    mkdir -p "$output_dir"
    ffmpeg -i "$input" -vf "fps=${FPS}" "$output_filename"
  fi
done
