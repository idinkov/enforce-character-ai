# TO Test

Providers handle video files:
-> Allow providers to import video files and extract frames as images. This will enable users to work with video content more effectively.
-> Default frame extraction should be done by making scene detection using PySceneDetect or similar library to avoid extracting too many similar frames.
-> We will save scene_data as .yaml inside the folder where the provider video file is located. This will allow us to re-use the scene detection data if the user re-imports the same video file again.
-> From each scene we will extract a configurable number of frames (default 1 frame per scene, in the middle of the scene) to ensure we capture the most relevant content.
-> Name the extracted frames using the original video filename and scene number (e.g., video_filename_scene_002.jpg) to avoid naming conflicts and provide context.
-> Ensure that extracted frames are properly tagged and managed within the existing image management system.
-> Go through all the providers and add support for video files where it makes sense (e.g., Local Folder, URLs, etc.).
-> Update the requirements.txt to include any new dependencies needed for video processing.
-> You are already in the work dir so you can test directly calling "python" without making cd, and dont use &&, we are in windows so use ; if needed when you execute commands.