#%%
import cv2
import os
import winsound

# Declare filenames
ROOT_DATA_DIR = (r"~\EyeTracking\Lib") # folder where the Input data sits
DATA_DIR_OUTPUT =(r"~\EyeTracking\Data\Semantic_00_Frames")


# Iterate through movie files in the input directory
for file_name in os.listdir(ROOT_DATA_DIR):
    
    if file_name.endswith(".mp4"):
        
        # Open the video file
        video_path = os.path.join(ROOT_DATA_DIR, file_name)
        cap = cv2.VideoCapture(video_path)

        # Get the framerate of the video
        framerate = cap.get(cv2.CAP_PROP_FPS)

        # Create the subfolder in the output directory for this movie file
        movie_folder = os.path.join(DATA_DIR_OUTPUT, os.path.splitext(file_name)[0])
        print(movie_folder)
        os.makedirs(movie_folder, exist_ok=True)

        # Set the counter for the frame number
        frame_count = 0

        # Loop through the video frames
        while cap.isOpened():
            # Read the next frame
            ret, frame = cap.read()

            # If we've reached the end of the video, stop looping
            if not ret:
                break

            # Save the frame to a file in the subfolder
            frame_filename = os.path.join(movie_folder, f"frame_{frame_count:04d}.jpg")
            print(frame_filename)
            #cv2.imwrite(frame_filename, frame)

            # Increment the frame counter
            frame_count += 1

            # Wait for the specified amount of time before processing the next frame
            # This will ensure that we capture frames at the desired framerate
            key = cv2.waitKey(int(1000/framerate))

        # Release the video capture object
        cap.release()

# Beep to indicate completion
winsound.MessageBeep()
# %%
