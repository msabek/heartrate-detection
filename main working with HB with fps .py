import cv2
import pyramids
import heartrate
import preprocessing
import eulerian

# Frequency range for Fast-Fourier Transform
freq_min = 1
freq_max = 1.8

# Preprocessing phase
print("Reading + preprocessing video...")
video_frames, frame_ct, fps = preprocessing.read_video("videos/mohamed.mp4")

# Build Laplacian video pyramid
print("Building Laplacian video pyramid...")
lap_video = pyramids.build_video_pyramid(video_frames)

amplified_video_pyramid = []

for i, video in enumerate(lap_video):
    if i == 0 or i == len(lap_video)-1:
        continue

    # Eulerian magnification with temporal FFT filtering
    print("Running FFT and Eulerian magnification...")
    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    lap_video[i] += result

    # Calculate heart rate
    print("Calculating heart rate...")
    heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)

# Collapse laplacian pyramid to generate final video
print("Rebuilding final video...")
amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

# Output heart rate and final video
print("Heart rate: ", heart_rate, "bpm")
print("Displaying final video...")

######################################

# Assuming heart_rate has been calculated correctly before this code block
heart_rate_display = "Heart rate: {:.2f} bpm".format(heart_rate)

for frame in amplified_frames:
    # Choose a corner for the text with a margin from the top left corner
    org = (10, 30)  # (x, y) coordinates of the bottom-left corner of the text string in the image
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 1  # Font scale
    color = (0, 255, 0)  # Color of the text, green in this case
    thickness = 2  # Thickness of the lines used to draw the text

    # Put the text on each frame
    cv2.putText(frame, heart_rate_display, org, font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # Press 'q' to exit the loop/display
        break

# When everything is done, release the window
cv2.destroyAllWindows()



