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

# Assuming that preprocessing.read_video returns frames with consistent dimensions
height, width = video_frames[0].shape[:2]

# Build Laplacian video pyramid
print("Building Laplacian video pyramid...")
lap_video = pyramids.build_video_pyramid(video_frames)

amplified_video_pyramid = []

# Placeholder for heart rate - to be computed within the loop
heart_rate = None

for i, video in enumerate(lap_video):
    if i == 0 or i == len(lap_video) - 1:
        continue

    # Eulerian magnification with temporal FFT filtering
    print("Running FFT and Eulerian magnification...")
    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    lap_video[i] += result

    # Calculate heart rate
    print("Calculating heart rate...")
    heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)

# Collapse Laplacian pyramid to generate final video
print("Rebuilding final video...")
amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
out_filename = 'output_video.avi'  # AVI format

out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

# Assuming heart_rate has been calculated correctly before this code block
heart_rate_display = "Heart rate: {:.2f} bpm".format(heart_rate) if heart_rate else "Heart rate: N/A"

for frame in amplified_frames:
    # Choose a corner for the text with a margin from the top left corner
    org = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2

    # Put the text on each frame
    cv2.putText(frame, heart_rate_display, org, font, font_scale, color, thickness, cv2.LINE_AA)

    # Write the frame into the file
    out.write(frame)

    # Display the frame
    cv2.imshow("Amplified Frame", frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):  # waitKey duration is adjusted for the video's fps
        break

# Release everything when the job is finished
out.release()
cv2.destroyAllWindows()