import cv2
import numpy as np
import mediapipe as mp

def extract_frames(video_path, max_frames=45):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    step = max(1, frame_count // max_frames)
    
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def extract_joint_coordinates(frames):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    all_coordinates = []
    
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            frame_coordinates = []
            for landmark in results.pose_landmarks.landmark:
                frame_coordinates.append((landmark.x, landmark.y, landmark.z))
            all_coordinates.append(frame_coordinates)
    
    return all_coordinates

video_path = r"C:\Users\dell\OneDrive\Desktop\hackathon\Greetings\49. How are you\MVI_0033.MOV"
frames = extract_frames(video_path)

coordinates = extract_joint_coordinates(frames)

num_joints = 33  # Number of joints detected by MediaPipe

# Initialize the numpy array to store coordinates
coordinates_array = np.zeros((len(frames), num_joints, 3))

for i, frame_coordinates in enumerate(coordinates):
    for j, joint in enumerate(frame_coordinates):
        coordinates_array[i, j] = joint
print(len(coordinates_array[0]))


# storing the np array (coordinates_array into a file ):


import numpy as np
import h5py

# Save the array
with h5py.File('coordinates.h5', 'w') as f:
    f.create_dataset('coordinates_array', data=coordinates_array)

# Load the array
with h5py.File('coordinates.h5', 'r') as f:
    large_array = f['coordinates_array'][:]

