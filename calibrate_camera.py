import cv2
import numpy as np
import glob

# Define the size of your checkerboard (number of inner corners)
CHECKERBOARD = (6, 9)  # Change this to match your checkerboard

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Read a set of calibration images
images = glob.glob('calibration_images/*.jpg')  # Change this path

# Store image dimensions
image_size = None

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"Warning: Could not read image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Store the image size from the first valid image
    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners (optional)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(500)
    else:
        print(f"Checkerboard not found in {fname}")

cv2.destroyAllWindows()

# Check if we have any valid calibration images
if len(objpoints) == 0:
    print("Error: No valid checkerboard images found!")
    exit()

if image_size is None:
    print("Error: Could not determine image size!")
    exit()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

# Print and save the calibration parameters
print("Camera matrix (K):")
print(camera_matrix)
print("\nDistortion coefficients (D):")
print(dist_coeffs)
print(f"\nSuccessfully used {len(objpoints)} images for calibration")

# Save the parameters for later use
np.savez('camera_calibration_params.npz',
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         image_size=image_size)

print("\nCalibration parameters saved to 'camera_calibration_params.npz'")