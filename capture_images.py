import cv2
import os
from datetime import datetime


def capture_and_save_images(camera_index=0, save_directory="calibration_images", image_prefix="calib_img"):
    """
    Capture and save images from a camera for calibration purposes.

    Args:
        camera_index (int): Camera index (0 for default camera)
        save_directory (str): Directory to save images
        image_prefix (str): Prefix for saved image filenames
    """

    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Initialize camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened successfully")
    print("Controls:")
    print("- Press 'SPACE' to capture and save an image")
    print("- Press 'q' or 'ESC' to quit")
    print("- Images will be saved to:", save_directory)

    img_count = 0

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Display the frame
            cv2.imshow('Camera Feed - Press SPACE to capture', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Spacebar
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{image_prefix}_{timestamp}.jpg"
                filepath = os.path.join(save_directory, filename)

                # Save the image
                success = cv2.imwrite(filepath, frame)

                if success:
                    img_count += 1
                    print(f"Saved image {img_count}: {filename}")
                else:
                    print(f"Failed to save image: {filename}")

            elif key == ord('q') or key == 27:  # 'q' or ESC
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone! Captured {img_count} images in '{save_directory}'")


# Alternative version with specific resolution
def capture_with_resolution(camera_index=0, width=1920, height=1080, save_directory="calibration_images"):
    """
    Capture images with specific resolution.
    """
    os.makedirs(save_directory, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print(f"Camera opened with resolution {width}x{height}")
    print("Controls:")
    print("- Press 'SPACE' to capture and save an image")
    print("- Press 'q' or 'ESC' to quit")

    img_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Display resolution info on frame
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Images saved: {img_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Camera Feed', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"calib_{timestamp}_{img_count:03d}.jpg"
                filepath = os.path.join(save_directory, filename)

                success = cv2.imwrite(filepath, frame)

                if success:
                    img_count += 1
                    print(f"Saved: {filename}")
                else:
                    print(f"Failed to save: {filename}")

            elif key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total images captured: {img_count}")


# Simple single image capture function
def capture_single_image(camera_index=0, filename="captured_image.jpg"):
    """
    Capture a single image and save it.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return False

    # Warm up the camera
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()

    if ret:
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"Image saved as: {filename}")
        else:
            print("Failed to save image")
    else:
        print("Failed to capture image")

    cap.release()
    return ret and success


if __name__ == "__main__":
    # Run the interactive capture tool
    print("Choose capture mode:")
    print("1. Interactive capture (press SPACE to save images)")
    print("2. Capture with specific resolution")
    print("3. Capture single image")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        capture_and_save_images()
    elif choice == "2":
        width = int(input("Enter width (default 1920): ") or 1920)
        height = int(input("Enter height (default 1080): ") or 1080)
        capture_with_resolution(0, width, height)
    elif choice == "3":
        filename = input("Enter filename (default: captured_image.jpg): ") or "captured_image.jpg"
        capture_single_image(0, filename)
    else:
        print("Invalid choice")