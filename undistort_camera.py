import numpy as np
import cv2
from lerobot.cameras.opencv import OpenCVCamera


class UndistortedOpenCVCamera(OpenCVCamera):
    """
    A wrapper for OpenCVCamera that undistorts frames using pre-computed
    calibration parameters.
    """

    def __init__(self, config, calib_file_path='camera_calibration_params.npz'):
        super().__init__(config)

        # Load calibration parameters
        calib_data = np.load(calib_file_path)
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeffs']

        # Use image size from calibration or config
        if 'image_size' in calib_data:
            self.image_size = tuple(calib_data['image_size'])
        else:
            self.image_size = (config.width, config.height)

        # Pre-compute undistortion maps for better performance
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            self.image_size,
            cv2.CV_32FC1
        )

        # Store the original read method for later use
        self.original_read = super().read

    def read(self):
        """
        Override the read method to return undistorted frames.
        """
        # Get the original frame from parent class
        frame = self.original_read()

        # Handle different frame types
        if isinstance(frame, np.ndarray):
            # Frame is directly the image array
            return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        elif hasattr(frame, 'rgb') and frame.rgb is not None:
            # Frame has an rgb attribute
            frame.rgb = cv2.remap(frame.rgb, self.map1, self.map2, cv2.INTER_LINEAR)
            return frame
        elif hasattr(frame, 'image') and frame.image is not None:
            # Frame has an image attribute
            frame.image = cv2.remap(frame.image, self.map1, self.map2, cv2.INTER_LINEAR)
            return frame
        else:
            # Unknown frame format, return as-is
            print(f"Warning: Unknown frame format: {type(frame)}")
            return frame


# Usage example
if __name__ == "__main__":
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import ColorMode, Cv2Rotation

    config = OpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )

    camera = UndistortedOpenCVCamera(config, 'camera_calibration_params.npz')

    try:
        camera.connect()

        for i in range(10):
            frame = camera.read()
            print(f"Frame {i} type: {type(frame)}")

            if isinstance(frame, np.ndarray):
                print(f"Frame shape: {frame.shape}")
                cv2.imshow('Undistorted Feed', frame)
            elif hasattr(frame, 'rgb') and frame.rgb is not None:
                print(f"Frame.rgb shape: {frame.rgb.shape}")
                cv2.imshow('Undistorted Feed', frame.rgb)
            elif hasattr(frame, 'image') and frame.image is not None:
                print(f"Frame.image shape: {frame.image.shape}")
                cv2.imshow('Undistorted Feed', frame.image)
            else:
                print("Unknown frame format, cannot display")

            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        camera.disconnect()
        cv2.destroyAllWindows()