## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 RealSense, Inc. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
"""
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        h, w = depth_image.shape

        clipped = np.clip(depth_image, 300, 600)
        depth_image = ((clipped - 300) * 0.85).astype(np.uint8)
        print(round(depth_image[h // 2, w //2] / 0.85 + 300))

        #depth_image = depth_image.astype(np.uint8)
        dist = depth_frame.get_distance(w // 2, h //2) * 100
        print(f"{dist:.1f}")


        depth_image[239: 241, 310: 330] = 255
        depth_image[230: 250, 319: 321] = 255
        #depth_image[start_y:start_y + small_arr.shape[0], start_x:start_x + small_arr.shape[1]] = small_arr


        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# 1. Настройка конвейера (Pipeline)
pipeline = rs.pipeline()
config = rs.config()

# Рекомендуется использовать одинаковое разрешение для обоих потоков
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# 2. Создание объекта выравнивания
# Указываем целевой поток, к которому подгоняем второй (в данном случае к COLOR)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Получаем кадры
        frames = pipeline.wait_for_frames()

        # Согласуем кадры (магия происходит здесь)
        aligned_frames = align.process(frames)

        # Берем выровненные кадры
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Проверяем, что оба кадра получены
        if not aligned_depth_frame or not color_frame:
            continue

        # Конвертируем в массивы NumPy для OpenCV
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Опционально: делаем карту глубины цветной для красивой визуализации
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Отображаем результат (теперь пиксели строго совпадают!)
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense Aligned (Color | Depth)', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

