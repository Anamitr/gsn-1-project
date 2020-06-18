import os

from presentation_controller import PresentationController
from utils import detector_utils as detector_utils
from train_classifier import define_three_block_model, INPUT_SHAPE, predict_hand_capture_gesture
import cv2
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import datetime
import argparse
import time
import numpy as np

# tf.compat.v1.disable_v2_behavior()
tf.disable_v2_behavior()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

detection_graph, sess = detector_utils.load_inference_graph()


def remove_background(bgModel, frame):
    fgmask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    bgSubThreshold = 50

    is_background_captured = 0
    frame_divisor = 10
    frame_counter = 0

    photo_counter = 302
    should_take_photo = False

    model = define_three_block_model()
    model.load_weights('trained_classifier_1/trained_classifier_1')

    presentation_controller = PresentationController()
    is_presenter_controller_started = False

    while True:
        ret, image_np = cap.read()
        cv2.imshow('original', image_np)

        pressed_key = cv2.waitKey(1)
        if pressed_key == 27:  # press ESC to exit all windows at any time
            break
        elif pressed_key == ord('b'):
            background_subtractor = cv2.createBackgroundSubtractorMOG2(0, 50)
            time.sleep(2)
            is_background_captured = 1
            print("Background captured")
        elif pressed_key == ord(' '):  # photo capture
            should_take_photo = True
        elif pressed_key == ord('p'):
            print('Starting presenter mode')
            is_presenter_controller_started = not is_presenter_controller_started
            time.sleep(2)

        if is_background_captured == 1 and frame_counter % frame_divisor == 0:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

            # image_np = cv2.flip(image_np, 1)
            try:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(image_np,
                                                          detection_graph, sess)

            # image_np = background_subtractor.apply(image_np)
            image_np = remove_background(background_subtractor, image_np)

            # boxes = boxes[:0]
            # scores = scores[:0]

            # draw bounding boxes on frame
            # image_np = detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
            #                                  scores, boxes, im_width, im_height,
            #                                  image_np)

            # cut bounding box with hand from frame
            image_np = detector_utils.cut_hand_bounding_box_from_image(num_hands_detect, args.score_thresh,
                                                                       scores, boxes, im_width, im_height,
                                                                       image_np)
            if image_np is not None:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            if should_take_photo and image_np is not None:
                status = cv2.imwrite(os.path.join("db/volume_down/", str(photo_counter) + ".jpg"), image_np)
                print("Photo", photo_counter, "taken:", status)
                photo_counter += 1
                should_take_photo = False

            # hand_gesture_classifier.classify_baw_frame(image_np)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            if (args.display > 0):
                # Display FPS on frame
                # if (args.fps > 0):
                #     detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                #                                      image_np)

                try:
                    if image_np is not None:
                        cv2.imshow('Single-Threaded Detection',
                                   image_np)
                        predicted_hand_gesture = predict_hand_capture_gesture(model, image_np)
                        # print(predicted_hand_gesture, end=',')
                        if is_presenter_controller_started:
                            presentation_controller.add_gesture(predicted_hand_gesture)
                            if predicted_hand_gesture == 'pointer':
                                # center_pos = detector_utils.get_center_pos_of_bounding_box(boxes, im_width, im_height)
                                # print("Center pos:", center_pos)
                                # relative_center_pos = (center_pos[0] / im_width, center_pos[1] / im_height)
                                # presentation_controller.set_current_relative_pointer_pos(relative_center_pos)
                                pos = detector_utils.get_left_and_top_pos_of_first_bounding_box(boxes, im_width, im_height)
                                relative_pos = (pos[0] / im_width, pos[1] / im_height)
                                print(relative_pos)
                                presentation_controller.set_current_relative_pointer_pos(relative_pos)

                except Exception:
                    print("cv2.imshow failed")

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                print("frames processed: ", num_frames, "elapsed time: ",
                      elapsed_time, "fps: ", str(int(fps)))
        frame_counter += 1

# if __name__ == '__main__':
#     camera = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')
#     while True:
#         (grabbed, frame) = camera.read()
#         cv2.imshow('Photo', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()
