import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from datetime import datetime
import os
import logging
import argparse
import time
from pathlib import Path

# Set base directory for relative paths
BASE_DIR = Path(__file__).resolve().parent

class DrowsinessDetector:

    def __init__(self, ear_threshold=0.25, drowsy_time=2.0, predictor_path=None):
        """
        Initialize the drowsiness detector

        Args:
            ear_threshold (float): Eye aspect ratio threshold for drowsiness detection
            drowsy_time (float): Time in seconds before considering drowsy
            predictor_path (str): Path to dlib's facial landmark predictor
        """
        self.EAR_THRESHOLD = ear_threshold
        self.DROWSY_TIME = drowsy_time
        self.photo_captured = False
        self.sleep_start_time = None

        # Use default predictor path if none provided
        if predictor_path is None:
            predictor_path = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('drowsiness_detector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize detectors
        self._init_detectors(predictor_path)

        # Create output directory
        self.output_dir = BASE_DIR / "drowsy_photos"
        self.output_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def _init_detectors(self, predictor_path):
        """Initialize dlib detectors with error handling"""
        try:
            self.detector = dlib.get_frontal_face_detector()

            print("🔍 Looking for predictor at:", predictor_path)
            print("📂 Absolute path:", os.path.abspath(str(predictor_path)))

            if not os.path.exists(predictor_path):
                raise FileNotFoundError(f"Predictor file not found: {predictor_path}")

            self.predictor = dlib.shape_predictor(str(predictor_path))

            # Get eye landmark indices
            self.left_eye_indices = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            self.right_eye_indices = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            self.logger.info("Detectors initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize detectors: {e}")
            raise

    @staticmethod
    def calculate_ear(eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks

        Args:
            eye_landmarks: Array of eye landmark coordinates

        Returns:
            float: Eye aspect ratio
        """
        try:
            A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
            C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
            if C == 0:
                return 0.0
            return (A + B) / (2.0 * C)
        except Exception as e:
            logging.error(f"Error calculating EAR: {e}")
            return 0.0

    def process_frame(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            drowsiness_detected = False

            if len(faces) == 0:
                self.sleep_start_time = None
                self.photo_captured = False
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                face = faces[0]
                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[self.left_eye_indices[0]:self.left_eye_indices[1]]
                right_eye = shape[self.right_eye_indices[0]:self.right_eye_indices[1]]

                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < self.EAR_THRESHOLD:
                    if self.sleep_start_time is None:
                        self.sleep_start_time = datetime.now()
                        self.logger.info("Drowsiness detection started")
                    drowsy_duration = (datetime.now() - self.sleep_start_time).total_seconds()
                    if drowsy_duration >= self.DROWSY_TIME and not self.photo_captured:
                        self._capture_drowsy_photo(frame)
                        drowsiness_detected = True
                    cv2.putText(frame, f"DROWSY! Duration: {drowsy_duration:.1f}s", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if self.sleep_start_time is not None:
                        self.logger.info("Alert state restored")
                    self.sleep_start_time = None
                    self.photo_captured = False

                cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed_time = time.time() - self.start_time
                self.fps = self.frame_count / elapsed_time

            cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return frame, drowsiness_detected

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame, False

    def _capture_drowsy_photo(self, frame):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            photo_path = self.output_dir / f"drowsy_{timestamp}.jpg"
            success = cv2.imwrite(str(photo_path), frame)
            if success:
                self.logger.warning(f"DROWSINESS DETECTED! Photo saved: {photo_path}")
                self.photo_captured = True
            else:
                self.logger.error("Failed to save drowsy photo")
        except Exception as e:
            self.logger.error(f"Error capturing drowsy photo: {e}")

    def run(self, camera_id=0):
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise ValueError(f"Cannot open camera with ID: {camera_id}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            self.logger.info("Starting drowsiness detection system...")
            self.logger.info(f"EAR Threshold: {self.EAR_THRESHOLD}")
            self.logger.info(f"Drowsy Time: {self.DROWSY_TIME}s")
            self.logger.info("Press 'q' or ESC to quit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    break
                processed_frame, drowsiness_detected = self.process_frame(frame)
                cv2.imshow("Drowsiness Detection System", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self.sleep_start_time = None
                    self.photo_captured = False
                    self.logger.info("System reset by user")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_path = self.output_dir / f"manual_save_{timestamp}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    self.logger.info(f"Frame saved manually: {save_path}")

        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            total_time = time.time() - self.start_time
            self.logger.info(f"Session ended. Total frames: {self.frame_count}, "
                           f"Duration: {total_time:.1f}s, Avg FPS: {self.frame_count/total_time:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Drowsiness Detection System")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--ear-threshold", "-e", type=float, default=0.25,
                       help="Eye aspect ratio threshold (default: 0.25)")
    parser.add_argument("--drowsy-time", "-t", type=float, default=2.0,
                       help="Time in seconds before considering drowsy (default: 2.0)")
    parser.add_argument("--predictor", "-p", type=str, 
                       default=None,
                       help="Path to facial landmark predictor file")

    args = parser.parse_args()

    try:
        detector = DrowsinessDetector(
            ear_threshold=args.ear_threshold,
            drowsy_time=args.drowsy_time,
            predictor_path=args.predictor
        )
        detector.run(camera_id=args.camera)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
