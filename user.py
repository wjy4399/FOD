import os
import cv2
from cv2 import VideoCapture, imencode
from PIL import Image
import time
from gradio_client import Client, handle_file
from datetime import datetime
import pygame
class WebcamStream:
    def __init__(self):
        self.stream = None
        self.frame = None
        self.running = False

    def start(self):
        if self.running:
            return self

        self.running = True
        self.stream = VideoCapture(find_camera_index())
        _, self.frame = self.stream.read()
        return self

    def read(self, encode=False):
        if self.stream is None:
            return None

        _, frame = self.stream.read()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer).decode("utf-8")

        return frame

    def stop(self):
        if self.running:
            self.running = False
            if self.stream:
                self.stream.release()

    @staticmethod
    def _save_image_locally(image):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not os.path.exists("./images"):
            os.makedirs("./images")
        file_name = f"./images/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        img.save(file_name)
        return file_name

def find_camera_index():
    max_index_to_check = 10  # Maximum index to check for camera

    for index in range(max_index_to_check):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            return index

    # If no camera is found
    raise ValueError("No camera found.")

def capture_and_send(client):
    webcam_stream = WebcamStream().start()

    try:
        while True:
            # Capture frame-by-frame
            frame = webcam_stream.read()
            if frame is None:
                print("Failed to capture image")
                break

            # Save frame as an image file
            image_path = WebcamStream._save_image_locally(frame)

            # Send image to server and get result
            try:
                result = client.predict(
                    image=handle_file(image_path),
                    api_name="/predict"
                )
                if result=='No detection results.':
                    print(f"Results: {result}")
                    continue
                else:
                    pygame.mixer.init()
                    pygame.mixer.music.load('hello.mp3')
                    pygame.mixer.music.play()
                    print(f"Results: {result}")
            except Exception as e:
                print(f"Error processing image: {e}")

            # Wait for 1 second
            time.sleep(1)

    finally:
        # When everything done, release the capture
        webcam_stream.stop()

if __name__ == "__main__":
    client = Client("https://05f4372e8de7e0cb92.gradio.live/")
    capture_and_send(client)
