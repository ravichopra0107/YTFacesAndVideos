# Importing libraries
import cv2
import imutils
from vidgear.gears import CamGear
from youtube_search import YoutubeSearch

# Important utility constants
DETECTION_TO_Q = {
    "bhuvanbam": "Bhuvan Bam",
    "carryminati": "Carry Minati",
    "Flyingbeast": "Flying Beast",
    "sauravjoshi": "Saurav Joshi",
    "technicalguruji": "Technical Guruji"
}


# Read image from webcam
img = cv2.imread("bb.jpg")
# By face detection
x_offset = 400
y_offset = 150
face_h = 200
face_w = 200
name = "bhuvanbam"

# get video id
results = YoutubeSearch(DETECTION_TO_Q[name], max_results=1).to_dict()
id = results[0]['id']

# Initialise stream
stream = CamGear(source=f'https://youtu.be/{id}', stream_mode=True,
                 logging=True).start()

# Show frames
while True:
    cv2.namedWindow('frame')
    # Read stream
    frame = stream.read()
    # Stream End
    if frame is None:
        break
    # frame resized to fit face
    frame = imutils.resize(frame, width=face_w+100)
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Overlay video frame and image
    img[y_offset:y_offset+frame.shape[0],
        x_offset:x_offset+frame.shape[1]] = frame
    # Callback on onClick

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if(y >= y_offset and y <= y_offset+frame_h and x >= x_offset and x <= x_offset+frame_w):
                while True:
                    f = stream.read()
                    if f is None:
                        break
                    f = imutils.resize(f, width=1200)
                    cv2.imshow("f", f)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        stream.stop()
                        break

    cv2.setMouseCallback('frame', on_mouse)
    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
stream.stop()
