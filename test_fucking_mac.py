import warnings
import os
import absl.logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging._warn = lambda msg: open("absl_logs.txt", "a").write(msg + "\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')
    from importingmodules import *
    # print("loading EYES...",end = '')
    # PATH = "."
    # #model = keras.models.load_model(PATH+'/EYESOFCAT3.h5')
    # model = keras.models.load_model(PATH+'/EYESOFCAT4.h5')
    print("done\n")

print("connecting cap...",end = '')
cap = cv2.VideoCapture(0)
print("done\n")

_IDX_ = [x for x in range(0,32)]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)

def create_dimage(h, w, d):
    image = np.zeros((h, w,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image

while cap.isOpened():
    success, image = cap.read()
    if success:
        print(image.shape)
        break

while cap.isOpened():
    success, video = cap.read()
    if not success :
        continue

    image = cv2.cvtColor(cv2.flip(video, 1), cv2.COLOR_BGR2RGB)
    bimage = create_dimage(video.shape[0], video.shape[1], 3)

    results = hands.process(image)
    if results.multi_hand_landmarks:
        hands_type_d = []

        for idx, hand_handedness in enumerate(results.multi_handedness):
            hands_type_d.append((hand_handedness.classification[0].label == "Right"))
        
        hands_type_d = hands_type_d[:2]
        
        idx = 0
        for hls in results.multi_hand_landmarks:
            idx_real = hands_type_d[idx]

            mp_drawing.draw_landmarks(bimage, hls, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(30,30,250) if idx_real else (250,30,30), thickness=2, circle_radius=2))
            idx = 1

    cv2.imshow('CAT', bimage) 
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

cap.release()
