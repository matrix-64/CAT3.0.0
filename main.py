print("importing keras...",end = '')
import keras
print("done")
print("importing modules...",end = '')
import numpy as np
import cv2
import mediapipe as mp
import pyautogui as pg
import time
print("done")
print("waking mmcat up...",end = '')
from mmcat import mouseModeCAT
print("done")
print("waking kmcat up...",end = '')
from kmcat import keyboardModeCAT
print("done")
print()

print("loading EYES...",end = '')
PATH = "C:/Users/JOOWAN/Desktop/jonghab/gifted/korea/projects/re_CAT_dataset"
#model = keras.models.load_model(PATH+'/EYESOFCAT3.h5')
model = keras.models.load_model(PATH+'/EYESOFCAT4.h5')
IDX = [x for x in range(0,32)]
print("done")
print()

pg.PAUSE = 0
#pg.FAILSAFE = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def create_dimage(h, w, d):
    image = np.zeros((h, w,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image

def create_vector(point1, point2):
    v = np.array([point2.x-point1.x,point2.y-point1.y,point2.z-point1.z])
    return v / np.linalg.norm(v)

def rotate_array(arr):
    B1 = np.array([[arr[1][5],-arr[0][5],0,0],[arr[0][5],arr[1][5],0,0],[0,0,1,0],[0,0,0,1]])
    M = [[],[],[]]
    for i in range(6):
        m = np.array([arr[0][i],arr[1][i],arr[2][i],1])
        r = B1 @ m
        M[0].append(r[0])
        M[1].append(r[1])
        M[2].append(r[2])
    M = np.array(M)
                        
    B2 = np.array([[1,0,0,0],[0,M[2][5],-M[1][5],0],[0,M[1][5],M[2][5],0],[0,0,0,1]])
    R = [[],[],[]]
    for i in range(6):
        m = np.array([M[0][i],M[1][i],M[2][i],1])
        r = B2 @ m
        R[0].append(r[0])
        R[1].append(r[1])
        R[2].append(r[2])

    return R

class CAT:
    hands = mp_hands.Hands(False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    k_img = cv2.imread("KEYBOARD.png",cv2.IMREAD_COLOR)
    p_img = cv2.imread("PAD.png",cv2.IMREAD_COLOR)

    def __init__(self,i_shape):
        self.cus_bef = [-1,-1]
        self.cus_cur = [-1,-1]
        self.finger = [[[0,0],[0,0],[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0],[0,0],[0,0]]]
        self.shape = [0,0]
        self.stdp = [[-1,-1],[-1,-1]]
        self.image_shape = i_shape
        self.dimage = create_dimage(self.image_shape[0],self.image_shape[1],self.image_shape[2])
        self.Bimage = self.dimage.copy()
        self.mode = "sleeping"   #3가지 모드
        # mode 1 : "mouse" - 초기 CAT의 기능인 마우스 모드
        # mode 2 : "keyboard" - 왼손과 오른손의 조합으로 키를 입력하는 모드
        # mode 0 : "sleeping" - 대기 상태, 모드 변경 기능
        #                       모든 모드에서 오른손과 왼손을 모두 주먹쥐면 이 모드로 변경
        #                       모드 변경을 위해 반드시 거쳐야 하는 통로
        self.mmcat = mouseModeCAT()
        self.kmcat = keyboardModeCAT()

        self.k_img = cv2.resize(CAT.k_img, dsize = (self.kmcat.w,self.kmcat.h), interpolation=cv2.INTER_AREA)
        self.p_img = cv2.resize(CAT.p_img, dsize = (self.mmcat.w,self.mmcat.h), interpolation=cv2.INTER_LINEAR)

    def action(self,fings):
        if self.shape == [31,31]:
            self.mode = "sleeping"

        if self.mode == "mouse":
            #마우스 패드 영역 표시
            h, w = self.mmcat.h, self.mmcat.w
            self.p_img = cv2.resize(CAT.p_img, dsize = (w,h), interpolation=cv2.INTER_LINEAR)

            h_c = self.mmcat.h_c
            w_c = self.mmcat.w_c
            mid = self.Bimage[h_c:h_c+h, w_c:w_c+w]

            mid = cv2.addWeighted(mid, 0.7, self.p_img, 0.3, 0)

            self.Bimage[h_c:h_c+h, w_c:w_c+w] = mid
            
            #mmcat에 전달할 값 리스트 생성
            a = int(fings[0][1][0]*(self.image_shape[1]/100))
            b = int(fings[1][1][0]*(self.image_shape[1]/100))
            c = int(fings[0][1][1]*(self.image_shape[0]/100))
            d = int(fings[1][1][1]*(self.image_shape[0]/100))
            pfm = ((b,d,a,c), self.stdp[1],self.image_shape)
            
            self.mmcat.action(self.shape,pfm)
            
            self.Bimage = cv2.circle(self.Bimage, (w_c,h_c), self.mmcat.rarr[0], (120,80,120), -1)
            self.Bimage = cv2.circle(self.Bimage, (w_c+w,h_c+h), self.mmcat.rarr[1], (120,80,120), -1)

        if self.mode == "keyboard":
            #배경에 키보드 보이기
            h, w = self.kmcat.h, self.kmcat.w
            self.k_img = cv2.resize(CAT.k_img, dsize = (w,h), interpolation=cv2.INTER_AREA)
            
            h_c = self.kmcat.h_c
            w_c = self.kmcat.w_c
            roi = self.Bimage[h_c:h_c+h, w_c:w_c+w]

            mask = cv2.cvtColor(self.k_img, cv2.COLOR_BGR2GRAY)
            mask[mask[:]==255]=0
            mask[mask[:]>0]=255

            mask_inv = cv2.bitwise_not(mask)
            roi_fg = cv2.bitwise_and(self.k_img, self.k_img, mask=mask)
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            dst = cv2.add(roi_fg, roi_bg)

            self.Bimage[h_c:h_c+h, w_c:w_c+w] = dst
            
            #키보드 액션
            #kmcat에 전달할 값 리스트 생성
            a = int(fings[0][1][0]*(self.image_shape[1]/100))
            b = int(fings[1][1][0]*(self.image_shape[1]/100))
            c = int(fings[0][1][1]*(self.image_shape[0]/100))
            d = int(fings[1][1][1]*(self.image_shape[0]/100))

            pfk = ((a,c),(b,d),self.image_shape)

            self.kmcat.action(self.shape,pfk)
            self.Bimage = cv2.circle(self.Bimage, (w_c,h_c), self.kmcat.rarr[0], (255,0,255), -1)
            self.Bimage = cv2.circle(self.Bimage, (w_c+w,h_c+h), self.kmcat.rarr[1], (255,0,255), -1)
            self.Bimage = cv2.circle(self.Bimage, (a,c), 5, (0,255,0), -1)
            self.Bimage = cv2.circle(self.Bimage, (b,d), 5, (0,255,0), -1)
            
        if self.mode == "sleeping":
            if self.shape == [29,29]:
                self.mode = "mouse"
            if self.shape == [25,25]:
                self.mode = "keyboard"
                
    def operate(self,video):
        self.Bimage = self.dimage.copy()
        
        image = cv2.cvtColor(cv2.flip(video, 1), cv2.COLOR_BGR2RGB)
        results = CAT.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hands_type_d = []
            for idx, hand_handedness in enumerate(results.multi_handedness):
                hands_type_d.append((hand_handedness.classification[0].label == "Right"))
            hands_type_d = hands_type_d[:2]

            self.shape = [0,0]
            is_folded = [[],[]]
            idx = 0
            for hls in results.multi_hand_landmarks:
                idx_real = hands_type_d[idx]
                
                self.finger[idx_real] = [(hls.landmark[4].x * 100, hls.landmark[4].y * 100),
                          (hls.landmark[8].x * 100, hls.landmark[8].y * 100),
                          (hls.landmark[12].x * 100, hls.landmark[12].y * 100),
                          (hls.landmark[16].x * 100, hls.landmark[16].y * 100),
                          (hls.landmark[20].x * 100, hls.landmark[20].y * 100)]

                a_x = hls.landmark[0].x
                a_y = hls.landmark[0].y
                
                v1 = create_vector(hls.landmark[mp_hands.HandLandmark(4).value],hls.landmark[mp_hands.HandLandmark(2).value])
                v2 = create_vector(hls.landmark[mp_hands.HandLandmark(8).value],hls.landmark[mp_hands.HandLandmark(6).value])
                v3 = create_vector(hls.landmark[mp_hands.HandLandmark(12).value],hls.landmark[mp_hands.HandLandmark(10).value])
                v4 = create_vector(hls.landmark[mp_hands.HandLandmark(16).value],hls.landmark[mp_hands.HandLandmark(14).value])
                v5 = create_vector(hls.landmark[mp_hands.HandLandmark(20).value],hls.landmark[mp_hands.HandLandmark(18).value])
                v6 = create_vector(hls.landmark[mp_hands.HandLandmark(9).value],hls.landmark[mp_hands.HandLandmark(0).value])

                arr = np.array([v1,v2,v3,v4,v5,v6])
                arr = np.transpose(arr)
                
                if idx_real == 0:
                    for j in range(0,6):
                        arr[0][j] = - arr[0][j]
                        
                R = rotate_array(arr)
                T = R[0][:5]+R[1][:5]+R[2][:5]
                self.shape[idx_real] = IDX[np.argmax(model.predict([T],verbose = None), axis=1)[0]]

                self.stdp[idx_real] = (a_x * 100, a_y * 100)

                mp_drawing.draw_landmarks(self.Bimage, hls, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(30,30,250) if idx_real else (250,30,30), thickness=2, circle_radius=2))
                idx = 1
                
            cv2.putText(self.Bimage, text='mode : %s  shape:(%d,%d) %d' % (self.mode,self.shape[0],self.shape[1],self.mmcat.in_pad),
                        org=(10, 30),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255,255,255), thickness=2)
            
        self.action(self.finger)
        self.Bimage = cv2.pyrUp(self.Bimage)
        cv2.imshow('CAT', self.Bimage)

    def stop(self):
        cv2.destroyAllWindows()
        del self

    def __del__(self):
        print("Meow!")

#############################################################################################################################################

print("connecting cap...",end = '')
cap = cv2.VideoCapture(0)
print("done")
print()

global cat
print("CAT woke up")

while cap.isOpened():
    success, image = cap.read()
    if success:
        cat = CAT(image.shape)
        break

while cap.isOpened():
    success, video = cap.read()
    if not success :
        continue
    cat.operate(video)
    if cv2.waitKey(1) == 27 and cat.mode == "sleeping":
        cat.stop()
        break

cap.release()
del(cat)


time.sleep(2)
