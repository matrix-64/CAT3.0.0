import math
from pynput.mouse import Button, Controller
import pyautogui as pg

def distance(x1, y1, x2, y2):
    result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result

class RemoteMouse:
    def __init__(self):
        self.mouse = Controller()

    def getPosition(self):
        return self.mouse.position

    def setPos(self, xPos, yPos):
        self.mouse.position = (xPos, yPos)
    def movePos(self, xPos, yPos):
        self.mouse.move(xPos, yPos)

    def click(self):
        self.mouse.click(Button.left)
    def doubleClick(self):
        self.mouse.click(Button.left, 2)
    def clickRight(self):
        self.mouse.click(Button.right)
    
    def down(self):
        self.mouse.press(Button.left)
    def up(self):
        self.mouse.release(Button.left)
        
    def down_right(self):
        self.mouse.press(Button.right)
    def up_right(self):
        self.mouse.release(Button.right)

class mouseModeCAT:
    SENSE = [0,1,2,4,6]
    X, Y = pg.size()
    def __init__(self):
        self.Rclicking = False
        self.Dclicking = False
        self.mouse = RemoteMouse()
        self.sense = 0 #왼손 숫자로 감도 조절(방식 : 카운트식)
        self.cus_bef = [-1,-1]
        self.cus_cur = [-1,-1]
        self.cus_dif = [0,0]
        self.R_zero_bef = [-1,-1]
        self.in_pad = False
        self.rarr = [5,5]

        #패드 영역 관련 변수
        self.h = 200
        self.w = 272
        self.h_c = 64
        self.w_c = 320

    def cal_mov(self,pfd):
        if pfd[0] >= self.w_c and pfd[0] <= self.w_c + self.w :
            if pfd[1] >= self.h_c and pfd[1] <= self.h_c + self.h :
                self.cus_dif = [self.cus_cur[0]-self.cus_bef[0], self.cus_cur[1]-self.cus_bef[1]]
                if not self.in_pad :
                    self.in_pad = True
                    return (0,0)
                else :
                    dx = (self.cus_dif[0]/self.w)*mouseModeCAT.X
                    dy = (self.cus_dif[1]/self.h)*mouseModeCAT.Y
                    return (dx,dy)
        self.in_pad = False
        return (0,0)
            
    def act_Rclick(self):
        if not self.Rclicking :
            self.Rclicking = True
            self.mouse.clickRight()

    def act_Dclick(self):
        if not self.Dclicking :
            self.Dclicking = True
            self.mouse.doubleClick()

    def act_scroll(self,zero_y):
        R_zero_ydif = zero_y-self.R_zero_bef[1]
        if abs(R_zero_ydif) < 0.3 : R_zero_ydif = 0
        
        moveY = self.sense * math.sqrt(pow(abs(R_zero_ydif*3),3))*(1 if R_zero_ydif>0 else -1)
        
        pg.scroll((-1)*int(moveY))

    def action(self,sh,pfm):
        self.cus_cur = [pfm[0][0], pfm[0][1]]
        
        if sh[0] != 29 :
            self.rarr = [5,5]
        if distance(self.w_c,self.h_c,pfm[0][2],pfm[0][3]) < 10 :
            self.rarr[0] = 10
        if distance(self.w_c+self.w,self.h_c+self.h,pfm[0][2],pfm[0][3]) < 10 :
            self.rarr[1] = 10

        if sh[0] == 29 :
            if self.rarr[0] == 10 :
                if pfm[0][3] < self.h_c + self.h :
                    t = self.h_c
                    self.h_c = pfm[0][3]
                    if self.h_c < 0 : self.h_c = 0
                    self.h = self.h + t - self.h_c
                if pfm[0][2] < self.w_c + self.w :
                    t = self.w_c
                    self.w_c = pfm[0][2]
                    if self.w_c < 0 : self.w_c = 0
                    self.w = self.w + t - self.w_c
            if self.rarr[1] == 10 :
                if pfm[0][3] - self.h_c > 0:
                    self.h = int(pfm[0][3] - self.h_c)
                if pfm[0][2] - self.w_c > 0:
                    self.w = int(pfm[0][2] - self.w_c)
        
        mx, my = self.cal_mov((pfm[0][0],pfm[0][1]))
        if self.in_pad :
            if sh[1] == 25 :
                self.mouse.movePos(mx/1.35,my/1.35)
                self.mouse.down()
            else :
                self.mouse.up()
                if sh[1] == 24 :
                    self.mouse.movePos(mx/1.35,my/1.35)
                else : self.cus_bef = [-1,-1]
            
            if sh[1] == 9 : self.act_Dclick()
            else : self.Dclicking = False
            
            if sh[1] == 28 : self.act_Rclick()
            else : self.Rclicking = False
            
            if sh[1] == 12 : self.act_scroll(pfm[1][1])
        
        self.R_zero_bef = (pfm[1][0], pfm[1][1])
        self.cus_bef = self.cus_cur
