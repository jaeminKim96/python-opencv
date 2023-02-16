import cv2
import numpy as np
import pafy

url = 'https://www.youtube.com/watch?v=tZixREYOIZQ'
video = pafy.new(url)
best = video.getbest()

cap=cv2.VideoCapture(best.url)
WIDTH = 500
HEIGHT = 300


cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH) # 500으로 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT) # 300으로 설정
rows, cols = HEIGHT, WIDTH 
map_y, map_x = np.indices((rows, cols), dtype=np.float32) #매핑 배열 선언

# 거울 왜곡
map_mirrorh_x,map_mirrorh_y = map_x.copy(), map_y.copy() 
map_mirrorv_x,map_mirrorv_y = map_x.copy(), map_y.copy()   
 
## 좌우 대칭 거울 좌표 계산식 (맨 왼쪽이 맨 오른쪽으로 들어가기)
map_mirrorh_x[: , cols//2:] = cols - map_mirrorh_x[:, cols//2:]-1

## 상하 대칭 거울 좌표 계산식 (맨 위가 제일 아래로 들어가기)
map_mirrorv_y[rows//2:, :] = rows - map_mirrorv_y[rows//2:, :]-1

# 물결 효과
map_wave_x, map_wave_y = map_x.copy(), map_y.copy()
map_wave_x = map_wave_x + 15*np.sin(map_y/50) # 숫자 작을수록 주기가 짧아짐
map_wave_y = map_wave_y + 15*np.sin(map_x/50)    


# 렌즈 효과
## 렌즈 효과, 중심점 이동
map_lenz_x = 2*map_x/(cols-1)-1 
map_lenz_y = 2*map_y/(rows-1)-1

## 극좌표 변환
r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)
r_convex = r.copy() #볼록
r_concave = r.copy() # 오목

## 볼록 렌즈 효과 매핑 좌표 연산
r_convex[r< 1] = r_convex[r<1] **2  # 효과를 어느 정도로 줄 것인가
print(r.shape, r_convex[r<1].shape) # [r<1]= 변환 구역 크기

## 오목 렌즈 효과 좌표 설정
r_concave[r< 1] = r_concave[r<1] **0.5 # 효과의 정도

## 렌즈 효과, 직교 좌표로 다시 전환 # 거리와 각도로 좌표 계산
map_convex_x, map_convex_y = cv2.polarToCart(r_convex, theta) 
map_concave_x, map_concave_y = cv2.polarToCart(r_concave, theta)

## 렌즈 효과, 좌상단 좌표 복원
map_convex_x = ((map_convex_x + 1)*cols-1)/2
map_convex_y = ((map_convex_y + 1)*rows-1)/2
map_concave_x = ((map_concave_x + 1)*cols-1)/2
map_concave_y = ((map_concave_y + 1)*rows-1)/2

while True:
    ret, frame = cap.read()
    frame = frame[:HEIGHT, :WIDTH]
    
    # remap함수로 리매핑
    mirrorh=cv2.remap(frame,map_mirrorh_x,map_mirrorh_y,cv2.INTER_LINEAR)
    mirrorv=cv2.remap(frame,map_mirrorv_x,map_mirrorv_y,cv2.INTER_LINEAR)
    wave = cv2.remap(frame,map_wave_x,map_wave_y,cv2.INTER_LINEAR, \
                    None, cv2.BORDER_REPLICATE)
    convex = cv2.remap(frame,map_convex_x,map_convex_y,cv2.INTER_LINEAR)
    concave = cv2.remap(frame,map_concave_x,map_concave_y,cv2.INTER_LINEAR)
    
    # 영상 합치기
    r1 = np.hstack(( frame, mirrorh, mirrorv))
    r2 = np.hstack(( wave, convex, concave))
    merged = np.vstack((r1, r2))

    cv2.imshow('distorted', merged)
    if cv2.waitKey(1) & 0xFF== 27:
        break
cap.release
cv2.destroyAllWindows()