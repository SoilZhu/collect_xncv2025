#图片测试
from typing import Sequence
from maix import camera, display,image,time
from maix._maix.image import COLOR_BLUE, COLOR_GREEN, COLOR_RED, Format, Line, Rect, Threshold

#初始化参数
threshold1=[[4,25,-11,9,-11,9]]#找黑线
threshold2=[[71,85,-21,18,-21,18]]#白块色
cam=camera.Camera(640,480)
img=image.load("/root/project-1/棋盘.png")
img=img.resize(480,480)
print(img.width())
print()
disp=display.Display()
#disp.show(img)
#time.sleep(2)
coord=[]#动态更新棋盘坐标roi
Sudoku=[]#存九宫格坐标
#找直线（多条直线怎么找？）

#找到棋盘主函数
blobs=img.find_blobs(threshold1,pixels_threshold=20000,area_threshold=150000,merge=False)
#img_new=img.copy()
if blobs:
    # 打印找到的色块数量
    print("找到 {} 个目标色块，blobs的类型为{}".format(len(blobs),type(blobs)))
    print(blobs)

for blob in blobs:
    coord=[blob.x(),blob.y(),blob.w(),blob.h()]
    Sudoku=[
        blob.x()+blob.w()//6,blob.y()+blob.h()//6,
        blob.x()+blob.w()*3//6,blob.y()+blob.h()//6,
        blob.x()+blob.w()*5//6,blob.y()+blob.h()//6,
            
        blob.x()+blob.w()//6,blob.y()+blob.h()*3//6,
        blob.x()+blob.w()*3//6,blob.y()+blob.h()*3//6,
        blob.x()+blob.w()*5//6,blob.y()+blob.h()*3//6,
        
        blob.x()+blob.w()//6,blob.y()+blob.h()*5//6,
        blob.x()+blob.w()*3//6,blob.y()+blob.h()*5//6,
        blob.x()+blob.w()*5//6,blob.y()+blob.h()*5//6]#注意：不能存为元组类型
    Sudoku = [i for i in Sudoku]
    print("Sudoku 类型:", type(Sudoku))
    #print("Sudoku解包后为{}".format(*Sudoku))#解包的错误使用
    #*a,=Sudoku
    #print(f"正确解包后为{a}")#其实也没啥用，还是返回的列表
    print("第一个坐标类型:", type(Sudoku), type(Sudoku))
    print("第一个坐标值:", Sudoku, Sudoku)
    img.draw_rect(int(blob[0]),int(blob[1]),int(blob[2]),int(blob[3]),image.Color.from_rgb(0, 220, 0),thickness=10)
    img.draw_keypoints(Sudoku,color=COLOR_RED)
    #img.draw_circle(blob.x()+blob.w()//6,blob.y()+blob.h()//6,50,color=(0,0,220))#只能画一个
    print("值为：",blob.x(),blob.y(),blob.w(),blob.h(),blob.cx(),blob.cy(),blob.pixels(),blob.area())
    '''
    #--------------
    #检验直线插入
    Line=img.get_regression(thresholds=threshold1,roi=coord,pixels_threshold=10)
    if Line:
        print("hello")
        for Li in Line:
            theta = Li.theta()
            rho = Li.rho()
            if theta > 90:
                theta = 270 - theta
            else:
                theta = 90 - theta
            img.draw_string(12, 10,"theta: " + str(theta) + ", rho: " + str(rho), image.COLOR_BLUE,3)
            print("Line找到的值:",Li[0],Li[1],Li[2],Li[3])
            img.draw_line(Li[0],Li[1],Li[2],Li[3],image.Color.from_rgb(0,0,220),10) 
        disp.show(img)
        time.sleep(3)
    #--------------
    '''
    #img_lpb=img.find_lbp([blob[0],blob[1],20,8])
    #print("lbp值为：",img_lpb)
    #disp.show(img)
    #img_new=img.crop((blob[0]-1),(blob[1]-1),((blob[2]//2)*2+2),((blob[3]//2)*2+2))

#输出图像
print("11")
disp.show(img)
time.sleep(5)

'''
#找小方块（out）
blob2=img_new.find_blobs(threshold2)
    if blob2:
        print("hello")
        for blob in blob2:
            img.draw_rect(blob[0],blob[1],blob[2],blob[3],COLOR_BLUE,10)
        disp.show(img_new)
        time.sleep(5)    
'''
    
'''
from maix import camera, display,image,time
from maix._maix.image import COLOR_GREEN
#img=image.load("/root/project-1/棋盘.png")
cam=camera.Camera()
disp=display.Display()
while True:
    img=cam.read()
    blobs=img.find_blobs([[0,15,-30,30,-30,30]],pixels_threshold=20000,area_threshold=150000)
    img_new=img.copy()
    for blob in blobs:
        img.draw_rect(blob[0],blob[1],blob[2],blob[3],COLOR_GREEN,10)
        print("值为：",blob.x(),blob.y(),blob.w(),blob.h(),blob.cx(),blob.cy(),blob.pixels(),blob.area())

#disp.show(img)

        img_new=img.crop(blob[0],blob[1],(blob[2]//2)*2,(blob[3]//2)*2)
    if blobs:
        disp.show(img_new)
    else:
        disp.show(img)
    print("wait 1s")
    time.sleep(1)
'''

