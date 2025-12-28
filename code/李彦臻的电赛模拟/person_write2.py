
from maix import camera, display,image,time


#初始化参数
threshold1=[[4,25,-11,9,-11,9]]#找黑线
threshold2=[[71,85,-21,18,-21,18]]#白块色
disp=display.Display()
#cam=camera.Camera(320,320)

img=image.load("/root/project-1/棋盘（任务一白棋）.png")
img.resize(320,320)
disp.show(img)
time.sleep(1)
circle=img.find_circles()
for c in circle:
    print("c:",c)
    #img.draw_circle(c.x(),c.y(),c.r(),color=image.Color.from_rgb(255,0,0))
disp.show(img)
time.sleep(1)

#放射变换方法，？怎么跟着官网还是不行
#img_affine=img.affine([(205, 207), (817, 207),(205, 815)],[(-1.7, 347.2),(517.4, 867.1),(-517.4, 867.1)])
#disp.show(img_affine)
#img = image.Image(320, 240, image.Format.FMT_RGB888)
#img_new = img.affine([(10, 10), (100, 10), (10, 100)], [(10, 10), (100, 20), (20, 100)])
#print(img, img_new)
#time.sleep(1)
#缩放方法resize不知道为什么用不了
#img_resize=img.resize(640,480)
#disp.show(img_resize)
#print(img_resize)