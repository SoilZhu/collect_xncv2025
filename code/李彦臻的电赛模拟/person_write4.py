'''
from maix import camera,display,image
from maix._maix.image import COLOR_BLACK, COLOR_BLUE, COLOR_GRAY, COLOR_RED, Color


Cam=camera.Camera(640,480)#image.Format.FMT_GRAYSCALE为灰度图
#Cam.gain()
#Cam.exp_mode(0)
Dis=display.Display()
Cam.skip_frames(30) 
while True:
    img=Cam.read()
    #img=img.lens_corr(strength=1)
    img.draw_rect(10,10,200,150,COLOR_RED)
    img.draw_string(300,300,"hello",COLOR_BLACK,scale=3)
    img.draw_line(200,200,200,300,COLOR_BLUE)
    img.draw_circle(150,150,30,COLOR_GRAY)
    img_new=img.copy()
    img_new=img_new.resize(480,350)
    #img_new=img_new.crop(10,10,100,100)
    #img_new=img_new.rotate(90)
    key=[10,20,30,40]
    img_new.draw_keypoints(key,COLOR_RED,5)
    Dis.show(img_new)
'''
#屏幕使用
'''

from maix import touchscreen, app, time

ts = touchscreen.TouchScreen()

pressed_already = False
last_x = 0
last_y = 0
last_pressed = False
while not app.need_exit():
    x, y, pressed = ts.read()
    if x != last_x or y != last_y or pressed != last_pressed:
        print(x, y, pressed)
        last_x = x
        last_y = y
        last_pressed = pressed
    if pressed:
        pressed_already = True
    else:
        if pressed_already:
            print(f"clicked, x: {x}, y: {y}")
            pressed_already = False
    time.sleep_ms(1)  # sleep some time to free some CPU usage
'''

'''
from maix import touchscreen, app, time, display, image

ts = touchscreen.TouchScreen()
disp = display.Display()

img = image.Image(disp.width(), disp.height())

# draw exit button
exit_label = "< Exit"
size = image.string_size(exit_label)
exit_btn_pos = [0, 0, 8*2 + size.width(), 12 * 2 + size.height()]
img.draw_string(8, 12, exit_label, image.COLOR_WHITE)
img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3],  image.COLOR_WHITE, 2)

def is_in_button(x, y, btn_pos):
    return x > btn_pos[0] and x < btn_pos[0] + btn_pos[2] and y > btn_pos[1] and y < btn_pos[1] + btn_pos[3]

while not app.need_exit():
    x, y, pressed = ts.read()
    if is_in_button(x, y, exit_btn_pos):
        app.set_exit_flag(True)
    img.draw_circle(x, y, 1, image.Color.from_rgb(255, 255, 255), 2)
    disp.show(img)
'''
#寻找色块
from maix import camera, display,image
from maix._maix.image import COLOR_GREEN
#img=image.load("/root/project-1/棋盘.png")
cam=camera.Camera()
dis=display.Display()

while True:
    img=cam.read()
    blobs=img.find_blobs([[0,25,-30,30,-30,30]])
    for blob in blobs:
        img.draw_rect(blob[0],blob[1],blob[2],blob[3],COLOR_GREEN,10)
        #print("值为：",blob.x(),blob.y(),blob.w(),blob.h(),blob.cx(),blob.cy(),blob.pixels(),blob.area())
    dis.show(img)