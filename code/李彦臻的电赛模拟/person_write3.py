# from maix import camera,image,display,time
# cam=camera.Camera()
# dis=display.Display()
# for i  in range(10):
#     img=cam.read()
#     dis.show(img)
#     time.sleep(1)
from maix import touchscreen, app, time,image,display
from maix._maix.image import Format

ts = touchscreen.TouchScreen()
img=image.Image(640,480,format=Format.FMT_RGB888)
img.clear()
dis=display.Display()
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
            img.draw_keypoints([x,y],image.Color.from_rgb(255,0,0))
            dis.show(img)
            pressed_already = False
    time.sleep_ms(1)  # sleep some time to free some CPU usage