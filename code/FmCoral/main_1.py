"""方案2：获取棋盘信息，发送【0xF1 0xF2 0x02 0xF3】触发，数据格式为[位置编号，有无棋子，旋转角度]，
编号为01-09,00无棋子，01黑棋，02白棋；角度数据线下谈。返回的数据为【0xA1 0xA2 · · · · · · · ·  0xA3】"""
import time

from maix import camera, display, app, image, touchscreen
import traceback
import cv2 as cv
import numpy as np
import Uart

# 初始化
disp = display.Display()
cam = camera.Camera(360, 360, fps = 30)
ts = touchscreen.TouchScreen()  # 初始化触摸屏
real_time_date = [] # 实时检测的三组数据
will_deliver_date = [] # 准备发送数据列表
delivered_date = [] # 已发送的数据，最大长度暂定为5
should_exit = False  # 全局退出标志
# # 串口初始化
uart_obj = Uart.UartHandler(
    Pin_1="A18",
    Pin_2="A19",
    Rx="UART1_RX",
    Tx="UART1_TX",
    bitrate=9600,
    device="/dev/ttyS1"
)


def find_center(corners):
    """
    根据四个角点计算出九个方格时中心位置
    :param corners: 角点坐标
    :return: 中心点坐标
    """
    # 排除识别到非矩形形状
    if len(corners) != 4:
        return []

    # 预定中心点坐标
    center_points = np.array([[0, 0], [1/3, 0], [2/3, 0], [1, 0],
                              [0, 1/3], [1/3, 1/3], [2/3, 1/3], [1, 1/3],
                              [0, 2/3], [1/3, 2/3], [2/3, 2/3], [1, 2/3],
                              [0, 1], [1/3, 1], [2/3, 1], [1, 1]])
    # print(center_points)
    # 转换目标点
    dst_points = np.array(corners, dtype=np.float32)
    # 归一化矩形坐标
    src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    # 添加映射关系
    transform_matrix = cv.getPerspectiveTransform(src_points, dst_points)
    # 转换坐标
    transformed_points = cv.perspectiveTransform(np.array([center_points], dtype=np.float32), transform_matrix)
    # 转换二维数组
    points = transformed_points[0]

    # 检查变换后是否是16个点
    if len(points) != 16:
        return []

    # 计算九宫格每个格子的中心
    centers = []
    for i in range(3):
        for j in range(3):
            # 小矩形角点坐标
            left_top = i * 4 + j
            right_top = left_top + 1
            left_bottom = (i + 1) * 4 + j
            right_bottom = left_bottom + 1

            # 计算中心
            x = (points[left_top][0] + points[right_top][0] +
                 points[left_bottom][0] + points[right_bottom][0]) / 4
            y = (points[left_top][1] + points[right_top][1] +
                 points[left_bottom][1] + points[right_bottom][1]) / 4

            centers.append((int(x), int(y)))

    return centers

def exit_program(img):
    """
    Exit触摸屏幕退出程序模块
    :param img: 用于承载标识
    :return: 是否退出
    """
    # 按钮配置
    exit_text, text_scale, padding = "Exit", 3, 8
    # 创建Exit按钮
    text_w, text_h = image.string_size(exit_text, scale=text_scale)
    btn_x = img.width() - text_w - padding * 2
    btn_y = img.height() - text_h - padding * 2
    btn_w, btn_h = text_w + padding * 2, text_h + padding * 2
    img.draw_rect(btn_x, btn_y, btn_w, btn_h, color=image.Color.from_rgb(51, 51, 51), thickness=1)
    img.draw_string(btn_x + padding, btn_y + padding, exit_text, color=image.COLOR_GREEN, scale=text_scale)

    # 触摸检测+点击判断
    global should_exit
    touch_x, touch_y, pressed = ts.read()  # 读取触摸数据
    if pressed and not should_exit:
        # 触摸坐标转换
        img_x, img_y = image.resize_map_pos_reverse(
            img.width(), img.height(), disp.width(), disp.height(),
            image.Fit.FIT_CONTAIN, touch_x, touch_y
        )
        # 判断是否点击按钮区域
        if btn_x < img_x < btn_x + btn_w and btn_y < img_y < btn_y + btn_h:
            should_exit = True  # 标记退出

    return should_exit  # 返回状态

def main():
    global will_deliver_date
    global delivered_date
    global real_time_date
    done = 1
    # 定义卷积核
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    while not app.need_exit():
        img = cam.read()
        # 转换成OpenCV格式灰度图
        img_cv = image.image2cv(img, False, False) # 这一步已经转化为BGR格式
        img_gray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)

        # 去除噪点-高斯模糊(暂定5,5)
        img_blur = cv.GaussianBlur(img_gray, [5,5], 0)
        # 边缘检测-阈值暂定5,150
        img_edge = cv.Canny(img_blur, 50, 150)

        # 检查边缘是否被识别出
        # img_edge_maixcam = image.cv2image(img_edge)
        # disp.show(img_edge_maixcam)

        # 膨胀
        img_dilate = cv.dilate(img_edge, kernel)
        # 腐蚀
        img_erode = cv.erode(img_dilate, kernel)

        # 查找轮廓
        contours, _ = cv.findContours(img_erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 检查
        # print(len(contours))
        if len(contours) > 0:
            # 找出最大轮廓
            biggest_contours = max(contours, key = cv.contourArea)
            # 棋盘镶嵌在一个圆里面，可能会用到第二大轮廓，预留
            # second_biggest = sorted(contours, key=cv.contourArea, reverse=True)[1]
            # 近似多边形
            epsilon = 0.02 * cv.arcLength(biggest_contours, True)
            approx = cv.approxPolyDP(biggest_contours, epsilon, True) # 角点个数

            if len(approx) == 4:
                # 转换数组
                corners = approx.reshape((4, 2))

                # 进行大矩形角点编号（左上0、右上1、右下2、左下3）
                corn = np.zeros((4,2), dtype = "int")
                corn_sum = corners.sum(axis=1)
                corn_sub = np.diff(corners, axis=1)
                # 左上，x+y最小
                corn[0] = corners[np.argmin(corn_sum)]
                # 右上，y-x最小
                corn[1] = corners[np.argmin(corn_sub)]
                # 右下, x+y最大
                corn[2] = corners[np.argmax(corn_sum)]
                # 左下，y-x最大
                corn[3] = corners[np.argmax(corn_sub)]
                corners = corn

                # 绘制棋盘外框
                cv.drawContours(img_cv, [approx], -1, (0, 255, 0), 2)
                # 检查棋盘是否被正常框出
                # img = image.cv2image(img_cv, False, False)

                # 检查是否识别出九个中心
                centers = find_center(corners)
                # print(centers) # 打印中心点坐标
                if len(centers) == 9:
                    one_group_deliver = [] # 临时储存发送数据列表
                    one_group_judge = [] # 判定是否发送临时列表

                    # 角度识别
                    rect = cv.minAreaRect(approx)
                    (cx, cy), (w, h), angle = rect
                    if 45 < angle < 90:
                        angle = abs(angle - 90)
                    elif 0 < angle < 45:
                        angle = -angle
                    elif angle == 0 or angle == 90:
                        angle = 0

                    angle = int(angle)
                    img.draw_string(5, 10, f"Angle: {angle}", image.COLOR_YELLOW)
                    # print(f"角度为{int(angle)}")

                    for i in range(9):
                        x, y = centers[i][0], centers[i][1]
                        # 在中心画点
                        cv.circle(img_cv, (x, y), 2, (0, 255, 0), -1)
                        img = image.cv2image(img_cv, False, False)
                        # 给格子编号
                        img.draw_string(x, y, f"{i + 1}", image.COLOR_WHITE)
                        # 在中心画框
                        width = 30
                        img.draw_rect(int(x - width / 2), int(y - width / 2), width, width,
                                      image.COLOR_WHITE)

                        # 直方图设置
                        # 增强对比度
                        img.histeq(adaptive=True)
                        # 设置像素值统计范围，roi大小
                        hist = img.get_histogram(thresholds=[[0, 100, -128, 127, -128, 127]],
                                                 roi=[int(x - width / 2), int(y - width / 2), width, width])
                        # 提取亮度通道中位数
                        value = hist.get_statistics().a_median()

                        # 检查数值大小，调整阈值
                        # img.draw_string(x, y+10, f'{value}', image.COLOR_WHITE)

                        # 根据值判定棋子颜色-暂定
                        color_chess = 0
                        if value < -110:
                            color_chess = 1 # 黑
                        elif value > -80:
                            color_chess = 2 # 白
                        else:
                            color_chess = 0

                        if color_chess == 1:
                            img.draw_string(x, y + 10, "black", image.COLOR_WHITE)

                        elif color_chess == 2:
                            img.draw_string(x, y + 10, "white", image.COLOR_BLACK)

                        one_group_deliver.extend([i+1,color_chess, angle])
                        one_group_judge.extend([i+1, color_chess])

                    # 存入棋盘数据(软件做的逻辑，只发送原始棋盘数据)

                    if len(one_group_deliver) == 27:
                        real_time_date.append(one_group_judge.copy())
                        if len(real_time_date) >3:
                            real_time_date.pop(0)
                            if len(real_time_date) == 3 and real_time_date[0] == real_time_date[1] == real_time_date[2]:
                                # 存入准备发送列表数据
                                will_deliver_date = one_group_deliver.copy()
                                real_time_date.clear()

                        one_group_deliver.clear()
                        one_group_judge.clear()

                # print(real_time_date)

                # 方案1：通过串口发送数据-一直发送
                if will_deliver_date and done % 3 == 0:
                    uart_obj.send_data(will_deliver_date)
                    # 先判断delivered_date是否为空，再对比
                    if len(will_deliver_date) > 0:
                        # 首次发送,直接追加
                        if len(delivered_date) == 0:
                            delivered_date.append(will_deliver_date.copy())
                        # 非首次发送,对比最后一条，不同才追加
                        elif will_deliver_date != delivered_date[-1]:
                            delivered_date.append(will_deliver_date.copy())
                    # 控制列表最大长度为5
                    if len(delivered_date) > 5:
                        delivered_date.pop(0)
                    will_deliver_date.clear()
                    time.sleep(0.01)

            else:
                print(f"角点数为{len(approx)}")

        elif len(contours) == 0:
            print(f"未检测到轮廓{done}")

         # 调用exit_program，判断是否需要退出
        if exit_program(img):
            break  # 退出程序

        done += 1
        disp.show(img)

if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        # 显示报错信息
        print(f"程序异常报错：{e}")
        traceback.print_exc()  # 显示报错

    finally:
        # 退出时释放资源
        cam.close()
        disp.close()
        uart_obj.close()
        print("程序退出")

