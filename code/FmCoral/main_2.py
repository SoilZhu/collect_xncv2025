from maix import camera, display, app, image, touchscreen, time
import traceback
import cv2 as cv
import numpy as np
import play_logic
import Uart

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
    center_points = np.array([[0, 0], [1 / 3, 0], [2 / 3, 0], [1, 0],
                              [0, 1 / 3], [1 / 3, 1 / 3], [2 / 3, 1 / 3], [1, 1 / 3],
                              [0, 2 / 3], [1 / 3, 2 / 3], [2 / 3, 2 / 3], [1, 2 / 3],
                              [0, 1], [1 / 3, 1], [2 / 3, 1], [1, 1]])

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
    global real_data
    global judge_data
    global one_group_data
    global angle
    done = 1
    # 定义卷积核
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    while not app.need_exit():
        img = cam.read()
        # 转换成OpenCV
        img_cv = image.image2cv(img, False, False)
        # 将原图转换为HSV颜色空间
        hsv_img = cv.cvtColor(img_cv, cv.COLOR_BGR2HSV)

        # 定义目标绿色的HSV范围--好像是通道转换步骤错了，红色HSV实际上变成了绿的，将错就错
        lower_green = np.array([100, 80, 50])  # H:100-130（覆盖青绿/草绿），S≥80（低饱和），V≥50（低明度）
        upper_green = np.array([130, 255, 255]) # S/V上限拉满
        # 3. 生成绿色掩码
        green_mask = cv.inRange(hsv_img, lower_green, upper_green)

        # 去除噪点-高斯模糊(暂定3,3)
        img_blur = cv.GaussianBlur(green_mask, [3, 3], 0)
        # 边缘检测-阈值暂定30,90
        img_edge = cv.Canny(img_blur, 30, 90)

        # 检查边缘是否被识别出
        # img_edge_maixcam = image.cv2image(img_edge)
        # disp.show(img_edge_maixcam)

        # 膨胀
        img_dilate = cv.dilate(img_edge, kernel)
        # 腐蚀
        img_erode = cv.erode(img_dilate, kernel)

        # 查找轮廓：得到仅包含绿色（实际为红色）区域的轮廓
        contours, _ = cv.findContours(img_erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # 检查
        # print(len(contours))
        if len(contours) > 0:
            # 找出最大轮廓
            biggest_contours = max(contours, key=cv.contourArea)
            # 近似多边形
            epsilon = 0.02 * cv.arcLength(biggest_contours, True)
            approx = cv.approxPolyDP(biggest_contours, epsilon, True)  # 角点个数

            if len(approx) == 4:
                # 转换数组
                corners = approx.reshape((4, 2))

                # 进行大矩形角点编号（左上0、右上1、右下2、左下3）
                corn = np.zeros((4, 2), dtype="int")
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
                cv.drawContours(img_cv, [approx], -1, (0, 0, 255), 2)

                # 检查棋盘是否被正常框出
                # img = image.cv2image(img_cv, False, False)

                # 检查是否识别出九个中心
                centers = find_center(corners)
                # print(centers) # 打印中心点坐标
                if len(centers) == 9:
                    one_group_data.clear()

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
                        width = 20
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
                        # img.draw_string(x, y-10, f'{value}', image.COLOR_BLUE)

                        # 根据值判定棋子颜色-暂定
                        color_chess = 0
                        if value < -105:
                            color_chess = 1  # 黑
                        elif value > -65:
                            color_chess = 2  # 白
                        else:
                            color_chess = 0

                        # 标记棋子信息
                        if color_chess == 1:
                            img.draw_string(x, y + 10, "black", image.COLOR_WHITE)
                        elif color_chess == 2:
                            img.draw_string(x, y + 10, "white", image.COLOR_BLACK)

                        one_group_data.extend([i + 1, color_chess])

                    judge_data.append(one_group_data.copy())

                    # 有效数据筛选
                    if len(judge_data) > 2:
                        # 先判断3个数据是否全相等
                        if len(judge_data) == 3 and judge_data[0] == judge_data[1] == judge_data[2]:
                            if len(real_data) == 0:  # 如果首次采集
                                real_data.append(judge_data.copy()[0])
                            else:
                                # 省略重复数据
                                if judge_data[0] != real_data[-1]:
                                    real_data.append(judge_data.copy()[0])
                            # 清空临时判断列表
                            judge_data.clear()
                        else:
                            # 3个数据不全相等，清空重采
                            judge_data.clear()

                    # 控制历史数据列表长度
                    if len(real_data) > 10:
                        real_data.pop(0)

                # 检查
                print (len(real_data), end="")
                if len(real_data) > 0:
                    # print(f"最新{real_data[-1]}")
                    print("·", end="")

            else:
                print(f"[ X ]角点{len(approx)}")

        elif len(contours) == 0:
            print(f"[ X ]轮廓{done}")

        # 对弈部分-触发数据部分
        recv_frame = uart_obj.get_data()
        if recv_frame and angle:
            print(f"收到：{recv_frame}")
            if recv_frame == [241, 1, 242]: # 发送角度数据
                uart_obj.send_1([angle])
                print(f"角度是{angle}")

            elif recv_frame == [225, 1, 226]: # 人类无落子 [序号，角度]
                best_place = play_logic.attack_logic(real_data[-1])
                uart_obj.send_2([best_place, angle])
                print(f"落 {best_place} 号格")

            elif recv_frame == [225, 2, 226]: # 正常对弈
                state_now, best_place = play_logic.main_1(real_data) # state_now = 0，11,22,33 内 黑 白 平
                if state_now == 0: # 继续下棋
                    uart_obj.send_2([best_place, angle])
                    print(f"落 {best_place} 号格")

                elif state_now == 11: # 黑棋赢
                    uart_obj.send_4([])

                elif state_now == 22: # 白棋赢
                    uart_obj.send_5([])

                elif state_now == 33: # 平局
                    uart_obj.send_6([])

            recv_frame.clear()

        # 主动识别错误部分
        if len(real_data) > 2:
            is_legal = play_logic.judge_rules(real_data[-2],real_data[-1])
            if is_legal:
                # 合法，不做处理
                 is_win = play_logic.check_win(real_data[-1])
                 if is_win == 11: # 黑棋赢
                    uart_obj.send_4([])

                 elif is_win == 22: # 白棋赢
                    uart_obj.send_5([])

                 elif is_win== 33: # 平局
                    uart_obj.send_6([])

            elif not is_legal:
                print("检测到违规")


        # 调用exit_program，判断是否需要退出
        if exit_program(img):
            break  # 退出程序

        done += 1
        disp.show(img)
        time.sleep(0.001)


if __name__ == '__main__':
    try:
        # 初始化
        disp = display.Display()
        cam = camera.Camera(360, 360, fps=30)
        ts = touchscreen.TouchScreen()  # 初始化触摸屏
        judge_data = []  # 临时判断列表
        real_data = []  # 棋盘历史数据列表，长度最大为10
        one_group_data = []  # 临时储存发送数据列表
        should_exit = False  # 全局退出标志
        angle = None  # 角度
        # 串口初始化
        uart_obj = Uart.UartHandler()

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