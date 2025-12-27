import threading
from maix import camera, display, app, image, touchscreen, time, uart, pinmap
import traceback
import cv2 as cv
import numpy as np

# 初始化
disp = display.Display()
cam = camera.Camera(360, 360, fps=30)
ts = touchscreen.TouchScreen()  # 初始化触摸屏
judge_data = [] # 临时判断列表
real_data = []  # 棋盘历史数据列表，长度最大为10
one_group_data = []  # 临时储存发送数据列表
should_exit = False  # 全局退出标志
angle = None  # 角度


class UartHandler:
    # 定义帧头帧尾映射（避免硬编码，便于扩展）
    FRAME_CONFIG = {
        1: (0xA1, 0xAA, 0xA2), # 发送角度
        2: (0xA1, 0xBB, 0xA2), # 下在哪里
        3: (0xA1, 0xCC, 0xA2), # 违规情况
        4: (0xA1, 0xEE, 0xA2), # 黑棋赢
        5: (0xA1, 0xDD, 0xA2), # 白棋赢
        6: (0xA1, 0xFF, 0xA2), # 平局
        7: (0xF1, 0xF2),
        8: (0xE1, 0xE2)
    }

    def __init__(self,
                 Pin_1="A18",  # UART1_RX引脚
                 Pin_2="A19",  # UART1_TX引脚
                 Rx="UART1_RX",
                 Tx="UART1_TX",
                 bitrate=9600,
                 device="/dev/ttyS1"):
        # 1. 硬件引脚初始化（UART1）
        pinmap.set_pin_function(Pin_1, Rx)
        pinmap.set_pin_function(Pin_2, Tx)
        self.serial = uart.UART(device, bitrate)

        # 2. 负数映射配置
        self.offset = 50  # 偏移量（十进制）

        # 接收缓存
        self.receive_data = []
        self.lock = threading.Lock()
        self.running = True

        threading.Thread(target=self._recv_loop, daemon=True).start()

        print(f"串口初始化完成：{device} ，波特率={bitrate}")

    def _recv_loop(self):
        while self.running:
            read_byte = self.serial.read(1)
            if read_byte:
                hex_num = ord(read_byte)  # 字节→十进制数值
                with self.lock:
                  self.receive_data.append(hex_num)
            time.sleep(0.001)

    def get_data(self, clear=True):
        """获取接收数据（修改：提取有效帧后清空全部缓存）"""
        with self.lock:
            # 复制缓存数据，避免操作原列表
            data = self.receive_data.copy()
            # 清空原缓存
            if clear:
                self.receive_data = []

        # 提取所有帧头、帧尾及对应关系
        frame_heads = [v[0] for v in self.FRAME_CONFIG.values()]
        frame_tails = [v[1] for v in self.FRAME_CONFIG.values()]
        head_tail_map = dict(zip(frame_heads, frame_tails))  # {帧头:帧尾}

        complete_frame = []
        tail_pos = -1
        match_head = None

        # 步骤1：从后往前找最后一个有效帧尾
        for i in range(len(data)-1, -1, -1):
            if data[i] in frame_tails:
                tail_pos = i
                # 找到该帧尾对应的帧头
                match_head = [k for k, v in head_tail_map.items() if v == data[i]][0]
                break

        # 步骤2：未找到帧尾 → 把数据放回缓存（仅clear=True时）
        if tail_pos == -1:
            if clear:
                with self.lock:
                    self.receive_data = data  # 放回未解析数据
            return []

        # 步骤3：从帧尾往前找对应的帧头
        head_pos = -1
        for i in range(tail_pos-1, -1, -1):
            if data[i] == match_head:
                head_pos = i
                break

        # 步骤4：未找到对应帧头 → 清空缓存，直接返回空
        if head_pos == -1:
            return []

        # 步骤5：提取完整帧
        complete_frame = data[head_pos:tail_pos+1]

        return complete_frame

    def send_data(self, data_list, frame_type):
        """
        通用发送方法
        :param data_list: 待发送的原始数据列表
        :param frame_type: 帧类型（1/2/3···，对应FRAME_CONFIG中的配置）
        :return: 发送的字节数，失败返回None
        """
        try:
            # 1. 负数映射
            mapped_data = [num + self.offset for num in data_list]

            # 2. 获取帧头、帧尾并拼接完整帧
            header_1, data_type, footer = self.FRAME_CONFIG[frame_type]
            full_frame = [header_1, data_type] + mapped_data + [footer]

            # 3. 转字节并发送
            send_bytes = bytes(full_frame)
            send_len = self.serial.write(send_bytes)

            # 4. 调试打印
            hex_str = ' '.join([f'0x{b:02X}' for b in send_bytes])
            print(f"十六进制：{hex_str} | 字节数：{send_len}")

            return send_len

        except Exception as e:
            print(f"发送失败：{e}")
            return None

    # 发送接口
    def send_1(self, data_list):
        """仅发送角度"""
        return self.send_data(data_list, 1)

    def send_2(self, data_list):
        return self.send_data(data_list, 2)

    def send_3(self, data_list):
        return self.send_data(data_list, 3)

    def send_4(self, data_list):
        return self.send_data(data_list, 4)

    def send_5(self, data_list):
        return self.send_data(data_list, 5)

    def send_6(self, data_list):
        return self.send_data(data_list, 6)

    def close(self):
        """关闭串口"""
        self.running = False  # 停止接收线程
        time.sleep(0.01)      # 等待线程退出
        self.serial.close()
        print("串口已关闭")


def check_win(current_data):
    """
    判断棋盘输赢
    :param current_data: 18位棋盘数据
    :return: 11=黑赢，22=白赢，33=平局，0=继续；同时打印匹配的赢线
    """
    # 赢线位置组合
    win_lines = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 5, 9], [3, 5, 7]]

    for line in win_lines:
        # 计算格子索引并获取颜色值
        idx1 = 2 * line[0] - 1
        idx2 = 2 * line[1] - 1
        idx3 = 2 * line[2] - 1
        color1 = current_data[idx1]
        color2 = current_data[idx2]
        color3 = current_data[idx3]

        # 黑棋赢判断
        if color1 == color2 == color3 == 1:
            print(f"黑棋赢, 赢线：{line}")
            return 11
        # 白棋赢判断
        if color1 == color2 == color3 == 2:
            print(f"白棋赢, 赢线：{line}")
            return 22

        # 平局判断
        color_indices = [2 * n - 1 for n in range(1, 10)]
        is_board_full = True  # 标记是否下满
        for idx in color_indices:
            if current_data[idx] == 0:  # 存在空位置
                is_board_full = False
                break  # 只要有一个空位置，就不是满的，直接退出循环

        if is_board_full:
            print("棋盘已下满，平局")
            return 33  # 平局

    return 0


def judge_rules(prev_data, curr_data):
    """
    判断落子是否合规
    :param prev_data: 上一次棋盘数据
    :param curr_data: 当前棋盘数据
    :return: bool
    """

    # 提取两次棋盘的颜色位值
    prev_color = [prev_data[2 * n - 1] for n in range(1, 10)]  # 上一次各位置颜色
    curr_color = [curr_data[2 * n - 1] for n in range(1, 10)]  # 当前各位置颜色

    # 找出两次颜色变化的位置
    changed_positions = []
    for idx in range(9):
        if prev_color[idx] != curr_color[idx]:
            changed_positions.append(idx)

    # 1、判断落子数量合规（仅能有1处变化）
    if len(changed_positions) > 1:
        print(f"犯规：多落子（共{len(changed_positions)}处位置变化")
        return False

    # 2、判断落子位置合规（只能落在空位置，且落子颜色有效）
    change_idx = changed_positions[0]  # 唯一变化的位置（0-8）
    prev_val = prev_color[change_idx]  # 上一次该位置颜色
    curr_val = curr_color[change_idx]  # 当前该位置颜色

    if prev_val != 0:
        print(f"犯规：覆盖已有棋子（棋盘{change_idx + 1}号位上一次为{prev_val}，当前为{curr_val}）")
        return False

    # 所有核心条件满足 → 合规
    return True


def attack_logic(attack_data):
    """
    三子棋落子逻辑（优先级：直接赢→堵对方→占中心→占角→占边）
    :param attack_data: 棋盘一次有效数据列表，状态0=空/1=黑/2=白
    :return: 最优落子格子序号（1-9），无落子位置返回None（棋盘满）
    """
    #  1、 数据校验与格式化
    # 初始化棋盘映射：格子序号(1-9) → 状态(0/1/2)，默认空
    board_map = {num: 0 for num in range(1, 10)}
    # 遍历数据列表
    for i in range(0, len(attack_data), 2):
        grid_num = attack_data[i]
        grid_state = attack_data[i + 1]
        if 1 <= grid_num <= 9 and grid_state in [0, 1, 2]:
            board_map[grid_num] = grid_state

    # 转3×3棋盘（行0-2，列0-2），方便赢线判断
    board = [
        [board_map[1], board_map[2], board_map[3]],  # 第0行：格子1-3
        [board_map[4], board_map[5], board_map[6]],  # 第1行：格子4-6
        [board_map[7], board_map[8], board_map[9]]   # 第2行：格子7-9
    ]

    # 定义所有赢线（(行,列)坐标），共8条
    win_lines = [
        [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],  # 横
        [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],  # 竖
        [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]  # 斜
    ]

    # 2、 定义己方/对方棋子
    my_piece = 2  # 己方：1=黑棋，改2=白棋
    enemy_piece = 1 # 敌方

    # 3、直接赢/堵对方赢
    for line in win_lines:
        # 提取当前赢线的3个状态值
        line_vals = [board[x][y] for x, y in line]

        # 3.1、先判断己方差1子，直接赢
        if line_vals.count(my_piece) == 2 and line_vals.count(0) == 1:
            empty_x, empty_y = line[line_vals.index(0)]
            # 坐标转格子序号：行x列y → x*3 + y +1
            return empty_x * 3 + empty_y + 1

        # 3.2. 再判断对方差1子，需要堵截
        if line_vals.count(enemy_piece) == 2 and line_vals.count(0) == 1:
            empty_x, empty_y = line[line_vals.index(0)]
            return empty_x * 3 + empty_y + 1

    # 4、占中心
    if board[1][1] == 0:
        return 5

    # 5、占角（格子1/3/7/9）
    corners = [(0, 0, 1), (0, 2, 3), (2, 0, 7), (2, 2, 9)]  # (行,列,格子序号)
    for x, y, grid_num in corners:
        if board[x][y] == 0:
            return grid_num

    # 6、占边（格子2/4/6/8）
    edges = [(0, 1, 2), (1, 0, 4), (1, 2, 6), (2, 1, 8)]  # (行,列,格子序号)
    for x, y, grid_num in edges:
        if board[x][y] == 0:
            return grid_num

    # 7、 棋盘已满
    return None


def main_1(qipan_data):
    """
    封装完整下棋逻辑
    :param qipan_data:
    :return: 操作，位置
    """
    result = None
    best_place = None
    if len(qipan_data) ==0:
        result =  0 # 0 代表继续下棋

    elif len(qipan_data) == 1:
        result = check_win(qipan_data[0])

    elif len(qipan_data) > 1:
        # 判断是否遵守规则
        rules = judge_rules(qipan_data[-2],qipan_data[-1])
        if rules: # 如果遵守了规则
            # 先判断有没有赢--0,11,22,33
            result = check_win(qipan_data[-1])
            # 如果没有赢并且没有平局
            if result == 0:
                best_place = attack_logic(qipan_data[-1])
                # print(f"下在{best_place}")
                time.sleep(0.01)
        else:
            uart_obj.send_3([])

    return result, best_place

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
                best_place = attack_logic(real_data[-1])
                uart_obj.send_2([best_place, angle])
                print(f"落 {best_place} 号格")

            elif recv_frame == [225, 2, 226]: # 正常对弈
                state_now, best_place = main_1(real_data) # state_now = 0，11,22,33 内 黑 白 平
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

        # 主动识别部分
        if len(real_data) > 2:
            is_legal = judge_rules(real_data[-2],real_data[-1])
            if is_legal:
                # 合法，不做处理
                 is_win = check_win(real_data[-1])
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
        # 串口初始化
        uart_obj = UartHandler()
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