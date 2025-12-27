from collections import deque
from maix import camera, display, image, time, app, touchscreen
import math
import numpy as np
import cv2
from Usart import Usart0

iRGB = lambda r, g, b: image.Color.from_rgb(r, g, b)
cam = camera.Camera(500, 368, fps=60)
disp = display.Display()
ts = touchscreen.TouchScreen()

mode = 1
Color = 1
radius = 10

x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = x5 = y5 = x6 = y6 = x7 = y7 = x8 = y8 = x9 = y9 = 0

initial_coordinates = {}
has_initial_coordinates = False
reference_angle = 0.0  # 初始角度

coordinates = {}
thresholds_total = []

# 颜色阈值
thresholds = [[0, 80, 40, 80, 10, 80]]  # red
usr_thresholds = [0, 0, 0, 0, 0, 0]

threshold_red = [0, 80, 30, 100, -120, -60]

threshold_white1 = [60, 89, -15, 12, -30, -10]
threshold_white2 = [70, 90, -13, 7, -26, -6]
threshold_white3 = [63, 83, -12, 8, -25, -5]

threshold_black1 = [0, 18, -3, 17, -12, 8]  
threshold_black2 = [0, 25, -10, 13, -9, 11]
threshold_black3 = [0, 25, -3, 17, -14, 6]

current_index = -1

get_label = "Get"
size = image.string_size(get_label)
get_btn_pos = [260, 0, 8*2 + size.width(), 12*2 + size.height()]

start_label = "Start  "
size = image.string_size(start_label)
start_btn_pos = [0, 60, 8*2 + size[0], 12*2 + size[1]]

color_label = "Color"
size = image.string_size(color_label)
color_btn_pos = [0, 120, 8*2 + size[0], 12*2 + size[1]]

minus_label = "EXIT"
size = image.string_size(minus_label)
minus_btn_pos = [0, 240, 8*2 + size[0], 12*2 + size[1]]

# AI表示机器下棋
ai_color_label = "AI: White"
size = image.string_size(ai_color_label)
ai_color_btn_pos = [0, 180, 8*2 + size[0], 12*2 + size[1]]

reset_label = "Reset"
size = image.string_size(reset_label)
reset_btn_pos = [0, 300, 8*2 + size[0], 12*2 + size[1]]

# 预设棋局
preset1_label = "Preset1"
size = image.string_size(preset1_label)
preset1_btn_pos = [450, 0, 8*2 + size[0], 12*2 + size[1]]

preset2_label = "P2"
size = image.string_size(preset2_label)
preset2_btn_pos = [450, 40, 8*2 + size[0], 12*2 + size[1]]

preset3_label = "P3"
size = image.string_size(preset3_label)
preset3_btn_pos = [450, 80, 8*2 + size[0], 12*2 + size[1]]

place7_label = "Place 7"
size = image.string_size(place7_label)
place7_btn_pos = [450, 120, 8*2 + size[0], 12*2 + size[1]]
target_place_color = 1  # 放置目标颜色（1=白，2=黑）

# 定义三局预设棋局序列 (每组3黑2白)
# 预设1: 
preset1_sequence = [
    (1, 2),  # 1黑
    (5, 1),  # 5白
    (3, 2),  # 3黑
    (7, 1),  # 7白
    (9, 2)   # 9黑
]

# 预设2: 
preset2_sequence = [
    (2, 2),  # 2黑
    (4, 1),  # 4白
    (6, 2),  # 6黑
    (8, 1),  # 8白
    (5, 2)   # 5黑
]

# 预设3: 
preset3_sequence = [
    (3, 2),  # 3黑
    (1, 1),  # 1白
    (7, 2),  # 7黑
    (9, 1),  # 9白
    (5, 2)   # 5黑
]

# 博弈
ai_color = 1  # 1:白棋 2:黑棋，默认AI执白
is_ai_turn = False 
game_over = False
last_pressed = False

# 预设投放
current_preset_step = 0
current_preset_sequence = []


def draw_btns(img):
    img.draw_string(268, 12, get_label, image.COLOR_YELLOW)
    img.draw_string(8, 132, color_label, image.COLOR_YELLOW)
    img.draw_string(8, 252, minus_label, image.COLOR_YELLOW)
    # 机器拿的棋子的颜色按钮
    ai_text = "AI: White" if ai_color == 1 else "AI: Black"
    img.draw_string(8, 192, ai_text, image.COLOR_YELLOW)
    img.draw_string(8, 312, reset_label, image.COLOR_YELLOW)
    img.draw_string(358, 12, preset1_label, image.COLOR_GREEN)
    img.draw_string(358, 52, preset2_label, image.COLOR_GREEN)
    img.draw_string(358, 92, preset3_label, image.COLOR_GREEN)
    img.draw_string(358, 132, place7_label, image.COLOR_GREEN)
    color_text = f"Target: {'White' if target_place_color == 1 else 'Black'}"
    img.draw_string(358, 172, color_text, image.COLOR_GREEN, scale=0.8)


def load_preset(preset_num):
    global qipan, game_over, is_ai_turn, current_preset_step, current_preset_sequence
    qipan = [0] * 9 
    if preset_num == 1:
        current_preset_sequence = preset1_sequence.copy()
    elif preset_num == 2:
        current_preset_sequence = preset2_sequence.copy()
    else:
        current_preset_sequence = preset3_sequence.copy()
    
    current_preset_step = 0 
    game_over = False
    is_ai_turn = False
    send_next_preset_piece()
    print(f"Loaded preset {preset_num}, starting sequence")


# 发送预设棋子
def send_next_preset_piece():
    global current_preset_step, current_preset_sequence, game_over, qipan
    if current_preset_step < len(current_preset_sequence):
        pos, color = current_preset_sequence[current_preset_step]
        qipan[pos-1] = color  # 更新棋盘状态
        if pos in coordinates:
            Usart0.send_chess_cmd(pos, color)
            print(f"Preset step {current_preset_step+1}: 位置{pos}, {'白棋' if color == 1 else '黑棋'}")
            current_preset_step += 1
        # 发送完整棋盘状态
        Usart0.send_board_state(qipan)
    else:
        print("Preset sequence completed")
        game_over = True


# 获取阈值
def collect_threshold_LAB(img, x, y):
    ROI = (x - radius, y - radius, radius * 2, radius * 2)
    Statistics = img.get_statistics(roi=ROI)
    thresholds = (Statistics.l_mode()-20, Statistics.l_mode()+20,
                 Statistics.a_mode()-20, Statistics.a_mode()+20,
                 Statistics.b_mode()-20, Statistics.b_mode()+20)
    print(x, y)
    img.draw_rect(x - radius, y - radius, radius * 2, radius * 2, color=image.COLOR_YELLOW, thickness=2)
    print(thresholds)
    return thresholds


def is_in_button(x, y, btn_pos):
    return btn_pos[0] < x < btn_pos[0] + btn_pos[2] and btn_pos[1] < y < btn_pos[1] + btn_pos[3]


# 计算角度
def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx))


# 旋转坐标
def rotate_point(x, y, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    x_trans = x - cx
    y_trans = y - cy
    x_rot = x_trans * math.cos(angle_rad) - y_trans * math.sin(angle_rad)
    y_rot = x_trans * math.sin(angle_rad) + y_trans * math.cos(angle_rad)
    x_new = x_rot + cx
    y_new = y_rot + cy
    return int(x_new), int(y_new)


executedOnce = True
start_blob_press = False
blobs_mode = False
points = deque()


def key_clicked(btn_rects):
    global last_pressed
    x, y, pressed = ts.read()
    if pressed:
        for i, btn in enumerate(btn_rects):
            if is_in_button(x, y, btn):
                if not last_pressed:
                    last_pressed = True
                    return True, i, btn
    else:
        last_pressed = False
    return False, 0, []


def check_mode_switch(img: image.Image, disp_w, disp_h):
    global mode
    btns = [[0, 0, 100, 40]]
    btn_rects_disp = [image.resize_map_pos(img.width(), img.height(), disp_w, disp_h, image.Fit.FIT_CONTAIN, 
                                         btns[0][0], btns[0][1], btns[0][2], btns[0][3])]
    clicked, idx, rect = key_clicked(btn_rects_disp)
    if clicked:
        mode += 1
        if mode > 3:
            mode = 1
    img.draw_string(2, 10, f"mode {mode}", color=image.COLOR_YELLOW, scale=1.5)
    img.draw_rect(btns[0][0], btns[0][1], btns[0][2], btns[0][3], image.COLOR_YELLOW, 2)


# 距离阈值
THRESHOLD_DISTANCE = 12 
qipan = [0] * 9


def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) **2 + (y1 - y2)** 2) **0.5


def compare_coordinates(x_black, y_black, coordinates):
    for key, (x, y) in coordinates.items():
        distance = euclidean_distance(x_black, y_black, x, y)
        if distance <= THRESHOLD_DISTANCE:
            return key 
    return None


def is_qipan_nonzero(qipan):
    return any(value != 0 for value in qipan)


# 棋子坐标存储
black_chess_coordinates = []
white_chess_coordinates = []

black_chess1_x = black_chess1_y = None
black_chess2_x = black_chess2_y = None
black_chess3_x = black_chess3_y = None
black_chess4_x = black_chess4_y = None
black_chess5_x = black_chess5_y = None

white_chess1_x = white_chess1_y = None
white_chess2_x = white_chess2_y = None
white_chess3_x = white_chess3_y = None
white_chess4_x = white_chess4_y = None
white_chess5_x = white_chess5_y = None


def is_coordinate_near(existing_coords, new_coord, threshold=10):
    for coord in existing_coords:
        distance = math.sqrt((coord[0] - new_coord[0])** 2 + (coord[1] - new_coord[1]) **2)
        if distance < threshold:
            return True
    return False


# 博弈
def check_win(board, color):
    # 检查行
    for i in range(3):
        if board[i*3] == board[i*3+1] == board[i*3+2] == color:
            return True
    # 检查列
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] == color:
            return True
    # 检查对角线
    if board[0] == board[4] == board[8] == color:
        return True
    if board[2] == board[4] == board[6] == color:
        return True
    return False


def is_draw(board):
    return all(cell != 0 for cell in board)


def evaluate(board):
    if check_win(board, ai_color):
        return 10
    opponent = 2 if ai_color == 1 else 1
    if check_win(board, opponent):
        return -10
    return 0


def minimax(board, depth, is_maximizing):
    score = evaluate(board)
    if score != 0:
        return score - depth  # 优先选择更快获胜的走法
    
    if is_draw(board):
        return 0

    if is_maximizing:
        max_score = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = ai_color
                score = minimax(board, depth + 1, False)
                board[i] = 0
                max_score = max(score, max_score)
        return max_score
    else:
        min_score = float('inf')
        opponent = 2 if ai_color == 1 else 1
        for i in range(9):
            if board[i] == 0:
                board[i] = opponent
                score = minimax(board, depth + 1, True)
                board[i] = 0
                min_score = min(score, min_score)
        return min_score


def ai_move(board):
    best_score = -float('inf')
    best_move = -1
    for i in range(9):
        if board[i] == 0:
            board[i] = ai_color
            score = minimax(board, 0, False)
            board[i] = 0
            if score > best_score:
                best_score = score
                best_move = i
    return best_move if best_move != -1 else None


def toggle_ai_color():
    global ai_color, game_over
    ai_color = 2 if ai_color == 1 else 1
    game_over = False
    return ai_color


def update_qipan(x, y, color, coordinates):
    global is_ai_turn, game_over
    if game_over:
        return
    pos = compare_coordinates(x, y, coordinates)
    if pos is not None and 1 <= pos <= 9 and qipan[pos-1] == 0:
        qipan[pos-1] = color
        # 检查是否是玩家落子且不是机器回合
        human_color = 2 if ai_color == 1 else 1
        if color == human_color:
            is_ai_turn = True  # 轮到机器走棋


def find_qipan():
    global x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9
    global qipan, coordinates, initial_coordinates, has_initial_coordinates, reference_angle
    global threshold_red, threshold_white1, threshold_white2, threshold_white3
    global threshold_black1, threshold_black2, threshold_black3, Color, current_index, thresholds_total
    global is_ai_turn, game_over, current_preset_sequence, current_preset_step, target_place_color

    usr_thresholds = [0, 0, 0, 0, 0, 0]
    pause_press = False
    start_press = False
    color_press = False
    plus_press = False
    minus_press = False
    
    # 棋盘参数
    find_center_method = 1  # 1, 2
    area_threshold_qipan = 80
    pixels_threshold_qipan = 50
    thresholds_qipan = [(0, 40, 49, 55, -1, 71)] # red
    
    # 棋子参数
    area_threshold_qizi = 300
    pixels_threshold_qizi = 300

    thresholds_total.append(threshold_red)
    thresholds_total.append(threshold_black1)
    thresholds_total.append(threshold_black2)
    thresholds_total.append(threshold_black3)
    thresholds_total.append(threshold_white1)
    thresholds_total.append(threshold_white2)
    thresholds_total.append(threshold_white3)
    
    t1 = 0
    
    while mode == 1:
        x, y, pressed = ts.read()
        current_time = time.time()
        
        img = cam.read()
        img.lens_corr(strength=1.7)
        
        blobs = img.find_blobs(thresholds_total, roi=[70,1,220, img.height()-1], x_stride=2, y_stride=1, 
                             area_threshold=area_threshold_qizi, pixels_threshold=pixels_threshold_qizi)
        draw_btns(img)
        img.draw_string(8, 72, f"C{Color}", image.COLOR_YELLOW)
        
        x, y = image.resize_map_pos_reverse(img.width(), img.height(), disp.width(), disp.height(), 
                                          image.Fit.FIT_CONTAIN, x, y)
        if t1 > 30:
            Usart0.usart_receive()
            if current_preset_sequence and current_preset_step < len(current_preset_sequence):
                if Usart0.ok_received:
                    Usart0.reset_ok_flag()
                    send_next_preset_piece()
            t1 = 0
        if pressed:
            img.draw_circle(x, y, 1, image.Color.from_rgb(255, 255, 255), 2)
            # 检测按钮点击
            if is_in_button(x, y, get_btn_pos):
                pause_press = True
                start_press = False
                color_press = False
                plus_press = False
                minus_press = False

            if is_in_button(x, y, start_btn_pos):
                start_press = True
                pause_press = False
                color_press = False
                plus_press = False
                minus_press = False
                global executedOnce
                executedOnce = True
                
            if is_in_button(x, y, color_btn_pos):
                executedOnce = True
                color_press = True
                pause_press = False
                start_press = False
                plus_press = False
                minus_press = False
                # 切换目标放置颜色
                target_place_color = 2 if target_place_color == 1 else 1
                print(f"目标颜色切换为: {'白色' if target_place_color == 1 else '黑色'}")

            if is_in_button(x, y, minus_btn_pos):
                infoimg = image.Image(disp.width(), disp.height(), image.Format.FMT_RGB888)
                infoimg.draw_string(
                    disp.width() // 2 - 80,
                    disp.height() // 2,
                    "EXITING...",
                    color=iRGB(255, 0, 0),
                    scale=2,
                )
                disp.show(infoimg)
                break

            # 机器棋子颜色切换按钮
            if is_in_button(x, y, ai_color_btn_pos):
                new_color = toggle_ai_color()
                print(f"AI color changed to {'White' if new_color == 1 else 'Black'}")
                color_press = False
                start_press = False
                pause_press = False

            # 重置按钮
            if is_in_button(x, y, reset_btn_pos):
                qipan = [0] * 9
                game_over = False
                is_ai_turn = False
                current_preset_sequence = []
                current_preset_step = 0
                print("Game reset")

            # 预设棋局按钮点击处理
            if is_in_button(x, y, preset1_btn_pos):
                load_preset(1)
                print("Loaded preset 1 sequence")
            if is_in_button(x, y, preset2_btn_pos):
                load_preset(2)
                print("Loaded preset 2 sequence")
            if is_in_button(x, y, preset3_btn_pos):
                load_preset(3)
                print("Loaded preset 3 sequence")

            if is_in_button(x, y, place7_btn_pos):
                # 放置棋子到7号方格
                if 7 in coordinates:  # 检查7号位置坐标是否已识别
                    if qipan[6] == 0:  # 检查7号位置是否为空（索引6对应位置7）
                        # 发送放置指令到STM32
                        Usart0.send_chess_cmd(7, target_place_color)
                        # 更新本地棋盘状态
                        qipan[6] = target_place_color
                        print(f"放置{target_place_color}色棋子到7号位置")
                        # 发送更新后的棋盘状态
                        Usart0.send_board_state(qipan)
                    else:
                        print("7号位置已有棋子")
                else:
                    print("未识别到7号位置坐标，请确保棋盘已正确识别")

            points.append((x, y, current_time))

        else:
            if pause_press:
                threshold = collect_threshold_LAB(img, x, y)
                print(x, y)
                usr_thresholds = list(threshold)
                
                # 白棋阈值设置
                if Color == 1:
                    threshold_white1 = usr_thresholds
                    img.draw_string(8, 290, str(threshold_white1), image.COLOR_GREEN)
                # 黑棋阈值设置
                elif Color == 2:
                    threshold_black1 = usr_thresholds
                    img.draw_string(8, 290, str(threshold_black1), image.COLOR_GREEN)

            if start_press:
                if executedOnce:
                    executedOnce = False
                    Color += 1
                    if Color > 2:
                        Color = 1
                
            if color_press:
                if executedOnce:
                    executedOnce = False
                    thresholds_total = []
                    thresholds_total.append(threshold_red)
                    thresholds_total.append(threshold_black1)
                    thresholds_total.append(threshold_black2)
                    thresholds_total.append(threshold_black3)
                    thresholds_total.append(threshold_white1)
                    thresholds_total.append(threshold_white2)
                    thresholds_total.append(threshold_white3)
                    img.draw_string(8, 290, str(threshold_white1), image.COLOR_GREEN)

        while points and current_time - points[0][2] > 1:
            points.popleft()
        
        # 棋盘检测逻辑
        if Usart0.qipan_flag == 0:
            img_cv = image.image2cv(img, False, False)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

            # 边缘检测
            edged = cv2.Canny(gray, 50, 100)

            # 膨胀处理
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edged, kernel, iterations=1)

            # 查找轮廓
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # 筛选最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)

                # 近似多边形
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                # 检测四边形
                if len(approx) == 4:
                    corners = approx.reshape((4, 2))

                    # 绘制顶点
                    for corner in corners:
                        cv2.circle(img_cv, tuple(corner), 4, (0, 255, 0), -1)

                    # 绘制轮廓
                    cv2.drawContours(img_cv, [approx], -1, (0, 255, 0), 2)

                    # 洪泛填充外部
                    img.flood_fill(corners[0][0] - 5, corners[0][1] - 5, 0.3, 0.3, image.COLOR_BLUE)
                    img.flood_fill(corners[1][0] - 5, corners[1][1] + 5, 0.5, 0.05, image.COLOR_BLUE)

                    # 排序角点（左上、右上、右下、左下）
                    tl = corners[0]
                    bl = corners[1]
                    br = corners[2]
                    tr = corners[3]
                    
                    # 计算3x3交叉点
                    cross_points = []
                    for i in range(4):
                        for j in range(4):
                            cross_x = int((tl[0] * (3 - i) + tr[0] * i) * (3 - j) / 9 +
                                        (bl[0] * (3 - i) + br[0] * i) * j / 9)
                            cross_y = int((tl[1] * (3 - i) + tr[1] * i) * (3 - j) / 9 +
                                        (bl[1] * (3 - i) + br[1] * i) * j / 9)
                            cross_points.append((cross_x, cross_y))
                            cv2.circle(img_cv, (cross_x, cross_y), 3, (0, 255, 0), -1)

                    centers = []
                    # 计算格子中心点
                    if find_center_method == 1:
                        for i in range(3):
                            for j in range(3):
                                center_x = int((cross_points[i * 4 + j][0] + cross_points[i * 4 + j + 1][0] + 
                                              cross_points[(i + 1) * 4 + j][0] + cross_points[(i + 1) * 4 + j + 1][0]) / 4)
                                center_y = int((cross_points[i * 4 + j][1] + cross_points[i * 4 + j + 1][1] + 
                                              cross_points[(i + 1) * 4 + j][1] + cross_points[(i + 1) * 4 + j + 1][1]) / 4)
                                centers.append((center_x, center_y))
                                cv2.circle(img_cv, (center_x, center_y), 2, (0, 255, 0), -1)
                    elif find_center_method == 2:
                        roi = [corners[:,0].min(), corners[:,1].min(), 
                              corners[:,0].max() - corners[:,0].min(), 
                              corners[:,1].max() - corners[:,1].min()]
                        img.draw_rect(roi[0], roi[1], roi[2], roi[3], image.COLOR_GREEN)
                        blobs = img.find_blobs(thresholds_qipan, roi=roi, x_stride=2, y_stride=1, 
                                              area_threshold=area_threshold_qipan, 
                                              pixels_threshold=pixels_threshold_qipan)
                        for b in blobs:
                            centers.append((b.cx(), b.cy()))
                            img.draw_circle(b.cx(), b.cy(), 2, image.COLOR_GREEN, -1)

                    if len(centers) == 9:
                        centers = np.array(centers)
                        rect = np.zeros((9, 2), dtype="float32")
                        s = centers.sum(axis=1)
                        idx_0 = np.argmin(s)
                        idx_8 = np.argmax(s)
                        diff = np.diff(centers, axis=1)
                        idx_2 = np.argmin(diff)
                        idx_6 = np.argmax(diff)
                        rect[0] = centers[idx_0]
                        rect[2] = centers[idx_2]
                        rect[6] = centers[idx_6]
                        rect[8] = centers[idx_8]
                        
                        calc_center = (rect[0] + rect[2] + rect[6] + rect[8]) / 4
                        mask = np.zeros(centers.shape[0], dtype=bool)
                        idxes = [1, 3, 4, 5, 7]
                        mask[idxes] = True
                        others = centers[mask]
                        idx_l = others[:,0].argmin()
                        idx_r = others[:,0].argmax()
                        idx_t = others[:,1].argmin()
                        idx_b = others[:,1].argmax()
                        found = np.array([idx_l, idx_r, idx_t, idx_b])
                        mask = np.isin(range(len(others)), found, invert=False)
                        idx_c = np.where(mask == False)[0]
                        
                        if len(idx_c) == 1:
                            rect[1] = others[idx_t]
                            rect[3] = others[idx_l]
                            rect[4] = others[idx_c]
                            rect[5] = others[idx_r]
                            rect[7] = others[idx_b]
                            
                            # 计算棋盘中心
                            cx = int(np.mean(rect[:, 0]))
                            cy = int(np.mean(rect[:, 1]))
                        
                            current_angle = calculate_angle(rect[0][0], rect[0][1], rect[2][0], rect[2][1])
                            
                            if not has_initial_coordinates:
                                reference_angle = current_angle
                                for i in range(9):
                                    initial_coordinates[i + 1] = (int(rect[i][0]), int(rect[i][1]))
                                has_initial_coordinates = True
                                adjusted_rect = rect
                            else:
                                # 旋转校正
                                angle_diff = reference_angle - current_angle
                                if -70 <= angle_diff <= 70:
                                    adjusted_rect = []
                                    for i in range(9):
                                        x_rot, y_rot = rotate_point(rect[i][0], rect[i][1], cx, cy, angle_diff)
                                        adjusted_rect.append([x_rot, y_rot])
                                    adjusted_rect = np.array(adjusted_rect)
                                else:
                                    adjusted_rect = rect
                            
                            # 绘制编号并更新坐标
                            for i in range(9):
                                img.draw_string(adjusted_rect[i][0], adjusted_rect[i][1], f"{i + 1}", 
                                              image.COLOR_GREEN, scale=2, thickness=-1)
                                x = int(adjusted_rect[i][0] / 2)
                                y = int(adjusted_rect[i][1] / 2)
                                coordinates[i + 1] = (x*2, y*2)
                                data3 = bytearray([0xF1, 0xF2, x, y, i+1, 0xF3])   
                                print(f"x:{x*2}, y:{y*2}, location:{i+1}") 
                                Usart0.serial.write_str(data3)
        
        if Usart0.qizi_flag == 0 and not current_preset_sequence:
            temp_qipan = [0] * 9               
            for b in blobs:                
                # 白棋检测
                if b.code() == 32 or b.code() == 16 or b.code() == 8:
                    if 300 < b.area() < 3000:      
                        enclosing_circle = b.enclosing_circle()
                        img.draw_circle(enclosing_circle[0], enclosing_circle[1], enclosing_circle[2], 
                                      image.COLOR_GREEN, 2)
                        x_white = int(enclosing_circle[0])
                        y_white = int(enclosing_circle[1])
                        pos = compare_coordinates(x_white, y_white, coordinates)
                        if pos:
                            temp_qipan[pos-1] = 1
                
                # 黑棋检测
                if b.code() == 1 or b.code() == 2 or b.code() == 4:
                    if 300 < b.area() < 3000:
                        enclosing_circle = b.enclosing_circle()
                        img.draw_circle(enclosing_circle[0], enclosing_circle[1], enclosing_circle[2], 
                                      image.COLOR_RED, 2)
                        x_black = int(enclosing_circle[0])
                        y_black = int(enclosing_circle[1])
                        pos = compare_coordinates(x_black, y_black, coordinates)
                        if pos:
                            temp_qipan[pos-1] = 2
            
            if temp_qipan != qipan:
                qipan = temp_qipan.copy()

            if not game_over and Usart0.qizi_flag == 0:
                human_color = 2 if ai_color == 1 else 1
                # 检查玩家是否获胜
                if check_win(qipan, human_color):
                    img.draw_string(200, 20, "You Win!", image.COLOR_GREEN, scale=2)
                    game_over = True
                elif is_draw(qipan):
                    img.draw_string(200, 20, "Draw!", image.COLOR_YELLOW, scale=2)
                    game_over = True
                elif is_ai_turn:
                    # 机器走棋
                    move = ai_move(qipan.copy())
                    if move != -1 and move is not None and qipan[move] == 0:
                        qipan[move] = ai_color
                        # 发送机器落子
                        if (move + 1) in coordinates:
                            Usart0.send_chess_cmd(move + 1, ai_color)
                            print(f"AI move: {move+1}")
                    if check_win(qipan, ai_color):
                        img.draw_string(200, 20, "AI Wins!", image.COLOR_RED, scale=2)
                        game_over = True
                    is_ai_turn = False

            # 发送棋盘状态
            if is_qipan_nonzero(qipan):
                Usart0.send_board_state(qipan)

        t1 += 1
        disp.show(img)
        time.sleep_ms(10)

if __name__ == "__main__":
    find_qipan()