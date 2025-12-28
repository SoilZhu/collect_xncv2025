#!/usr/bin/env python
from maix import camera, display, image, uart, app,time

from maix._maix.image import COLOR_BLUE, COLOR_RED


# --- 配置参数 ---
UART_DEV = "/dev/ttyS1" # 根据实际连接修改
BAUDRATE = 115200

# 颜色阈值 [L_min, L_max, A_min, A_max, B_min, B_max]
# 需通过工具实际校准
BOARD_TH = [(0, 100, -20, 20, -20, 20)] # 假设棋盘是深色或特定颜色，或者是找白色背景
BLACK_PIECE_TH = [[4,25,-11,9,-11,9]]#找黑线
WHITE_PIECE_TH = [[71,85,-21,18,-21,18]]#白块色

# 状态定义
EMPTY = 0
BLACK = 1
WHITE = 2

class TicTacToeSystem:
    def __init__(self):
        self.cam = camera.Camera(320, 240)
        self.disp = display.Display()
        self.uart = uart.UART(UART_DEV, baudrate=BAUDRATE)
        
        # 棋盘状态 3x3
        self.board_matrix = [[0,0,0], [0,0,0], [0,0,0]]
        self.prev_matrix = [[0,0,0], [0,0,0], [0,0,0]]
        self.stable_counter = 0
        self.grid_points = [] # 存储9个格子的中心坐标
        
        # 标定标志
        self.is_calibrated = False

    def find_board_corners(self, img):
        # 寻找最大的色块作为棋盘
        blobs = img.find_blobs(BLACK_PIECE_TH, pixels_threshold=2000, area_threshold=2000, merge=True)
        if blobs:
            # 找到最大的一块
            b = max(blobs, key=lambda x: x.pixels())
            # 获取近似矩形的4个角点
            # 注意：MaixPy 的 blob.corners() 返回的是外接矩形角点，如果棋盘旋转，
            # 需要用 min_area_rect 或凸包逻辑，这里简化处理，假设使用 corners() 配合几何计算
            # 更好的方法是识别棋盘上的四个定位点(如果有)，或者用边缘检测
            
            # 这里演示简单的逻辑：直接画出ROI，实际建议在棋盘贴4个ArUco码或鲜艳色块辅助定位
            # 假设我们通过某种方式(如最大的矩形轮廓)拿到了4个顶点 pts
            # pts = [tl, tr, br, bl] 
            
            # 模拟：简单将画面分割，实际比赛请用 blob.corners() 排序
            w, h = b.w(), b.h()
            x, y = b.x(), b.y()
            
            # 简单的 3x3 均分 (如果不考虑透视，只考虑平移)
            # 如果考虑透视，需要根据角点做透视插值，这里给出计算逻辑：
            # 实际部署建议：在棋盘四角贴红胶带，识别4个红点，最稳。
            return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)] 
        return None

    def calculate_9_grids(self, corners):
        """
        根据4个角点，通过双线性插值计算9个格子的中心点
        corners: [TL, TR, BR, BL]
        """
        # 简化版：直接均分矩形。进阶版需要透视变换公式。
        # 为了应对倾斜，我们使用几何比例分割
        tl, tr, br, bl = corners
        
        points = []
        for row in range(3):
            for col in range(3):
                # 归一化坐标 (0.16, 0.5, 0.83 是 1/6, 3/6, 5/6 处，即格子中心)
                r_ratio = (row * 2 + 1) / 6.0
                c_ratio = (col * 2 + 1) / 6.0
                
                # 左右边上的插值点
                left_x = tl[0] + (bl[0] - tl[0]) * r_ratio
                left_y = tl[1] + (bl[1] - tl[1]) * r_ratio
                
                right_x = tr[0] + (br[0] - tr[0]) * r_ratio
                right_y = tr[1] + (br[1] - tr[1]) * r_ratio
                
                # 最终中心点
                cx = int(left_x + (right_x - left_x) * c_ratio)
                cy = int(left_y + (right_y - left_y) * c_ratio)
                points.append((cx, cy))
        return points

    def check_cell_status(self, img, cx, cy):
        # 在中心点周围截取一个小区域 ROI (如 20x20)
        roi_size = 20
        roi = (cx - roi_size//2, cy - roi_size//2, roi_size, roi_size)
        
        # 统计 ROI 内的颜色
        # 方法1: get_statistics (推荐)
        stats = img.get_statistics(roi=roi)
        l_mean = stats.l_mean()
        
        # 简单阈值判断
        if l_mean < 40: return BLACK
        if l_mean > 70: return WHITE
        return EMPTY

    def logic_loop(self):
        while not app.need_exit():
            img = self.cam.read()
            #img = image.load("棋盘.png")
            # 1. 定位棋盘 (每一帧微调，或第一帧锁定)
            corners = self.find_board_corners(img)
            
            if corners:
                # 绘制棋盘框用于调试
                for i in range(4):
                    img.draw_line(corners[i][0], corners[i][1], corners[(i+1)%4][0], corners[(i+1)%4][1], COLOR_RED, thickness=2)
                
                # 2. 生成9个中心点
                self.grid_points = self.calculate_9_grids(corners)
                
                current_state = [[0]*3 for _ in range(3)]
                
                # 3. 遍历9个格子识别
                for i, (cx, cy) in enumerate(self.grid_points):
                    row, col = i // 3, i % 3
                    status = self.check_cell_status(img, cx, cy)
                    current_state[row][col] = status
                    
                    # 调试显示
                    
                    if status == BLACK: color = (0, 0, 0)
                    if status == WHITE: color = (255, 255, 255)
                    img.draw_circle(cx, cy, 5, COLOR_BLUE, -1)

                # 4. 状态滤波与异常检测
                if current_state == self.prev_matrix:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0
                    self.prev_matrix = [row[:] for row in current_state]
                
                # 5. 发送数据 (稳定5帧后)
                if self.stable_counter == 5:
                    self.send_uart(current_state)
                    # 创新功能：检测输赢
                    winner = self.check_winner(current_state)
                    if winner > 0:
                        img.draw_string(10, 10, f"Winner: {winner}", scale=2, color=COLOR_RED)

            self.disp.show(img)

    def check_winner(self, matrix):
        # 检查行、列、对角线
        lines = matrix + \
                [[matrix[r][c] for r in range(3)] for c in range(3)] + \
                [[matrix[i][i] for i in range(3)], [matrix[i][2-i] for i in range(3)]]
        
        for line in lines:
            if line[0] != 0 and line[0] == line[1] == line[2]:
                return line[0]
        return 0

    def send_uart(self, matrix):
        # 协议头: 0xAA, 0x55
        # 数据: 9字节 (0/1/2)
        # 校验: Sum
        # 协议尾: 0x0D, 0x0A
        packet = bytearray([0xAA, 0x55])
        for r in range(3):
            for c in range(3):
                packet.append(matrix[r][c])
        packet.append(sum(packet[2:]) & 0xFF) # Checksum
        packet.append(0x0D)
        packet.append(0x0A)
        self.uart.write(packet)
        print("Sent:", packet)

if __name__ == "__main__":
    sys = TicTacToeSystem()
    
    sys.logic_loop()