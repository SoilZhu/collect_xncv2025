# Trigame
三子棋装置视觉算法部分，仓库地址：https://github.com/FmCoral/Trigame
最终程序文件为main.py

## 目录
一、视觉识别
二、智能博弈
三、串口通信
四、其他

## 一、视觉识别

### 1、图像采集
- **硬件平台**: MaixPy开发板
- **摄像头参数**: 360×360分辨率，30fps帧率
- **图像格式**: 通过`img = cam.read()`进行采集
- **显示设备**: 通过`disp.show(img)`进行图像显示

### 2、图像处理
#### 颜色空间转换

```python
# 将RGB图像转换为HSV颜色空间
hsv_img = cv.cvtColor(img_cv, cv.COLOR_BGR2HSV)
```

#### 掩码生成
用于识别红色棋盘区域
```python
# 由于通道转换问题，实际识别的是绿色区域
lower_green = np.array([100, 80, 50])    # H:100-130, S≥80, V≥50
upper_green = np.array([130, 255, 255]) # S/V上限拉满
green_mask = cv.inRange(hsv_img, lower_green, upper_green)
```

#### 图像预处理流程
1. **高斯模糊**: `cv.GaussianBlur(green_mask, [3, 3], 0)`
2. **边缘检测**: Canny算法，阈值30-90
3. **形态学操作**: 
   - 膨胀: `cv.dilate(img_edge, kernel)`
   - 腐蚀: `cv.erode(img_dilate, kernel)`

### 3、棋盘识别
#### 轮廓检测与筛选
```python
contours, _ = cv.findContours(img_erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 找出最大轮廓
biggest_contours = max(contours, key=cv.contourArea)
# 多边形近似
epsilon = 0.02 * cv.arcLength(biggest_contours, True)
approx = cv.approxPolyDP(biggest_contours, epsilon, True)
```

#### 角点识别与排序
```python
# 角点排序算法
corn = np.zeros((4, 2), dtype="int")
corn_sum = corners.sum(axis=1)
corn_sub = np.diff(corners, axis=1)
corn[0] = corners[np.argmin(corn_sum)]  # 左上角
corn[1] = corners[np.argmin(corn_sub)]  # 右上角
corn[2] = corners[np.argmax(corn_sum)]  # 右下角
corn[3] = corners[np.argmax(corn_sub)]  # 左下角
```

#### 九宫格中心点计算
```python
def find_center(corners):
    # 使用透视变换计算16个参考点
    center_points = np.array([[0,0], [1/3,0], [2/3,0], [1,0],
                             [0,1/3], [1/3,1/3], [2/3,1/3], [1,1/3],
                             [0,2/3], [1/3,2/3], [2/3,2/3], [1,2/3],
                             [0,1], [1/3,1], [2/3,1], [1,1]])
    
    # 计算每个3×3格子的中心坐标
    for i in range(3):
        for j in range(3):
            left_top = i * 4 + j
            right_top = left_top + 1
            left_bottom = (i + 1) * 4 + j
            right_bottom = left_bottom + 1
            
            # 计算中心点坐标
            x = (points[left_top][0] + points[right_top][0] + 
                 points[left_bottom][0] + points[right_bottom][0]) / 4
            y = (points[left_top][1] + points[right_top][1] + 
                 points[left_bottom][1] + points[right_bottom][1]) / 4
```

### 4、有效数据筛选
- **轮廓筛选**: 只处理包含4个角点的有效轮廓
- **面积筛选**: 选择最大轮廓作为棋盘区域
- **形状验证**: 通过多边形近似验证矩形形状
- **角点验证**: 确保识别到4个有效角点
- **棋盘数据**: 每识别三次棋盘数据，进行一次数据对比，提高识别准确性

## 二、智能博弈

### 1、检查胜负状态
```python
def check_win(current_data):
    # 定义8条赢线
    win_lines = [[1,2,3], [4,5,6], [7,8,9], [1,4,7], 
                 [2,5,8], [3,6,9], [1,5,9], [3,5,7]]
    
    # 返回值说明:
    # 11 = 黑棋赢
    # 22 = 白棋赢  
    # 33 = 平局
    # 0 = 继续游戏
```

#### 棋盘数据结构
- 棋盘数据为18位列表，每2位表示一个格子
- 格式: `[编号, 颜色, 编号, 颜色, ...]`
- 棋子颜色: 0=空, 1=黑棋, 2=白棋

### 2、判断是否遵守规则
```python
def judge_rules(prev_data, curr_data):
    # 规则检查:
    # 1. 只能有一处位置变化
    # 2. 只能落在空位置
    # 3. 不能覆盖已有棋子
    
    # 提取两次棋盘的颜色位值
    prev_color = [prev_data[2*n-1] for n in range(1,10)]
    curr_color = [curr_data[2*n-1] for n in range(1,10)]
```

### 3、核心对弈逻辑
```python
def attack_logic(attack_data):
    # 优先级策略:
    # 1. 直接赢（己方差1子）
    # 2. 堵对方赢（对方差1子）
    # 3. 占中心（格子5）
    # 4. 占角（格子1,3,7,9）
    # 5. 占边（格子2,4,6,8）
```

#### 棋盘映射
```python
# 初始化棋盘映射
board_map = {num: 0 for num in range(1, 10)}
# 转3×3棋盘
board = [
    [board_map[1], board_map[2], board_map[3]],
    [board_map[4], board_map[5], board_map[6]],
    [board_map[7], board_map[8], board_map[9]]
]
```

### 4、正常对局逻辑封装
```python
def main_1(qipan_data):
    # 处理不同数据长度情况:
    # 长度0: 继续下棋
    # 长度1: 检查胜负
    # 长度>1: 判断规则→检查胜负→智能落子
    
    if len(qipan_data) == 0:
        result = 0  # 继续下棋
    elif len(qipan_data) == 1:
        result = check_win(qipan_data[0])
    elif len(qipan_data) > 1:
        rules = judge_rules(qipan_data[-2], qipan_data[-1])
        if rules:
            result = check_win(qipan_data[-1])
            if result == 0:
                best_place = attack_logic(qipan_data[-1])
```

## 三、串口通信

### 1、串口初始化设置
```python
class UartHandler:
    def __init__(self, Pin_1="A18", Pin_2="A19", 
                 Rx="UART1_RX", Tx="UART1_TX", 
                 bitrate=9600, device="/dev/ttyS1"):
    # 引脚映射
    pinmap.set_pin_function(Pin_1, Rx)
    pinmap.set_pin_function(Pin_2, Tx)
    self.serial = uart.UART(device, bitrate)
```

### 2、数据帧格式定义
```python
FRAME_CONFIG = {
    1: (0xA1, 0xAA, 0xA2),  # 发送角度
    2: (0xA1, 0xBB, 0xA2),  # 下在哪里
    3: (0xA1, 0xCC, 0xA2),  # 违规情况
    4: (0xA1, 0xEE, 0xA2),  # 黑棋赢
    5: (0xA1, 0xDD, 0xA2),  # 白棋赢
    6: (0xA1, 0xFF, 0xA2),  # 平局
}
```

### 3、数据转换与打包
```python
def send_data(self, data_list, frame_type):
    # 负数映射: 原始数据 + 50偏移量
    mapped_data = [num + self.offset for num in data_list]
    # 完整帧: [帧头, 数据类型, 数据..., 帧尾]
    header_1, data_type, footer = self.FRAME_CONFIG[frame_type]
    full_frame = [header_1, data_type] + mapped_data + [footer]
```

### 4、数据接收
#### 多线程接收机制
```python
def _recv_loop(self):
    # 后台线程持续接收数据
    while self.running:
        read_byte = self.serial.read(1)
        if read_byte:
            hex_num = ord(read_byte)
            with self.lock:
                self.receive_data.append(hex_num)
```

### 5、数据解析
```python
def get_data(self, clear=True):
    # 从后往前查找最后一个有效帧
    # 步骤1: 从后往前找最后一个有效帧尾
    # 步骤2: 从帧尾往前找对应的帧头
    # 步骤3: 提取完整帧数据
    
    # 提取所有帧头、帧尾及对应关系
    frame_heads = [v[0] for v in self.FRAME_CONFIG.values()]
    frame_tails = [v[1] for v in self.FRAME_CONFIG.values()]
    head_tail_map = dict(zip(frame_heads, frame_tails))
```

## 四、其他

### 退出机制
```python
def exit_program(img):
    # 屏幕右下角显示"Exit"按钮
    # 触摸检测实现程序退出
    
    # 按钮配置
    exit_text, text_scale, padding = "Exit", 3, 8
    # 创建Exit按钮
    text_w, text_h = image.string_size(exit_text, scale=text_scale)
    btn_x = img.width() - text_w - padding * 2
    btn_y = img.height() - text_h - padding * 2
```
