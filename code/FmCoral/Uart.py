from maix import uart, pinmap, time, app
import threading

class UartHandler:
    # 定义帧头帧尾映射（避免硬编码，便于扩展）
    FRAME_CONFIG = {
        1: (0xA1, 0xAA, 0xA2), # 发送角度
        2: (0xA1, 0xBB, 0xA2), # 下在哪里
        3: (0xA1, 0xCC, 0xA2), # 违规情况
        4: (0xA1, 0xFF, 0xA2), # 黑棋赢
        5: (0xA1, 0xFF, 0xA2), # 白棋赢
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

        print(f"串口初始化完成：{device} | 波特率={bitrate}")

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
            # 先清空原缓存（后续仅在未提取到有效帧时放回）
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

        # 步骤4：未找到对应帧头 → 缓存已清空，直接返回空
        if head_pos == -1:
            return []

        # 步骤5：提取完整帧 → 缓存已清空，无需保留剩余数据（核心改动）
        complete_frame = data[head_pos:tail_pos+1]

        return complete_frame

    def send_data(self, data_list, frame_type):
        """
        通用发送方法（提取公共逻辑，避免重复）
        :param data_list: 待发送的原始数据列表
        :param frame_type: 帧类型（1/2/3，对应FRAME_CONFIG中的配置）
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
            print(f"类型{frame_type} | 十六进制：{hex_str} | 字节数：{send_len}")

            return send_len

        except Exception as e:
            print(f"帧类型{frame_type}发送失败：{e}")
            return None

    # 发送接口（函数名完全不变）
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
        """关闭串口（程序结束时调用）"""
        self.running = False  # 停止接收线程
        time.sleep(0.01)      # 等待线程退出
        self.serial.close()
        print("串口已关闭")


if __name__ == "__main__":
    # 初始化串口
    uart_obj = UartHandler()

    while not app.need_exit():
        # 调用get_data解析完整帧
        # uart_obj.send_1([1])
        recv_frame = uart_obj.get_data()
        if recv_frame:
            print(f"完整帧：{recv_frame}")
            # # 提取帧内数据（去掉帧头帧尾）
            # if len(recv_frame) >= 2:
            #     business_data = recv_frame[1:-1]
            #     print(f"帧内数据：{business_data}")
        print("·", end='')
        time.sleep(0.5)

    uart_obj.close()