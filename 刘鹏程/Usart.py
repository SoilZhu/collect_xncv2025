from maix import time
from maix import uart, pinmap

class python_usart:
    qipan_flag = 0 #0表示开始检测
    qizi_flag = 0

    def __init__(self, Pin_1, Pin_2, Rx, Tx, bitrate, device):
        pinmap.set_pin_function(Pin_1, Rx)
        pinmap.set_pin_function(Pin_2, Tx)
        self.device = device
        self.serial = uart.UART(self.device, bitrate)
        self.USART_RX_BUF = [0] * 100
        self.USART_RX_STA = 0
        self.RxState = 0
        self.response_buf = []
        self.ok_received = False

    def usart_receive(self):
        data = self.serial.read()
        if data:
            for byte in data:
                # 帧头检测：0x2c -> 0x12
                if self.RxState == 0 and byte == 0x2c:
                    self.USART_RX_BUF[self.USART_RX_STA] = byte
                    self.RxState = 1
                elif self.RxState == 1 and byte == 0x12:
                    self.USART_RX_STA += 1
                    self.USART_RX_BUF[self.USART_RX_STA] = byte
                    self.RxState = 2
                elif self.RxState == 2:
                    self.USART_RX_STA += 1
                    self.USART_RX_BUF[self.USART_RX_STA] = byte
                    # 帧尾检测：0x5b
                    if self.USART_RX_BUF[self.USART_RX_STA-1] == 0x5b:
                        self.qipan_flag = self.USART_RX_BUF[self.USART_RX_STA-3]
                        self.qizi_flag = self.USART_RX_BUF[self.USART_RX_STA-2]
                        self.RxState = 3
                        time.sleep_ms(10)
                elif self.RxState == 3:
                    self.USART_RX_STA = 0
                    self.USART_RX_BUF = [0] * 100
                    self.RxState = 0
                # 检测OK\r\n（0x4F 0x4B 0x0D 0x0A）
                self.response_buf.append(byte)
                if len(self.response_buf) > 10:
                    self.response_buf.pop(0)
                if len(self.response_buf) >= 4 and self.response_buf[-4:] == [0x4F, 0x4B, 0x0D, 0x0A]:
                    self.ok_received = True
                    self.response_buf = []

    def reset_ok_flag(self):
        self.ok_received = False

    def send_chess_cmd(self, pos, color):
        """发送落子指令给STM32（位置：1-9，颜色：1=白，2=黑）"""
        # 指令格式：[0xF1, 0xF2, 位置, 颜色, 0xF3]
        cmd = bytearray([0xF1, 0xF2, pos, color, 0xF3])
        self.serial.write_str(cmd)

    def send_board_state(self, board):
        """发送完整棋盘状态给STM32"""
        # 数据格式：[0x4B, 0x4C, 9个位置状态, 0x4D]
        data = bytearray([0x4B, 0x4C])
        data.extend(board)
        data.append(0x4D)
        self.serial.write_str(data)

Pin_1 = "A16"
Pin_2 = "A17"
Rx = "UART0_RX"
Tx = "UART0_TX"
device = "/dev/ttyS0"
bitrate = 115200

Usart0 = python_usart(Pin_1, Pin_2, Rx, Tx, bitrate, device)