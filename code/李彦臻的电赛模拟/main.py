import math,cv2
from re import T
from maix import camera,display,image,time,uart,touchscreen,nn,comm
from maix._maix.image import COLOR_BLACK, COLOR_BLUE, COLOR_GRAY, COLOR_GREEN, COLOR_RED, COLOR_WHITE, Color, Format, Line, Rect, Threshold
import numpy as np
#print("cv2的版本号为:",cv2.__version__)

#初始化参数
threshold1=[[4,25,-11,9,-11,9]]#找黑线
threshold11=[[0,15,0,15,-5,5]]#实际黑
threshold2=[[71,85,-21,18,-21,18]]#白块色
threshold3=[[40,71,18,65,0,35]]#红色
CANNY_THRESH = (20, 200)#50,150  # Canny 的低阈值与高阈值（阈值范围依图像位深而异，常用(50,150) / (100,200)）
disp=display.Display()
cam=camera.Camera()
ts = touchscreen.TouchScreen()
#img=image.load("/root/project-1/棋盘.png",image.Format.FMT_RGB888)
#img=image.load("/root/project-1/棋盘旋转45°.png",image.Format.FMT_RGB888)
img=image.load("/root/project-1/棋盘（任务一黑）.png",image.Format.FMT_RGB888)
img=img.resize(640,640)

#串口设置
UART_DEV = "/dev/ttyS0" 
BAUDRATE = 115200
uart = uart.UART(UART_DEV, baudrate=BAUDRATE)

#外方法
def maixpy_to_opencv(maix_image):#假设传入的图像格式都为FMT_RGB888
    # 获取图像尺寸和数据
    byte_data = maix_image.to_bytes() 
    width = maix_image.width()
    height = maix_image.height()
    channels=3
    #print("在“maixpy_to_opencv”中：",width,height)
    # 转换为NumPy数组
    img_array = np.frombuffer(byte_data, dtype=np.uint8)#frombuffer方法的主要功能是将缓冲区数据（如字节数据）直接转换为NumPy数组
    img_array = img_array.reshape((height, width, channels))
    if channels >= 3:
        img_array = img_array[:, :,::-1 ]
    #cv2.imshow("Image", img_array)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
    return img_array
def opencv_to_maixpy(opencv_image):
    # 判断灰、彩
    if len(opencv_image.shape) == 2: 
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
    else:
        opencv_image = opencv_image[:, :,::-1 ]
    height, width, channels = opencv_image.shape  
    # 加载为MaixPy Image对象
    byte_data = opencv_image.tobytes()# 转换
    #return image.load(byte_data,format=image.Format.FMT_RGB888)#,size=(width, height)传入报错
    #return image.Image(width, height, format=Format.FMT_RGB888, data=byte_data)
    return image.cv2image(opencv_image)#上帝，终于找到你了
#奶奶的，写了半天发现有cv2image和imagecv2
def UART_deliver(byte1, byte2, byte3):#0~2
    data_to_send = bytes([byte1, byte2, byte3])
    #print("data_to_send:",data_to_send)
    try:
        uart.write(data_to_send)
    except:
        print("传输错误，请检查")
def judge_win(board):

        '''
        #检查工作
        if not isinstance(board, list) or len(board) != 9:
            return "invalid"
        if any(not isinstance(x, int) or x < 0 or x > 2 for x in board):#第二个没必要
            return "invalid"
        black_count = board.count(1)
        white_count = board.count(2)
        if black_count < white_count or black_count > white_count + 1:#黑棋比白棋多一个或者相等
            return "invalid"
        '''
        # 输赢平判断
        win_lines = [
            # 横三行
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            # 竖三列
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            # 两条对角线
            [0, 4, 8], [2, 4, 6]
        ]
        # 检查胜利情况
        black_wins = False
        white_wins = False
        for line in win_lines:
            values = [board[pos] for pos in line]
            if values[0] == values[1] == values[2] == 1:
                black_wins = True
            if values[0] == values[1] == values[2] == 2:
                white_wins = True
        if black_wins:
            return "black_win"
        if white_wins:
            return "white_win"
        '''
        关于平局的情况：我估计这里跟你的算法有关，由于我的基础算法机制是优先制的，所以有些情况是不可能出现的。
        人脑来下可以下到满格，（中心格子一定会在前两步下好）也就是黑棋比白棋多一步，但是如果下一步也是平局的话，
        那么就要提前终止，后面可以预先下了之后不去动棋子解决。还有就是7个棋子也可以平局
        '''
        if board.count(0) == 0:  # 棋盘已满
            return "draw" #draw原来也有平局的意思
        return "continue"
def in_range(rect_lab,lab_range):
    l_min, l_max, a_min, a_max, b_min, b_max = rect_lab[:6]
    return (
            l_min >= lab_range[0] and l_max <= lab_range[1] and
            a_min >= lab_range[2] and a_max <= lab_range[3] and
            b_min >= lab_range[4] and b_max <= lab_range[5])     

class three_chess:
    def __init__(self):
        #self.cam = camera.Camera()
        #self.uart = uart.UART(UART_DEV, baudrate=BAUDRATE)
        #self.img=image.Image
        self.coord=[]#动态更新棋盘坐标roi
        self.Sudoku=[]#存九宫格中心坐标
        self.edges=[]#直线坐标
        self.Sudoku_lab=[]#通过find_chess1/2联动find_roi_lab得到的各个格子的状态
        self.Sudoku_state=[0,0,0,0,0,0,0,0,0]
        self.detector1 = nn.YOLOv5("/root/project-1/model_246778.mud")
        self.detector2 = nn.YOLOv5("/root/train_model2/model_247058.mud")
    #def UART_deliver(self):
    
    def PID_control(self,img,goal_x,goal_y):#****这里可能还需要调整一下，传入图片改为传入坐标
        img_x,img_y=img.width()//2,img.height()//2
        TOLERANCE=img.width()//50
        for i in range(0,2):
            start.find_chess(img)#这里调用了查找
        while True:
            if abs(img_x - goal_x) <= TOLERANCE and abs(img_y - goal_y) <= TOLERANCE:
                print("已经定位在目标范围内")
                return False#*******
                break
            else:
                # X 轴判断
                if img_x < goal_x - TOLERANCE:
                    # 图像中心在目标点的左侧，向右移动
                    action_x = "Move_X_Right"
                    UART_deliver(0,0,1)
                    uart.read(kwargs=-1)
                elif img_x > goal_x + TOLERANCE:
                    # 图像中心在目标点的右侧，向左移动
                    action_x = "Move_X_Left"
                    UART_deliver(0,0,2)
                    uart.read(kwargs=-1)
                # Y 轴判断
                if img_y < goal_y - TOLERANCE:
                    # 图像中心在目标点的上方，向下移动
                    UART_deliver(0,0,3)
                    uart.read(kwargs=-1)
                elif img_y > goal_y + TOLERANCE:
                    # 图像中心在目标点的下方，向上移动
                    UART_deliver(0,0,4)
                    uart.read(kwargs=-1)
                print("hello")
                return True #联动后记得取消，不然循环不起来*******

    def find_chess(self,img):#最基础的找棋盘
        #试试find_llllines2中的图像处理方法，但是调用此方法后测试图片都无法检测出来
        # cv_img = image.image2cv(img)
        # blurred = cv2.GaussianBlur(cv_img, (5, 5), 1)# 高斯去噪：使用5x5的高斯核，标准差为1
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)# 转换为灰度图
        # edges_cv = cv2.Canny(gray, CANNY_THRESH[0], CANNY_THRESH[1])#canny边缘检测
        # img = opencv_to_maixpy(edges_cv)
        # del cv_img,blurred,gray,edges_cv
        #disp.show(img)
        time.sleep(1)
        #img=cam.read()
        #blobs=img.find_blobs(threshold1,pixels_threshold=20000,area_threshold=150000)
        #忽略了一个问题，此分辨率是针对1024*1024的，正常不是这么大，后续可以考虑做一个判断来选择阈值
        #blobs=img.find_blobs(threshold1,pixels_threshold=20000,area_threshold=150000)
        blobs=img.find_blobs(threshold3,pixels_threshold=800,area_threshold=15000)#320*240
        #blobs=img.find_blobs(threshold1,pixels_threshold=3000,area_threshold=60000)#640*480
        if blobs:
            for blob in blobs:
                self.coord=[blob.x(),blob.y(),blob.w(),blob.h()]
                self.Sudoku=[
                            blob.x()+blob.w()//6,blob.y()+blob.h()//6,
                            blob.x()+blob.w()*3//6,blob.y()+blob.h()//6,
                            blob.x()+blob.w()*5//6,blob.y()+blob.h()//6,
            
                            blob.x()+blob.w()//6,blob.y()+blob.h()*3//6,
                            blob.x()+blob.w()*3//6,blob.y()+blob.h()*3//6,
                            blob.x()+blob.w()*5//6,blob.y()+blob.h()*3//6,
        
                            blob.x()+blob.w()//6,blob.y()+blob.h()*5//6,
                            blob.x()+blob.w()*3//6,blob.y()+blob.h()*5//6,
                            blob.x()+blob.w()*5//6,blob.y()+blob.h()*5//6]
                img.draw_rect(blob.x(),blob.y(),blob.w(),blob.h(),color=COLOR_GREEN,thickness=3)
            #img.draw_keypoints(self.Sudoku,COLOR_RED)
            point_index = 1
            Sudoku_lab=[]
            for i in range(0, len(self.Sudoku), 2):#步进2
                x = self.Sudoku[i]
                y = self.Sudoku[i + 1]
                x_rect=x-img.width()//44
                y_rect=y-img.height()//44
                Sudoku_lab.append(start.find_roi_lab(img,x,y))
                img.draw_rect(x_rect,y_rect,img.width()//22,img.height()//22,image.Color.from_rgb(0,0,220),thickness=2)
                img.draw_string(x + 12, y + 12, f"{point_index}", COLOR_BLUE, scale=1)#这个scale向是像素的绝对值大小，不随分辨率改变而改变字体
                point_index += 1
            #print("找到边框")
            #print(self.coord)
            #print("Sudoku_lab:",Sudoku_lab)
            self.Sudoku_lab=Sudoku_lab
            #disp.show(img)
            #time.sleep(1)
        else:
            print("未找到边框!")
        return img

    def find_chess2(self, img):#用find_chess找不出，没招了，只能用模型试试了。发现可以通过模糊镜像来达到去除红块中杂质部分，以达到寻找色块的部分。
        try:
            objs = self.detector2.detect(img, conf_th=0.5)
        except Exception as e:
            print("[FIND_CHESS2] 模型推理失败:", e)
            return img
        if not objs:
            print("[FIND_CHESS2] 未检测到棋盘")
            return img
        for obj in objs:
            x , y , w , h = obj.x ,obj.y,obj.w,obj.h
            self.coord = [x, y, w, h]
            #img.draw_rect(x, y, w, h, color=COLOR_GREEN, thickness=3)
            self.Sudoku = [
                x + w // 6,     y + h // 6,
                x + w * 3 // 6, y + h // 6,
                x + w * 5 // 6, y + h // 6,

                x + w // 6,     y + h * 3 // 6,
                x + w * 3 // 6, y + h * 3 // 6,
                x + w * 5 // 6, y + h * 3 // 6,

                x + w // 6,     y + h * 5 // 6,
                x + w * 3 // 6, y + h * 5 // 6,
                x + w * 5 // 6, y + h * 5 // 6
            ]
        Sudoku_lab=[]
        point_index = 1
        for i in range(0, len(self.Sudoku), 2):
            cx = self.Sudoku[i]
            cy = self.Sudoku[i + 1]
            x_rect=cx-img.width()//44
            y_rect=cy-img.height()//44
            Sudoku_lab.append(start.find_roi_lab(img,cx,cy))
            img.draw_rect(x_rect,y_rect,img.width()//22,img.height()//22,image.Color.from_rgb(0, 0, 220),thickness=2)
            img.draw_string(cx + 8,cy + 8,str(point_index),COLOR_BLUE,scale=1)
            point_index += 1
        print("star.Sudoku_lab:",start.Sudoku_lab)
        self.Sudoku_lab=Sudoku_lab
        return img

    def find_roi_lab(self,j_img,j_x,j_y):#查找图像区域的lab值,目前对于所有九宫格的检测仅仅出现在find_chess2中
        RED_LAB =    [30, 70, 15, 60,  5, 60]
        BLACK_LAB =  [ 0, 40, -10, 10, -10, 10]
        WHITE_LAB =  [65,100, -10, 10, -10, 10]
        x_rect=j_x-j_img.width()//40
        y_rect=j_y-j_img.height()//40
        #rect_lbp=j_img.find_lbp(roi=[x_rect,y_rect,j_img.width()//20,j_img.height()//20])#没注意文档上写着此功能暂不支持.犯了一个概念理解错误致命错误
        rect_lab_tect=img.get_statistics(roi=[x_rect,y_rect,j_img.width()//20,j_img.height()//20])#此代码官方网站上不是还没实现吗，怎么还可以调用
        rect_lab=[
                rect_lab_tect.l_min(),rect_lab_tect.l_max(),
                rect_lab_tect.a_min(),rect_lab_tect.a_max(),
                rect_lab_tect.b_min(),rect_lab_tect.b_max(),
                rect_lab_tect.l_mean(),rect_lab_tect.a_mean(),rect_lab_tect.b_mean()]
        #print("rect_lbp:",rect_lab)    
        # for lab in rect_lab:
        #     print("rect_lbp:",lab)
        #return rect_lab
        if in_range(rect_lab,BLACK_LAB):
            return 1
        if in_range(rect_lab,WHITE_LAB):
            return 2
        if in_range(rect_lab,RED_LAB):
            return 0
        return 0

    def find_llllines2(self,img):#寄寄，不要了
        """
        使用：准确检测棋盘的主边框直线（2~4条）
        输入： img -> maix.image.Image 对象（RGB 或 GRAYSCALE）
        输出： 返回筛选后的直线对象列表（最多 4 条），并在原始 img 上以红色绘制这些直线（用于调试）
        注意：函数尽量兼容不同 MaixPy 版本；若某些滤波 API 不存在会回退到其他方法。
        """

        # ---------------------------
        # 参数（可调）
        # ---------------------------
        HOUGH_THRESH = 1500 #find_lines的阈值
        THETA_MARGIN = 2   # 角度合并阈值（度）
        RHO_MARGIN = img.height()//100     # rho 合并阈值（像素）
        #print("RHO_MARGIN",RHO_MARGIN)
        ANGLE_TOLERANCE = 12# 判断是否为“水平 / 垂直”直线的角度容差（单位：度）
        MAX_LINES = 4# 我们最终返回的最大直线数

        # ---------------------------
        # 预处理
        # ---------------------------
        # 格式转换
        #cv_img = maixpy_to_opencv(img)
        cv_img = image.image2cv(img)#奶奶的，结果一摸一样
        blurred = cv2.GaussianBlur(cv_img, (5, 5), 1)# 高斯去噪：使用5x5的高斯核，标准差为1
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)# 转换为灰度图
        edges_cv = cv2.Canny(gray, CANNY_THRESH[0], CANNY_THRESH[1])#canny边缘检测
        
        #测试一下几个cv2的查找方法：
        #edges = gray.find_edges(threshold=CANNY_THRESH)这个是maixpy自带的边缘检测函数
        lines_cv=cv2.HoughLinesP(image=edges_cv,rho=RHO_MARGIN,theta=THETA_MARGIN,threshold=100)#可惜还是只能找到正的
        #circle_cv=cv2.HoughCircles(image=edges_cv,method=cv2.HOUGH_GRADIENT,dp=1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
        #print("lines_cv",lines_cv)
        
        # 转回去
        edges = opencv_to_maixpy(edges_cv)
        disp.show(img)
        del cv_img, blurred, gray, edges_cv
        print("使用OpenCV完成图像预处理")
        
        try:
            for l1 in lines_cv:
                for l in l1:
                    edges.draw_line(l[0],l[1],l[2],l[3],color=image.Color.from_rgb(255,0,0),thickness=5)
            disp.show(edges)
            time.sleep(1)
            del lines_cv
        except Exception:
            print("未找到霍夫线")
        
        
        # ---------------------------
        # 找线（直线——>线段）
        # ---------------------------
        try:#原来try这么好用
            #raw_lines = edges.find_lines(threshold=HOUGH_THRESH,theta_margin=THETA_MARGIN,rho_margin=RHO_MARGIN)
            #wdf,找了半天发现查找线段有个专门的方法find_line_segments
            raw_lines = edges.find_line_segments(max_theta_difference=THETA_MARGIN,merge_distance=RHO_MARGIN)
            #raw_lines = edges.search_line_path()#wdf,新大陆
            # for j, line in enumerate(raw_lines[:5]):  # 只看前5条
            #     x1, y1, x2, y2 = line.lines()
            #     print(f"  线段 {j}: ({x1},{y1}) -> ({x2},{y2}), theta={line.theta()}")
            # for lg in raw_lines:
            #     print("找到的Linegroup参数：",lg.id)
        except Exception:   
            print("没有找到直线")
            raw_lines = []
        del edges
        
        # 每个 line 对象提供：line() -> (x1,y1,x2,y2)， length(), theta(), rho(), magnitude() 等

        '''
        #find_lines的筛选添加绘制直线
        # ---------------------------
        # 直线筛选
        # ---------------------------
        candidates = []
        for l in raw_lines:
            length = l.length()    # 线段长度（像素）
            theta = l.theta()      # 角度 0 - 179（度）
            rho = l.rho()          # rho 值
            # 判断水平：theta 接近 0 或接近 180；判断垂直：theta 接近 90
            is_horizontal = (theta <= ANGLE_TOLERANCE) or (theta >= (180 - ANGLE_TOLERANCE))
            is_vertical = (abs(theta - 90) <= ANGLE_TOLERANCE)
            if is_horizontal or is_vertical:
                # 不是主要方向线，跳过
                continue
            #去掉线段短的直线
            min_length = min(img.width(), img.height()) * 0.20  # 最少为图像短边的 20%
            if length < min_length:
                continue
            # 保存候选
            candidates.append({"line_obj": l,"length": length,"theta": theta,"rho": rho,
            })#"is_horizontal": is_horizontal,"is_vertical": is_vertical
        if not candidates:
            print("未找到直线！")
            return []
        #print("candidates:",candidates)
        # 按长度降序
        candidates = sorted(candidates, key=lambda x: x["length"], reverse=True)
        # 合并近似重复检测：按（theta,rho）聚类，保留每簇中最长的那条
        selected = []
        for c in candidates:
            l_theta = c["theta"]
            l_rho = c["rho"]
            duplicate = False
            for s in selected:
                # 若角度与 rho 都非常接近，则认为是同一条线（重复检测）
                if abs(s["theta"] - l_theta) <= THETA_MARGIN and abs(s["rho"] - l_rho) <= RHO_MARGIN:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(c)
            if len(selected) >= MAX_LINES:
                break
        # 最终直线对象列表（保留 line 对象）
        final_lines = [c["line_obj"] for c in selected]
        final_lines=[c["line_obj"]for c in candidates]
        del candidates
        '''
        
        #find_line_segments的添加绘制线段
        final_lines=[c for c in raw_lines]
        if final_lines:
                print("final_lines列表为：",final_lines)
        else:
            print("未找到final_lines列表")
        for l in final_lines:
            lines=[l[0],l[1],l[2],l[3]]
            img.draw_line(*lines, color=image.Color.from_rgb(255, 0, 0), thickness=4)
        print("final_lines:",final_lines)
        self.edges=final_lines
        '''
        #search_line_path的添加绘制线段
        for lg in raw_lines:
            final_lines=[c for c in lg.line()]
            if final_lines:
                print("final_lines列表为：",final_lines)
            else:
                print("未找到final_lines列表")
            for l in final_lines:
                lines=[l[0],l[1],l[2],l[3]]
                img.draw_line(*lines, color=image.Color.from_rgb(255, 0, 0), thickness=4)
            print("final_lines:",final_lines)
            self.edges=final_lines
        '''

        # ---------------------------
        # 在原始图像上绘制筛选后的直线，便于调试
        # ---------------------------
        # for l in final_lines:
        #     lines=[l[0],l[1],l[2],l[3]]
        #     img.draw_line(*lines, color=image.Color.from_rgb(255, 0, 0), thickness=4)
        

        return self.edges

    def find_llllines(self,img):#这个难解决,暂未解决
        img_line = img.copy()  
        #ai给出的图像操作，但是还不知道如何操作：
        #去噪
        #超黑边框 → 白线,反转色彩

        rois = [
            (0, 0, img_line.width()//3, img_line.height()),      # 左
            (0, 0, img_line.width(), img_line.height()//3),      # 上
            (2*img_line.width()//3, 0, img_line.width()//3, img_line.height()),  # 右
            (0, 2*img_line.height()//3, img_line.width(), img_line.height()//3)  # 下
            ]

        edges = []
        for i, roi in enumerate(rois):
            print("i为{}roi为{}".format(i,roi))
            line = img_line.get_regression(threshold1, roi=roi, pixels_threshold=60)
            if line:
                for a in line:
                    x1, y1, x2, y2 =a[0],a[1],a[2],a[3]
                    theta = a.theta()
                    if theta > 90: theta -= 180
                    theta = abs(theta)
                    edges.append((x1,y1,x2,y2, theta))
                    img_line.draw_line(x1,y1,x2,y2, image.Color.from_rgb(0,255,255), thickness=8)
                    img_line.draw_string(10, 30 + i*30, f"Edge {i+1}: {theta:.1f}deg", image.Color.from_rgb(0,255,255), scale=2)

        disp.show(img_line)
        time.sleep(2)
        self.edges=edges
        print(self.edges)
        del img_line
        return self.edges       

    def detect_chess_centers(self, img):
        results = []
        #print("len(results):",len(results))
        #while len(results) != 0:#这样避免未检测棋子，但是只适用于机器白棋后手，为啥不跑后面的程序
        try:
            objects = self.detector1.detect(img)
        except Exception as e:
            print("[DETECT] 模型推理失败:", e)
            return results
        for obj in objects:
            try:
                x , y , w , h = obj.x ,obj.y,obj.w,obj.h
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                cid=self.detector1.labels[obj.class_id]
                #print("cid:",cid)
                id=0
                if cid == "black":
                    id = 1
                if cid == "white":
                    id = 2
                results.append([cx, cy, id])
                
            except Exception as e:
                # 单个目标解析失败不影响整体
                print("[DETECT] 单个目标解析失败:", e)
                continue
        # print("走到这了")
        if results==0:
            return None
        else:
            print("results:",results)
            return results
        
    def judge_win(self,board):#传入九个状态，平局判断还需要完善一下
        '''
        #检查工作
        if not isinstance(board, list) or len(board) != 9:
            return "invalid"
        if any(not isinstance(x, int) or x < 0 or x > 2 for x in board):#第二个没必要
            return "invalid"
        black_count = board.count(1)
        white_count = board.count(2)
        if black_count < white_count or black_count > white_count + 1:#黑棋比白棋多一个或者相等
            return "invalid"
        '''
        # 输赢平判断
        win_lines = [
            # 横三行
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            # 竖三列
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            # 两条对角线
            [0, 4, 8], [2, 4, 6]
        ]
        # 检查胜利情况
        black_wins = False
        white_wins = False
        for line in win_lines:
            values = [board[pos] for pos in line]
            if values[0] == values[1] == values[2] == 1:
                black_wins = True
            if values[0] == values[1] == values[2] == 2:
                white_wins = True
        if black_wins:
            print("black_win")
            return "black_win"
        if white_wins:
            print("white_win")
            return "white_win"
        '''
        关于平局的情况：我估计这里跟你的算法有关，由于我的基础算法机制是优先制的，所以有些情况是不可能出现的。
        人脑来下可以下到满格，（中心格子一定会在前两步下好）也就是黑棋比白棋多一步，但是如果下一步也是平局的话，
        那么就要提前终止，后面可以预先下了之后不去动棋子解决。还有就是7个棋子也可以平局
        '''
        if board.count(0) == 0:  # 棋盘已满
            print("draw")
            return "draw" #draw原来也有平局的意思
        return "continue"

    def basic_suanfa(self,board, player):

        empty_positions = [i for i, v in enumerate(board,1) if v == 0]
        print("empty_positions",empty_positions)
        # 判断三子连线的情况
        for pos in empty_positions:
            board[pos-1] = player#这个pos-1真是害人（一定要注意自己定义的1~9）
            black_positions = [i for i, v in enumerate(board,1) if v == 1]
            white_positions = [i for i, v in enumerate(board,1) if v == 2]
            if black_positions < white_positions :
                return True    
            #print("在这！")
            if start.judge_win(board) == "black_win" and player == 1:
                UART_deliver(player,pos,0)
                uart.read(kwargs=-1)
                print("pos1:",pos)
                return False
            if start.judge_win(board) == "white_win" and player == 2:
                UART_deliver(player,pos,0)
                uart.read(kwargs=-1)
                print("pos2:",pos)
                return False
            if start.judge_win(board) == "draw":
                print("draw")
                return False
            #阻止但继续
            if start.judge_win(board) == "white_win" and player == 1:
                UART_deliver(player,pos,0)
                uart.read(kwargs=-1)
                print("pos3:",pos)
                return True
            if start.judge_win(board) == "black_win" and player == 2:
                UART_deliver(player,pos,0)
                uart.read(kwargs=-1)
                print("pos4:",pos)
                return True
            board[pos-1] = 0
        print("1")
        # 再中心 > 角点 > 边点
        priority_order = [5, 1, 7, 3, 9, 2, 4, 6, 8]
        for pos in priority_order:
            if pos in empty_positions:
                board[pos-1]=player
                UART_deliver(player,pos,0)
                print("下在了：",pos)
                return True
        
        return False  # 无位置可下（平局）

    def read_touch_click(self, ts):
        """
        触摸“点击事件”读取：
        只有在【按下 -> 抬起】这一瞬间返回一次 clicked=True
        """
        x, y, pressed = ts.read()

        if not hasattr(self, "_last_pressed"):
            self._last_pressed = False

        clicked = False
        if self._last_pressed and not pressed:
            clicked = True

        self._last_pressed = pressed
        return x, y, clicked

    def task_1(self):#一    

        #target_x = self.Sudoku[12]
        #target_y = self.Sudoku[13]
        #创建 320×240 的任务一的背景
        img = image.Image(320, 240)
        img.clear()
        img.draw_rect(0, 0, 320, 240, image.Color.from_rgb(0, 0, 0),thickness=-1)   # 黑底
        #img.draw_string(30, 20,"请选择要放置的棋子颜色",image.Color.from_rgb(255, 255, 255),scale=1,thickness=2)
        center_y = 130          # 圆心垂直位置
        radius   = 50           # 圆半径
        # 左侧：白棋按钮（实心白色圆 + 黑色边框）
        white_cx = 100
        img.draw_circle(white_cx, center_y, radius, image.Color.from_rgb(255, 255, 255), thickness=-1)  # 填充白
        img.draw_circle(white_cx, center_y, radius, image.Color.from_rgb(200, 200, 200), thickness=5)   # 灰边
        #img.draw_string(white_cx - 20, center_y + 60, "白棋", image.Color.from_rgb(255, 255, 255), scale=1)
        # 右侧：黑棋按钮（实心黑色圆 + 白色边框）
        black_cx = 220
        img.draw_circle(black_cx, center_y, radius, image.Color.from_rgb(0, 0, 0), thickness=-1)        # 填充黑
        img.draw_circle(black_cx, center_y, radius, image.Color.from_rgb(255, 255, 255), thickness=5)   # 白边
        #img.draw_string(black_cx - 20, center_y + 60, "黑棋", image.Color.from_rgb(255, 255, 255), scale=1)

        # 初始化触摸屏
        ts = touchscreen.TouchScreen()
        # 3.触摸检测循环：等待用户选择（实时更新显示）
        selected_color = None  # 初始未选中
        last_x, last_y, last_pressed = -1, -1, False  # 用于防抖（避免重复触发）
        print("[TOUCH] 等待触摸选择颜色...")
        while selected_color is None:
            # 读取触摸事件
            x, y, clicked = self.read_touch_click(ts)

            # 防抖：仅处理变化事件
            if (x != last_x or y != last_y or clicked != last_pressed):
                # 清空上一帧触摸点（可选，反馈用）
                img.draw_rect(0, 0, 320, 240, image.Color.from_rgb(0, 0, 0), thickness=-1)  # 刷新背景
                # 重新绘制固定 UI（标题 + 按钮）
                #img.draw_string(30, 20, "请选择要放置的棋子颜色", image.Color.from_rgb(255, 255, 255), scale=2.2, thickness=2)
                img.draw_circle(white_cx, center_y, radius, image.Color.from_rgb(255, 255, 255), thickness=-1)
                img.draw_circle(white_cx, center_y, radius, image.Color.from_rgb(200, 200, 200), thickness=5)
                #img.draw_string(white_cx - 20, center_y + 60, "白棋", image.Color.from_rgb(255, 255, 255), scale=2)
                img.draw_circle(black_cx, center_y, radius, image.Color.from_rgb(0, 0, 0), thickness=-1)
                img.draw_circle(black_cx, center_y, radius, image.Color.from_rgb(255, 255, 255), thickness=5)
                #img.draw_string(black_cx - 20, center_y + 60, "黑棋", image.Color.from_rgb(255, 255, 255), scale=2)

                if clicked:  # 仅在按下时判断
                    # 计算距离判断是否在白棋圆内（sqrt((x-cx)^2 + (y-cy)^2) < radius）
                    dist_white = math.sqrt((x - white_cx)**2 + (y - center_y)**2)
                    dist_black = math.sqrt((x - black_cx)**2 + (y - center_y)**2)

                    if dist_white < radius:
                        selected_color = "white"
                        # 高亮白棋（绿色外圈）
                        img.draw_circle(white_cx, center_y, radius + 12, image.Color.from_rgb(0, 255, 0), thickness=6)
                        #img.draw_string(50, 210, "已选择：白棋 → 即将放置到 7 号格", image.Color.from_rgb(0, 255, 0), scale=1.6)
                        print("[TOUCH] 选中白棋")
                        UART_deliver(2,7,0)

                    elif dist_black < radius:
                        selected_color = "black"
                        # 高亮黑棋
                        img.draw_circle(black_cx, center_y, radius + 12, image.Color.from_rgb(0, 255, 0), thickness=6)
                        #img.draw_string(50, 210, "已选择：黑棋 → 即将放置到 7 号格", image.Color.from_rgb(0, 255, 0), scale=1.6)
                        print("[TOUCH] 选中黑棋")
                        UART_deliver(1,7,0)

                    # 绘制当前触摸点（反馈，白色小圈）
                    if 0 <= x < 320 and 0 <= y < 240:
                        img.draw_circle(x, y, 3, image.Color.from_rgb(255, 255, 255), thickness=2)

                last_x, last_y, last_pressed = x, y, clicked

            # 显示更新
            disp.show(img)
            time.sleep_ms(50)  # 50ms 循环，响应快且省 CPU
        disp.show(img)
        time.sleep(2)

    def task_2(self):#二
        
        ts = touchscreen.TouchScreen()
        task2_list = []
        used_grids = set()   # 已使用的格子编号
        img = image.Image(320, 240)
        # -----------------------------
        # UI 固定参数
        # -----------------------------
        center_y = 130
        radius = 45
        black_cx = 90
        white_cx = 230
        print("[TASK2] 开始输入 5 组 黑/白 + 格子编号")
        for i in range(5):

            selected_color = None
            selected_grid = None
            # =============================
            # Step 1：选择黑 / 白
            # =============================
            last_x, last_y, last_pressed = 0, 0, False
            while selected_color is None:
                img.clear()
                img.draw_rect(0, 0, 320, 240, image.Color.from_rgb(0, 0, 0), thickness=-1)
                #img.draw_string(40, 20, "第 {} 次：选择棋子颜色".format(i + 1),image.Color.from_rgb(255, 255, 255), scale=2)
                # 黑棋
                img.draw_circle(black_cx, center_y, radius,image.Color.from_rgb(0, 0, 0), thickness=-1)
                img.draw_circle(black_cx, center_y, radius,image.Color.from_rgb(255, 255, 255), thickness=4)
                #img.draw_string(black_cx - 20, center_y + 55, "黑棋",image.Color.from_rgb(255, 255, 255), scale=1.6)
                # 白棋
                img.draw_circle(white_cx, center_y, radius,image.Color.from_rgb(255, 255, 255), thickness=-1)
                img.draw_circle(white_cx, center_y, radius,image.Color.from_rgb(180, 180, 180), thickness=4)
                #img.draw_string(white_cx - 20, center_y + 55, "白棋",image.Color.from_rgb(255, 255, 255), scale=1.6)
                x, y, clicked = self.read_touch_click(ts)
                img.draw_keypoints([x,y],image.Color.from_rgb(255,0,0))
                disp.show(img)
                if clicked and (x != last_x or y != last_y):
                    if math.sqrt((x - black_cx)**2 + (y - center_y)**2) < radius:
                        selected_color = 0
                        print("[TASK2] 选择黑棋")
                    elif math.sqrt((x - white_cx)**2 + (y - center_y)**2) < radius:
                        selected_color = 1
                        print("[TASK2] 选择白棋")
                    last_x, last_y = x, y
                disp.show(img)
                time.sleep_ms(50)
            # =============================
            # Step 2：选择九宫格编号
            # =============================
            last_x, last_y, last_pressed = 0, 0, False
            while selected_grid is None:
                img.clear()
                img.draw_rect(0, 0, 320, 240, image.Color.from_rgb(0, 0, 0), thickness=-1)
                #img.draw_string(20, 10, "选择放置格子 (1~9)",image.Color.from_rgb(255, 255, 255), scale=2)
                # 九宫格绘制参数
                start_x = 40
                start_y = 50
                cell = 70
                for r in range(3):
                    for c in range(3):
                        idx = r * 3 + c + 1
                        x0 = start_x + c * cell
                        y0 = start_y + r * cell
                        color = image.Color.from_rgb(80, 80, 80)
                        if idx in used_grids:
                            color = image.Color.from_rgb(40, 40, 40)
                        img.draw_rect(x0, y0, cell, cell, color, thickness=2)
                        img.draw_string(x0 + 25, y0 + 20, str(idx),image.Color.from_rgb(255, 255, 255), scale=1)
                x, y, clicked = self.read_touch_click(ts)
                img.draw_keypoints([x,y],image.Color.from_rgb(255,0,0))
                disp.show(img)
                if clicked and (x != last_x or y != last_y):
                    for r in range(3):
                        for c in range(3):
                            idx = r * 3 + c + 1
                            x0 = start_x + c * cell
                            y0 = start_y + r * cell
                            if (x0 < x < x0 + cell) and (y0 < y < y0 + cell):
                                if idx not in used_grids:
                                    selected_grid = idx
                                    used_grids.add(idx)
                                    print("[TASK2] 选择格子:", idx)
                                else:
                                    print("[TASK2] 格子已被使用")

                    last_x, last_y = x, y
                disp.show(img)
                time.sleep_ms(50)

            # =============================
            # 保存本次结果
            # =============================
            task2_list.append((selected_color+1, selected_grid))
            print("[TASK2] 当前结果:", task2_list)
            time.sleep_ms(300)
        for i,j in task2_list:
                #print("i,j",i,j)
                UART_deliver(i,j,0)
                uart.read()
                time.sleep_ms(100)
                start.PID_control(img,i,j)
                uart.read()

        print("[TASK2] 输入完成:", task2_list)
        return task2_list

    def task_3(self):#三（丢掉）
        return True

    def task_4(self):#四五六
        '''
        先找一个初始图像作为参照，点击此任务后即开始此步骤，先初始归位、再找棋盘、再定位。
        完成后正式开始下棋，做一个交互界面，（注意到题目要求执白棋，因此不需要考虑黑棋），人下好后就点击继续，此循环包裹整个对弈过程。
        '''
        #这个find_chess和find_detect_chess_centers都不能第一次找到，需要通过for i in range(0,2):至少循环两次找到
        UART_deliver(0,0,0)
        uart.read(kwargs=-1)
        #***因该还有一个接受函数，在接收后再继续执行
        img=cam.read()
        tol=img.width()//100#初始化一下误差
        result_temp=None
        #start.find_chess(img)下面函数开始就会有的
        start.PID_control(img,self.Sudoku[8],self.Sudoku[9])
        True_or_Flase=True
        x,y,press=0,0,False        
        while True_or_Flase:
            for i in range(0,2):
                img=cam.read()
                start.find_chess(img)
                result_temp=start.detect_chess_centers(img)
            if result_temp !=None:
                for cx, cy, chess_id in result_temp:
                    for i in range(9):#找九宫格
                        gx = self.Sudoku[i * 2]
                        gy = self.Sudoku[i * 2 + 1]
                        # 判断是否落在该格（矩形误差判断，最快最稳）
                        if abs(cx - gx) <= tol and abs(cy - gy) <= tol:
                            self.Sudoku_state[i] = chess_id
                            break   # 一个棋子只可能属于一个格
            print("Sudoku_state:",self.Sudoku_state)
            True_or_Flase=start.basic_suanfa(self.Sudoku_state,2)#应该只下白棋
            time.sleep_ms(100)
            img.clear()
            img.draw_string(50,50,"if finish,please press!",image.Color.from_rgb(255,0,0),scale=2)
            
            x,y,press=ts.read()
            if press == True:
                continue
            print("hello")             

    def show_task_menu(self):#菜单
        """
        在 MaixCam LCD 屏幕上绘制 2x2 任务选择菜单，并响应触摸点击调用对应任务方法。
        """
        # --- UI 布局常量 ---
        NUM_ROWS = 2
        NUM_COLS = 2
        grid_w = 100           # 每个方格宽度
        grid_h = 80            # 每个方格高度
        spacing_x = 20         # 列间距
        spacing_y = 30         # 行间距
        # 整体 2x2 区域的总宽度和高度
        total_w = NUM_COLS * grid_w + (NUM_COLS - 1) * spacing_x
        total_h = NUM_ROWS * grid_h + (NUM_ROWS - 1) * spacing_y
        # 左上角起始坐标（实现整体居中）
        start_x = (320 - total_w) // 2
        start_y = (240 - total_h) // 2
        print("[UI] 任务选择菜单已显示，等待触摸...")
        selected_task_id = None
        x, y, pressed=0,0,False
        pressed_already = False
        pressed_task_candidate = None
        while selected_task_id is None:
            # 1. 读取触摸事件
            x, y, pressed = ts.read()
            # 2. 绘制 UI (在循环内绘制保证触摸反馈不被清除)
            img = image.Image(320, 240)
            img.clear()
            img.draw_keypoints([x,y],image.Color.from_rgb(255,0,0))
            disp.show(img) 
            # 存储方格边界，用于触摸判断
            grid_bounds = {} 
            # 绘制 2x2 方格
            for i in range(4): # 任务 1 到 4
                row = i // NUM_COLS
                col = i % NUM_COLS
                # 计算当前方格坐标
                x_tl = start_x + col * (grid_w + spacing_x) # Top-Left X
                y_tl = start_y + row * (grid_h + spacing_y) # Top-Left Y
                x_br = x_tl + grid_w                        # Bottom-Right X
                y_br = y_tl + grid_h                        # Bottom-Right Y
                # 存储边界信息
                grid_bounds[i + 1] = (x_tl, y_tl, x_br, y_br)
                # 绘制矩形边框
                color = image.Color.from_rgb(0, 220, 0)
                thickness = 3
                # 触摸反馈：如果当前手指在方格内，则高亮（可选的触摸反馈，此处暂时不实现，只判断按下）
                if pressed and x_tl <= x < x_br and y_tl <= y < y_br:
                     color = image.Color.from_rgb(255, 255, 0) # 黄色高亮
                     thickness = -1 # 填充
                img.draw_rect(x_tl, y_tl, grid_w, grid_h, color, thickness)
                # 绘制数字
                num_str = str(i + 1)
                text_w = len(num_str) * 24 
                text_x = x_tl + (grid_w - text_w) // 2
                text_y = y_tl + (grid_h - 32) // 2   
                img.draw_string(text_x, text_y, num_str, image.Color.from_rgb(255, 255, 255), scale=3, thickness=3)
            # 3. 触摸判断（按下 + 松开 组成一次点击）
            if pressed:
                if not pressed_already:
                    # 第一次按下，判断是否在某个方格内
                    for task_id, bounds in grid_bounds.items():
                        x_tl, y_tl, x_br, y_br = bounds
                        if x_tl <= x < x_br and y_tl <= y < y_br:
                            pressed_task_candidate = task_id
                            pressed_already = True
                            break
            else:
                # 手指松开
                if pressed_already and pressed_task_candidate is not None:
                    selected_task_id = pressed_task_candidate
                    print(f"[TOUCH] 选中任务 {selected_task_id}.")
                    pressed_already = False
                    pressed_task_candidate = None

                    # 选中反馈
                    x_tl, y_tl, x_br, y_br = grid_bounds[selected_task_id]
                    img.draw_rect(x_tl, y_tl, grid_w, grid_h,
                    image.Color.from_rgb(0, 255, 0), thickness=-1)
                    disp.show(img)
                    time.sleep(0.3)
            # 4. 显示和延时
            disp.show(img)
            time.sleep_ms(50)
            
        # 5. 执行选中的任务
        if selected_task_id is not None:
            #print("selected_task_id:",selected_task_id)
            if selected_task_id == 1:
                start.task_1()
            elif selected_task_id == 2:
                start.task_2()
            elif selected_task_id == 4:
                start.task_4()
            else :
                print("此任务还未完成")
        return selected_task_id # 返回选中的任务号

if __name__ == "__main__":
    start=three_chess()
    UART_deliver(0,0,0)
    #start.task_4()
    #start.show_task_menu()
    #start.find_llllines()
    #start.find_chess(img)
    #start.find_llllines2()
    #maixpy_to_opencv(img)
    #start.find_roi_lab(img,start.Sudoku[12],start.Sudoku[13])
    #start.PID_control(img,start.Sudoku[8],start.Sudoku[9])
    # start.task_1()
    # start.task_2()
    disp.show(img)
    # print(start.Sudoku)
    # time.sleep(2)
    #start.show_task_menu()
    #time.sleep(2)
    # for i in range(0,3):
    #     img=cam.read()
    #     disp.show(img)
    #     start.find_chess2(img)
        #start.detect_chess_centers(img)
    judge=True
    while judge: 
        img=cam.read()
        #start.find_chess2(img)
        start.detect_chess_centers(img)
        # start.find_chess(img)
        #start.find_llllines2(img)
        #start.find_llllines(img)
        # try:
        #     judge=start.basic_suanfa(start.Sudoku_state,1)
        #     #print("start.Sudoku_state:",start.Sudoku_state)
        # except Exception:

        #     print("算法出错")
        disp.show(img)
        time.sleep(2)
    #print("start.Sudoku_lab:",start.Sudoku_lab)
    
    
    