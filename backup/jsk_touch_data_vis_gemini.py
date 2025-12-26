import sys
import time
import numpy as np
from pymodbus.client import ModbusTcpClient
from pymodbus.pdu import ExceptionResponse

# PyQt6 및 pyqtgraph 임포트
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- Modbus 및 주소 설정 (기존과 동일) ---

MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000
SENSOR_MAP = {
    'pinky': {
        'base': 3000, 'end': 3369,
        'parts': {
            'tip': {'offset': 0, 'count': 9, 'shape': (3, 3)},
            'nail': {'offset': 9, 'count': 96, 'shape': (12, 8)},
            'pad': {'offset': 9 + 96, 'count': 80, 'shape': (10, 8)}
        }
    },
    'ring': {
        'base': 3370, 'end': 3739,
        'parts': {
            'tip': {'offset': 0, 'count': 9, 'shape': (3, 3)},
            'nail': {'offset': 9, 'count': 96, 'shape': (12, 8)},
            'pad': {'offset': 9 + 96, 'count': 80, 'shape': (10, 8)}
        }
    },
    'middle': {
        'base': 3740, 'end': 4109,
        'parts': {
            'tip': {'offset': 0, 'count': 9, 'shape': (3, 3)},
            'nail': {'offset': 9, 'count': 96, 'shape': (12, 8)},
            'pad': {'offset': 9 + 96, 'count': 80, 'shape': (10, 8)}
        }
    },
    'index': {
        'base': 4110, 'end': 4479,
        'parts': {
            'tip': {'offset': 0, 'count': 9, 'shape': (3, 3)},
            'nail': {'offset': 9, 'count': 96, 'shape': (12, 8)},
            'pad': {'offset': 9 + 96, 'count': 80, 'shape': (10, 8)}
        }
    },
    'thumb': {
        'base': 4480, 'end': 4899,
        'parts': {
            'tip': {'offset': 0, 'count': 9, 'shape': (3, 3)},
            'nail': {'offset': 9, 'count': 96, 'shape': (12, 8)},
            'middle': {'offset': 9 + 96, 'count': 9, 'shape': (3, 3)},
            'pad': {'offset': 9 + 96 + 9, 'count': 96, 'shape': (12, 8)}
        }
    },
    'palm': {
        'base': 4900, 'end': 5123,
        'parts': {
            'palm': {'offset': 0, 'count': 112, 'shape': (8, 14)}
        }
    }
}
MAX_REGISTERS_PER_READ = 125
HEATMAP_VMAX = 4095 
MODBUS_READ_INTERVAL = 0.05

# --- Modbus/데이터 파싱 함수 (기존과 동일) ---

def read_register_range(client, start_addr, end_addr):
    total_bytes = end_addr - start_addr + 1
    total_registers = (total_bytes + 1) // 2 
    register_values = []
    current_addr = start_addr
    registers_left = total_registers
    while registers_left > 0:
        count_to_read = min(MAX_REGISTERS_PER_READ, registers_left)
        response = client.read_holding_registers(address=current_addr, count=count_to_read)
        if isinstance(response, ExceptionResponse) or response.isError():
            print(f"读取寄存器 {current_addr}  실패: {response}")
            register_values.extend([0] * count_to_read)
        else:
            register_values.extend(response.registers)
        current_addr += count_to_read * 2
        registers_left -= count_to_read
    return register_values

def parse_sensor_data(all_raw_data):
    parsed_data = {}
    def safe_reshape(key, raw_data, part_info):
        start = part_info['offset']
        length = part_info['count']
        shape = part_info['shape']
        if len(raw_data) < start + length:
            print(f"데이터 길이 부족: {key}. 필요: {start+length}, 실제: {len(raw_data)}")
            return np.zeros(shape)
        data_slice = raw_data[start : start + length]
        return np.array(data_slice).reshape(shape)

    for finger, config in SENSOR_MAP.items():
        raw_data = all_raw_data.get(finger, [])
        for part_name, part_info in config['parts'].items():
            key = f"{finger}_{part_name}"
            if key == 'palm_palm':
                shape_14x8 = (14, 8)
                data_matrix = safe_reshape(key, raw_data, {'offset': 0, 'count': 112, 'shape': shape_14x8})
                parsed_data['palm'] = data_matrix.T
            else:
                parsed_data[key] = safe_reshape(key, raw_data, part_info)
                if key == 'thumb_pad':
                    parsed_data[key] = np.fliplr(parsed_data[key])
                    parsed_data[key] = np.flipud(parsed_data[key])
    return parsed_data

# --- PyQt6 스레딩 및 GUI 구성 ---

class WorkerSignals(QtCore.QObject):
    '''
    Modbus 스레드에서 GUI 스레드로 데이터를 보내기 위한 시그널 정의
    '''
    data_ready = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

class ModbusWorker(QtCore.QObject):
    '''
    Modbus 통신을 수행하는 백그라운드 작업자 (QObject)
    '''
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.running = True

    @QtCore.pyqtSlot()
    def run(self):
        try:
            client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
            if not client.connect():
                self.signals.error.emit(f"Modbus 연결 실패: {MODBUS_IP}:{MODBUS_PORT}")
                return
        except Exception as e:
            self.signals.error.emit(f"Modbus 클라이언트 생성 실패: {e}")
            return
        
        print("Modbus 스레드 시작...")
        try:
            while self.running:
                start_time = time.time()
                all_raw_data = {}
                for finger, config in SENSOR_MAP.items():
                    all_raw_data[finger] = read_register_range(
                        client, config['base'], config['end']
                    )
                
                parsed_data = parse_sensor_data(all_raw_data)
                self.signals.data_ready.emit(parsed_data)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, MODBUS_READ_INTERVAL - elapsed)
                time.sleep(sleep_time)
        except Exception as e:
            self.signals.error.emit(f"Modbus 작업 중 오류: {e}")
        finally:
            client.close()
            self.signals.finished.emit()
            print("Modbus 스레드 종료.")

    def stop(self):
        self.running = False

class HandVisualizerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("로봇 핸드 택타일 센서 (PyQtGraph)")
        self.setGeometry(100, 100, 1600, 900)
        
        self.VMAX = HEATMAP_VMAX
        self.heatmap_artists = {} # pg.ImageItem 객체 저장

        # Colormap 설정 (viridis)
        self.cmap = pg.colormap.get('viridis')
        self.lut = self.cmap.getLookupTable() # Lookup Table 미리 생성

        self.create_layout()
        self.start_modbus_thread()

    def create_heatmap_widget(self, title, data_shape):
        """
        PyQtGraph의 PlotWidget과 ImageItem을 생성
        """
        # 1. PlotWidget 생성 (차트 영역)
        plot_widget = pg.PlotWidget()
        plot_widget.setTitle(title, size="10pt")
        plot_widget.showAxes(False)      # 축 숨기기
        plot_widget.setAspectLocked(lock=False) # 창 크기에 맞춰 스트레칭
        
        # 2. ImageItem 생성 (실제 히트맵 이미지)
        img_item = pg.ImageItem()
        plot_widget.addItem(img_item)
        
        # 3. Colormap 및 레벨 설정
        img_item.setLookupTable(self.lut)
        img_item.setLevels([0, self.VMAX]) # 고정 레벨 (0 ~ VMAX)
        
        # 0으로 초기화
        img_item.setImage(np.zeros(data_shape))
        
        # PlotWidget(컨테이너), ImageItem(데이터 업데이트용) 반환
        return plot_widget, img_item

    def create_finger_column(self, finger_name_en, finger_name_ko):
        """
        손가락(tip, nail, pad) 3개의 히트맵을 포함하는 QWidget(열)을 생성
        """
        # QWidget: 위젯들을 담는 컨테이너
        container = QtWidgets.QWidget()
        # QVBoxLayout: 위젯들을 수직(Vertical)으로 쌓는 레이아웃
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5) # 내부 여백
        layout.setSpacing(5)                   # 위젯 간 간격

        # 1. Tip
        key_tip = f"{finger_name_en}_tip"
        shape_tip = SENSOR_MAP[finger_name_en]['parts']['tip']['shape']
        plot_tip, img_tip = self.create_heatmap_widget(f"Tip ({shape_tip[0]}x{shape_tip[1]})", shape_tip)
        self.heatmap_artists[key_tip] = img_tip
        layout.addWidget(plot_tip)

        # 2. Nail
        key_nail = f"{finger_name_en}_nail"
        shape_nail = SENSOR_MAP[finger_name_en]['parts']['nail']['shape']
        plot_nail, img_nail = self.create_heatmap_widget(f"Nail ({shape_nail[0]}x{shape_nail[1]})", shape_nail)
        self.heatmap_artists[key_nail] = img_nail
        layout.addWidget(plot_nail)

        # 3. Pad
        key_pad = f"{finger_name_en}_pad"
        shape_pad = SENSOR_MAP[finger_name_en]['parts']['pad']['shape']
        plot_pad, img_pad = self.create_heatmap_widget(f"Pad ({shape_pad[0]}x{shape_pad[1]})", shape_pad)
        self.heatmap_artists[key_pad] = img_pad
        layout.addWidget(plot_pad)

        # QWidget에 제목 라벨 추가
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame_layout = QtWidgets.QVBoxLayout(frame)
        label = QtWidgets.QLabel(finger_name_ko)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        frame_layout.addWidget(label)
        frame_layout.addWidget(container)
        
        return frame

    def create_thumb_column(self):
        """
        엄지손가락 (4개 파트) 열을 생성
        """
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 1. Tip
        key_tip = "thumb_tip"
        shape_tip = SENSOR_MAP['thumb']['parts']['tip']['shape']
        plot, img = self.create_heatmap_widget(f"Tip ({shape_tip[0]}x{shape_tip[1]})", shape_tip)
        self.heatmap_artists[key_tip] = img
        layout.addWidget(plot)

        # 2. Nail
        key_nail = "thumb_nail"
        shape_nail = SENSOR_MAP['thumb']['parts']['nail']['shape']
        plot, img = self.create_heatmap_widget(f"Nail ({shape_nail[0]}x{shape_nail[1]})", shape_nail)
        self.heatmap_artists[key_nail] = img
        layout.addWidget(plot)

        # 3. Middle
        key_mid = "thumb_middle"
        shape_mid = SENSOR_MAP['thumb']['parts']['middle']['shape']
        plot, img = self.create_heatmap_widget(f"Middle ({shape_mid[0]}x{shape_mid[1]})", shape_mid)
        self.heatmap_artists[key_mid] = img
        layout.addWidget(plot)

        # 4. Pad
        key_pad = "thumb_pad"
        shape_pad = SENSOR_MAP['thumb']['parts']['pad']['shape']
        plot, img = self.create_heatmap_widget(f"Pad ({shape_pad[0]}x{shape_pad[1]})", shape_pad)
        self.heatmap_artists[key_pad] = img
        layout.addWidget(plot)
        
        # QWidget에 제목 라벨 추가
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame_layout = QtWidgets.QVBoxLayout(frame)
        label = QtWidgets.QLabel("엄지 (Thumb)")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        frame_layout.addWidget(label)
        frame_layout.addWidget(container)

        return frame

    def create_layout(self):
        """
        QGridLayout을 사용하여 오른손 레이아웃 생성
        """
        # 메인 창의 중앙에 놓일 위젯
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 그리드 레이아웃 생성
        grid_layout = QtWidgets.QGridLayout(central_widget)

        # --- 1행: 손가락 ---
        # 오른손 레이아웃: 새끼(0), 약지(1), 중지(2), 검지(3), 엄지(4)
        pinky_col = self.create_finger_column('pinky', '소지 (Pinky)')
        ring_col = self.create_finger_column('ring', '약지 (Ring)')
        middle_col = self.create_finger_column('middle', '중지 (Middle)')
        index_col = self.create_finger_column('index', '검지 (Index)')
        thumb_col = self.create_thumb_column() # 엄지 (4파트)

        grid_layout.addWidget(pinky_col, 0, 0)
        grid_layout.addWidget(ring_col, 0, 1)
        grid_layout.addWidget(middle_col, 0, 2)
        grid_layout.addWidget(index_col, 0, 3)
        grid_layout.addWidget(thumb_col, 0, 4)

        # --- 2행: 손바닥 ---
        key_palm = "palm"
        shape_palm = SENSOR_MAP['palm']['parts']['palm']['shape']
        palm_plot, palm_img = self.create_heatmap_widget(
            f"손바닥 (Palm) ({shape_palm[0]}x{shape_palm[1]})", shape_palm
        )
        self.heatmap_artists[key_palm] = palm_img
        
        # 손바닥 위젯: 1행, 0열에 위치하며, 1행(rowSpan=1), 5열(colSpan=5)을 차지
        grid_layout.addWidget(palm_plot, 1, 0, 1, 5)

        # --- 상대적 크기 조절 설정 ---
        
        # 1행(손가락)과 2행(손바닥)의 세로 비율을 1:2로 설정
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 2)

        # 0~4열(손가락)의 가로 비율을 1:1:1:1:1 (모두 동일하게) 설정
        for i in range(5):
            grid_layout.setColumnStretch(i, 1)

    def start_modbus_thread(self):
        """Modbus 워커 스레드 시작"""
        # 1. 스레드와 워커 객체 생성
        self.thread = QtCore.QThread()
        self.worker = ModbusWorker()
        
        # 2. 워커를 스레드로 이동
        self.worker.moveToThread(self.thread)
        
        # 3. 시그널 연결
        # 스레드가 시작되면 -> worker.run() 실행
        self.thread.started.connect(self.worker.run)
        # worker가 데이터 시그널을 보내면 -> self.update_all_heatmaps 실행
        self.worker.signals.data_ready.connect(self.update_all_heatmaps)
        # worker가 오류 시그널을 보내면 -> print
        self.worker.signals.error.connect(print)
        # worker가 끝나면 -> 스레드 종료
        self.worker.signals.finished.connect(self.thread.quit)
        # 스레드/워커가 모두 종료되면 -> 메모리에서 삭제
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # 4. 스레드 시작
        self.thread.start()

    @QtCore.pyqtSlot(dict)
    def update_all_heatmaps(self, data):
        """
        [슬롯] Modbus 스레드에서 받은 데이터로 히트맵 업데이트
        """
        for key, img_item in self.heatmap_artists.items():
            if key in data and data[key] is not None:
                # setImage가 pyqtgraph의 핵심 업데이트 함수
                # autoLevels=False로 해야 VMAX가 고정됨
                # 참고: pg.ImageItem은 (row, col) 순서(Numpy)가 아닌 (x, y) 순서를
                # 기본으로 기대할 수 있으나, setImage는 numpy 배열을 잘 처리합니다.
                # 만약 이미지가 90도 회전되어 보인다면 data[key].T (전치)로 설정합니다.
                img_item.setImage(data[key], autoLevels=False)

    def closeEvent(self, event):
        """창을 닫을 때 스레드를 안전하게 종료"""
        print("종료 요청. Modbus 스레드를 중지합니다...")
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread'):
            self.thread.quit()
            self.thread.wait() # 스레드가 완전히 종료될 때까지 대기
        event.accept()

if __name__ == "__main__":
    # QApplication: GUI 애플리케이션의 이벤트 루프 및 관리
    app = QtWidgets.QApplication(sys.argv)
    
    # 전역 스타일시트로 경계선 강조
    app.setStyleSheet("""
        QFrame {
            border: 1px solid #555;
            border-radius: 5px;
        }
        QLabel {
            padding: 5px;
        }
    """)
    
    window = HandVisualizerApp()
    window.show() # 창 표시
    
    # 이벤트 루프 시작
    sys.exit(app.exec())