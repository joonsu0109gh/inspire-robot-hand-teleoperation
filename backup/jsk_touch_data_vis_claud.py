import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pymodbus.client import ModbusTcpClient
from pymodbus.pdu import ExceptionResponse
import matplotlib
matplotlib.use('TkAgg')

# Modbus TCP 설정
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000

# 각 손가락 데이터 주소 범위
FINGER_ADDRESSES = {
    'pinky': (3000, 3369),
    'ring': (3370, 3739),
    'middle': (3740, 4109),
    'index': (4110, 4479),
    'thumb': (4480, 4899),
    'palm': (4900, 5123)
}

MAX_REGISTERS_PER_READ = 125


def read_register_range(client, start_addr, end_addr):
    """지정된 주소 범위의 레지스터 읽기"""
    register_values = []
    for addr in range(start_addr, end_addr + 1, MAX_REGISTERS_PER_READ * 2):
        current_count = min(MAX_REGISTERS_PER_READ, (end_addr - addr) // 2 + 1)
        response = client.read_holding_registers(address=addr, count=current_count)
        
        if isinstance(response, ExceptionResponse) or response.isError():
            register_values.extend([0] * current_count)
        else:
            register_values.extend(response.registers)
    
    return register_values


def format_finger_data(finger_name, data):
    """손가락 데이터를 구조화된 형태로 변환"""
    result = {}
    
    if finger_name != 'thumb':
        if len(data) < 185:
            return None
        
        idx = 0
        result['tip'] = np.array(data[idx:idx+9]).reshape(3, 3)
        idx += 9
        result['nail'] = np.array(data[idx:idx+96]).reshape(12, 8)
        idx += 96
        result['pad'] = np.array(data[idx:idx+80]).reshape(10, 8)
        
    else:  # 엄지손가락
        if len(data) < 210:
            return None
        
        idx = 0
        result['tip'] = np.array(data[idx:idx+9]).reshape(3, 3)
        idx += 9
        result['nail'] = np.array(data[idx:idx+96]).reshape(12, 8)
        idx += 96
        result['middle'] = np.array(data[idx:idx+9]).reshape(3, 3)
        idx += 9
        finger_pad = np.array(data[idx:idx+96]).reshape(12, 8)
        result['pad'] = np.flip(np.flip(finger_pad, axis=0), axis=1)
    
    return result


def format_palm_data(data):
    """손바닥 데이터를 8x14 행렬로 변환"""
    if len(data) < 112:
        return None
    return np.array(data[:112]).reshape(14, 8).T


class RobotHandVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Robot Hand Tactile Sensor Heatmap', fontsize=16, fontweight='bold')
        
        gs = GridSpec(4, 6, figure=self.fig, hspace=0.4, wspace=0.3)
        
        self.axes = {}
        self.images = {}
        
        # 손가락 4개
        finger_names = ['index', 'middle', 'ring', 'pinky']
        finger_labels = ['Index', 'Middle', 'Ring', 'Pinky']
        
        for i, (fname, flabel) in enumerate(zip(finger_names, finger_labels)):
            col = i + 1
            
            ax_tip = self.fig.add_subplot(gs[0, col])
            ax_nail = self.fig.add_subplot(gs[1, col])
            ax_pad = self.fig.add_subplot(gs[2, col])
            
            self.axes[f'{fname}_tip'] = ax_tip
            self.axes[f'{fname}_nail'] = ax_nail
            self.axes[f'{fname}_pad'] = ax_pad
            
            # 초기 히트맵 생성
            self.images[f'{fname}_tip'] = ax_tip.imshow(np.zeros((3, 3)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
            self.images[f'{fname}_nail'] = ax_nail.imshow(np.zeros((12, 8)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
            self.images[f'{fname}_pad'] = ax_pad.imshow(np.zeros((10, 8)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
            
            ax_tip.set_title(f'{flabel} Tip', fontsize=9)
            ax_nail.set_title(f'{flabel} Nail', fontsize=9)
            ax_pad.set_title(f'{flabel} Pad', fontsize=9)
            
            ax_tip.axis('off')
            ax_nail.axis('off')
            ax_pad.axis('off')
        
        # 엄지손가락
        ax_thumb_tip = self.fig.add_subplot(gs[0, 0])
        ax_thumb_nail = self.fig.add_subplot(gs[1, 0])
        ax_thumb_middle = self.fig.add_subplot(gs[2, 0])
        ax_thumb_pad = self.fig.add_subplot(gs[3, 0])
        
        self.axes['thumb_tip'] = ax_thumb_tip
        self.axes['thumb_nail'] = ax_thumb_nail
        self.axes['thumb_middle'] = ax_thumb_middle
        self.axes['thumb_pad'] = ax_thumb_pad
        
        self.images['thumb_tip'] = ax_thumb_tip.imshow(np.zeros((3, 3)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
        self.images['thumb_nail'] = ax_thumb_nail.imshow(np.zeros((12, 8)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
        self.images['thumb_middle'] = ax_thumb_middle.imshow(np.zeros((3, 3)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
        self.images['thumb_pad'] = ax_thumb_pad.imshow(np.zeros((12, 8)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
        
        ax_thumb_tip.set_title('Thumb Tip', fontsize=9)
        ax_thumb_nail.set_title('Thumb Nail', fontsize=9)
        ax_thumb_middle.set_title('Thumb Middle', fontsize=9)
        ax_thumb_pad.set_title('Thumb Pad', fontsize=9)
        
        ax_thumb_tip.axis('off')
        ax_thumb_nail.axis('off')
        ax_thumb_middle.axis('off')
        ax_thumb_pad.axis('off')
        
        # 손바닥
        ax_palm = self.fig.add_subplot(gs[3, 1:5])
        self.axes['palm'] = ax_palm
        self.images['palm'] = ax_palm.imshow(np.zeros((8, 14)), cmap='hot', vmin=0, vmax=4095, interpolation='nearest')
        ax_palm.set_title('Palm', fontsize=10)
        ax_palm.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # GUI가 완전히 렌더링될 때까지 대기
    
    def update(self, finger_data, palm_data):
        """히트맵 업데이트"""
        for finger in ['pinky', 'ring', 'middle', 'index']:
            if finger_data[finger] is not None:
                self.images[f'{finger}_tip'].set_data(finger_data[finger]['tip'])
                self.images[f'{finger}_nail'].set_data(finger_data[finger]['nail'])
                self.images[f'{finger}_pad'].set_data(finger_data[finger]['pad'])
        
        if finger_data['thumb'] is not None:
            self.images['thumb_tip'].set_data(finger_data['thumb']['tip'])
            self.images['thumb_nail'].set_data(finger_data['thumb']['nail'])
            self.images['thumb_middle'].set_data(finger_data['thumb']['middle'])
            self.images['thumb_pad'].set_data(finger_data['thumb']['pad'])
        
        if palm_data is not None:
            self.images['palm'].set_data(palm_data)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
    
    if not client.connect():
        print("Modbus 연결 실패!")
        return
    
    print("Modbus 연결 성공! 시각화 시작...")
    visualizer = RobotHandVisualizer()
    
    try:
        while True:
            start_time = time.time()
            
            # 모든 센서 데이터 읽기
            raw_data = {}
            for finger, (start, end) in FINGER_ADDRESSES.items():
                raw_data[finger] = read_register_range(client, start, end)
            
            # 데이터 포맷팅
            finger_data = {
                'pinky': format_finger_data('pinky', raw_data['pinky']),
                'ring': format_finger_data('ring', raw_data['ring']),
                'middle': format_finger_data('middle', raw_data['middle']),
                'index': format_finger_data('index', raw_data['index']),
                'thumb': format_finger_data('thumb', raw_data['thumb'])
            }
            
            palm_data = format_palm_data(raw_data['palm'])
            
            # 시각화 업데이트
            visualizer.update(finger_data, palm_data)
            
            elapsed = time.time() - start_time
            frequency = 1 / elapsed if elapsed > 0 else 0
            print(f"\rUpdate frequency: {frequency:.2f} Hz", end='', flush=True)
            
    except KeyboardInterrupt:
        print("\n\n프로그램 종료")
    finally:
        client.close()
        plt.close('all')


if __name__ == "__main__":
    main()