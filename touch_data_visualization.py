import sys
import time
import numpy as np
from pymodbus.client import ModbusTcpClient
from pymodbus.pdu import ExceptionResponse

# PyQt6 & pyqtgraph
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

# ============================================================
# Modbus configuration
# ============================================================

MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000

SENSOR_MAP = {
    'pinky': {
        'base': 3000, 'end': 3369,
        'parts': {
            'tip':   {'offset': 0,        'count': 9,  'shape': (3, 3)},
            'nail':  {'offset': 9,        'count': 96, 'shape': (12, 8)},
            'pad':   {'offset': 9 + 96,   'count': 80, 'shape': (10, 8)},
        }
    },
    'ring': {
        'base': 3370, 'end': 3739,
        'parts': {
            'tip':   {'offset': 0,        'count': 9,  'shape': (3, 3)},
            'nail':  {'offset': 9,        'count': 96, 'shape': (12, 8)},
            'pad':   {'offset': 9 + 96,   'count': 80, 'shape': (10, 8)},
        }
    },
    'middle': {
        'base': 3740, 'end': 4109,
        'parts': {
            'tip':   {'offset': 0,        'count': 9,  'shape': (3, 3)},
            'nail':  {'offset': 9,        'count': 96, 'shape': (12, 8)},
            'pad':   {'offset': 9 + 96,   'count': 80, 'shape': (10, 8)},
        }
    },
    'index': {
        'base': 4110, 'end': 4479,
        'parts': {
            'tip':   {'offset': 0,        'count': 9,  'shape': (3, 3)},
            'nail':  {'offset': 9,        'count': 96, 'shape': (12, 8)},
            'pad':   {'offset': 9 + 96,   'count': 80, 'shape': (10, 8)},
        }
    },
    'thumb': {
        'base': 4480, 'end': 4899,
        'parts': {
            'tip':    {'offset': 0,          'count': 9,  'shape': (3, 3)},
            'nail':   {'offset': 9,          'count': 96, 'shape': (12, 8)},
            'middle': {'offset': 9 + 96,     'count': 9,  'shape': (3, 3)},
            'pad':    {'offset': 9 + 96 + 9, 'count': 96, 'shape': (12, 8)},
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
MODBUS_READ_INTERVAL = 0.05  # seconds


# ============================================================
# Modbus utilities
# ============================================================

def read_register_range(client, start_addr, end_addr):
    """
    Read a range of Modbus holding registers safely,
    respecting the maximum registers per read.
    """
    total_bytes = end_addr - start_addr + 1
    total_registers = (total_bytes + 1) // 2

    values = []
    current_addr = start_addr
    remaining = total_registers

    while remaining > 0:
        count = min(MAX_REGISTERS_PER_READ, remaining)
        response = client.read_holding_registers(
            address=current_addr,
            count=count
        )

        if isinstance(response, ExceptionResponse) or response.isError():
            print(f"[WARN] Failed to read registers @ {current_addr}: {response}")
            values.extend([0] * count)
        else:
            values.extend(response.registers)

        current_addr += count * 2
        remaining -= count

    return values


def parse_sensor_data(raw_data_dict):
    """
    Parse raw Modbus data into structured numpy arrays
    for each finger part.
    """
    parsed = {}

    def safe_reshape(key, raw, info):
        start, length, shape = info['offset'], info['count'], info['shape']
        if len(raw) < start + length:
            print(f"[WARN] Insufficient data for {key}: "
                  f"expected {start + length}, got {len(raw)}")
            return np.zeros(shape)
        return np.array(raw[start:start + length]).reshape(shape)

    for finger, cfg in SENSOR_MAP.items():
        raw = raw_data_dict.get(finger, [])
        for part, info in cfg['parts'].items():
            key = f"{finger}_{part}"

            if key == "palm_palm":
                mat = safe_reshape(key, raw, info)
                parsed["palm"] = mat.T
            else:
                mat = safe_reshape(key, raw, info)
                if key == "thumb_pad":
                    mat = np.flipud(np.fliplr(mat))
                parsed[key] = mat

    return parsed


# ============================================================
# Threading & Signals
# ============================================================

class WorkerSignals(QtCore.QObject):
    """Signals for communication between worker thread and GUI."""
    data_ready = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)


class ModbusWorker(QtCore.QObject):
    """Background worker handling Modbus communication."""

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.running = True

    @QtCore.pyqtSlot()
    def run(self):
        try:
            client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
            if not client.connect():
                self.signals.error.emit(
                    f"Failed to connect to Modbus server "
                    f"{MODBUS_IP}:{MODBUS_PORT}"
                )
                return
        except Exception as e:
            self.signals.error.emit(f"Modbus client initialization failed: {e}")
            return

        print("[INFO] Modbus worker started")

        try:
            while self.running:
                t0 = time.time()
                raw_data = {
                    finger: read_register_range(client, cfg['base'], cfg['end'])
                    for finger, cfg in SENSOR_MAP.items()
                }

                parsed = parse_sensor_data(raw_data)
                self.signals.data_ready.emit(parsed)

                elapsed = time.time() - t0
                time.sleep(max(0.0, MODBUS_READ_INTERVAL - elapsed))

        except Exception as e:
            self.signals.error.emit(f"Runtime error in Modbus worker: {e}")
        finally:
            client.close()
            self.signals.finished.emit()
            print("[INFO] Modbus worker stopped")

    def stop(self):
        self.running = False


# ============================================================
# GUI Application
# ============================================================

class HandVisualizerApp(QtWidgets.QMainWindow):
    """Main application window for tactile heatmap visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Hand Tactile Sensor Visualization")
        self.setGeometry(100, 100, 1600, 900)

        self.vmax = HEATMAP_VMAX
        self.heatmap_items = {}

        self.cmap = pg.colormap.get("viridis")
        self.lut = self.cmap.getLookupTable()

        self._build_layout()
        self._start_modbus_thread()

    def _create_heatmap(self, title, shape):
        plot = pg.PlotWidget(title=title)
        plot.showAxes(False)
        plot.setAspectLocked(False)

        img = pg.ImageItem()
        img.setLookupTable(self.lut)
        img.setLevels([0, self.vmax])
        img.setImage(np.zeros(shape))

        plot.addItem(img)
        return plot, img

    def _start_modbus_thread(self):
        self.thread = QtCore.QThread()
        self.worker = ModbusWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.signals.data_ready.connect(self.update_heatmaps)
        self.worker.signals.error.connect(print)
        self.worker.signals.finished.connect(self.thread.quit)

        self.thread.start()

    @QtCore.pyqtSlot(dict)
    def update_heatmaps(self, data):
        for key, img in self.heatmap_items.items():
            if key in data:
                img.setImage(data[key], autoLevels=False)

    def closeEvent(self, event):
        print("[INFO] Shutting down application...")
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

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
    window.show()

    sys.exit(app.exec())
