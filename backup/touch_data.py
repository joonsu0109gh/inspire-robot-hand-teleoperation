import time
from pymodbus.client import ModbusTcpClient
from pymodbus.pdu import ExceptionResponse

# 定义 Modbus TCP 相关参数
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000

# 定义各部分数据地址范围
TOUCH_SENSOR_BASE_ADDR_PINKY = 3000  # 小拇指
TOUCH_SENSOR_END_ADDR_PINKY = 3369

TOUCH_SENSOR_BASE_ADDR_RING = 3370  # 无名指
TOUCH_SENSOR_END_ADDR_RING = 3739

TOUCH_SENSOR_BASE_ADDR_MIDDLE = 3740  # 中指
TOUCH_SENSOR_END_ADDR_MIDDLE = 4109

TOUCH_SENSOR_BASE_ADDR_INDEX = 4110  # 食指
TOUCH_SENSOR_END_ADDR_INDEX = 4479

TOUCH_SENSOR_BASE_ADDR_THUMB = 4480  # 大拇指
TOUCH_SENSOR_END_ADDR_THUMB = 4899

TOUCH_SENSOR_BASE_ADDR_PALM = 4900  # 掌心
TOUCH_SENSOR_END_ADDR_PALM = 5123

# Modbus 每次最多读取寄存器的数量
MAX_REGISTERS_PER_READ = 125


def read_register_range(client, start_addr, end_addr):
    """
    批量读取指定地址范围内的寄存器数据。
    """
    register_values = []
    # 分段读取寄存器
    for addr in range(start_addr, end_addr + 1, MAX_REGISTERS_PER_READ * 2):
        current_count = min(MAX_REGISTERS_PER_READ, (end_addr - addr) // 2 + 1)
        response = client.read_holding_registers(address=addr, count=current_count)

        if isinstance(response, ExceptionResponse) or response.isError():
            print(f"读取寄存器 {addr} 失败: {response}")
            register_values.extend([0] * current_count)
        else:
            register_values.extend(response.registers)

    return register_values

def format_finger_data(finger_name, data):
    """
    格式化四指触觉数据。
    """
    result = {}

    if finger_name != "大拇指":
        # 四指格式
        if len(data) < 185:
            print(f"{finger_name} 数据长度不足，至少185个数据，实际：{len(data)}")
            return None

        idx = 0
        # 指端数据 3x3
        result['tip_end'] = [data[idx + i*3: idx + (i+1)*3] for i in range(3)]
        idx += 9

        # 指尖触觉数据 12x8
        result['tip_touch'] = [data[idx + i*8: idx + (i+1)*8] for i in range(12)]
        idx += 96

        # 指腹触觉数据 10x8
        result['finger_pad'] = [data[idx + i*8: idx + (i+1)*8] for i in range(12)]
        idx += 80
    else:
        # 大拇指格式
        if len(data) < 210:
            print(f"{finger_name} 数据长度不足，至少210个数据，实际：{len(data)}")
            return None

        idx = 0
        # 指端数据 3x3
        result['tip_end'] = [data[idx + i*3: idx + (i+1)*3] for i in range(3)]
        idx += 9

        # 指尖触觉数据 12x8
        result['tip_touch'] = [data[idx + i*8: idx + (i+1)*8] for i in range(12)]
        idx += 96

        # 指中触觉数据 3x3
        result['middle_touch'] = [data[idx + i*3: idx + (i+1)*3] for i in range(3)]
        idx += 9

        # 指腹触觉数据 12x8
        finger_pad = [data[idx + i*8: idx + (i+1)*8] for i in range(12)]
        idx += 96

        # 指腹触觉数据行元素反转
        finger_pad = [row[::-1] for row in finger_pad]
        # 指腹触觉数据行顺序反转
        finger_pad.reverse()

        result['finger_pad'] = finger_pad

    return result

def format_palm_data(data):
    """
    格式化掌心数据为14x8矩阵，然后转置为8x14矩阵。
    """
    expected_len = 14 * 8
    if len(data) < expected_len:
        print(f"掌心数据长度不足，至少{expected_len}个数据，实际：{len(data)}")
        return None

    # 生成原始矩阵（14行8列）
    palm_matrix = [data[i*8:(i+1)*8] for i in range(14)]

    # 转置矩阵
    transposed = list(map(list, zip(*palm_matrix)))

    return transposed


def print_formatted_finger_data(finger_name, formatted_data):
    if formatted_data is None:
        print(f"{finger_name} 数据格式化失败")
        return

    print(f"--- {finger_name} 指端指端数据 (3x3) ---")
    for row in formatted_data.get('tip_end', []):
        print(row)

    print(f"--- {finger_name} 指尖触觉数据 (12x8) ---")
    for row in formatted_data.get('tip_touch', []):
        print(row)

    if finger_name == "大拇指":
        if 'middle_touch' in formatted_data:
            print(f"--- {finger_name} 指中触觉数据 (3x3) ---")
            for row in formatted_data['middle_touch']:
                print(row)
        else:
            print(f"{finger_name} 指中触觉数据缺失")

    print(f"--- {finger_name} 指腹触觉数据 ({'12x8' if finger_name == '大拇指' else '10x8'}) ---")
    for row in formatted_data.get('finger_pad', []):
        print(row)

def print_formatted_palm_data(palm_data):
    if palm_data is None:
        print("掌心数据格式化失败")
        return

    print("--- 掌心数据 (8x14) ---")
    for row in palm_data:
        print(row)


def read_multiple_registers():
    client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
    client.connect()

    try:
        while True:
            start_time = time.time()

            # 读取各部分数据
            pinky_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_PINKY,
                TOUCH_SENSOR_END_ADDR_PINKY
            )
            ring_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_RING,
                TOUCH_SENSOR_END_ADDR_RING
            )
            middle_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_MIDDLE,
                TOUCH_SENSOR_END_ADDR_MIDDLE
            )
            index_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_INDEX,
                TOUCH_SENSOR_END_ADDR_INDEX
            )
            thumb_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_THUMB,
                TOUCH_SENSOR_END_ADDR_THUMB
            )
            palm_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_PALM,
                TOUCH_SENSOR_END_ADDR_PALM
            )

            end_time = time.time()
            frequency = 1 / (end_time - start_time) if end_time > start_time else float('inf')

            # 格式化数据
            pinky_formatted = format_finger_data("小拇指", pinky_register_values)
            ring_formatted = format_finger_data("无名指", ring_register_values)
            middle_formatted = format_finger_data("中指", middle_register_values)
            index_formatted = format_finger_data("食指", index_register_values)
            thumb_formatted = format_finger_data("大拇指", thumb_register_values)
            palm_formatted = format_palm_data(palm_register_values)

            # 打印格式化数据
            print_formatted_finger_data("小拇指", pinky_formatted)
            print_formatted_finger_data("无名指", ring_formatted)
            print_formatted_finger_data("中指", middle_formatted)
            print_formatted_finger_data("食指", index_formatted)
            print_formatted_finger_data("大拇指", thumb_formatted)
            print_formatted_palm_data(palm_formatted)

            print(f"读取频率：{frequency:.2f} Hz")
            print("\n" + "="*40 + "\n")

    finally:
        client.close()


if __name__ == "__main__":
    read_multiple_registers()
