from pymodbus.client import ModbusTcpClient
import time

# Register address mapping for the Inspire dexterous hand
regdict = {
    'ID': 1000,
    'baudrate': 1001,
    'clearErr': 1004,
    'forceClb': 1009,
    'angleSet': 1486,      # Target angle (commanded position)
    'forceSet': 1498,      # Target force
    'speedSet': 1522,      # Target speed
    'angleAct': 1546,      # Actual angle (feedback)
    'forceAct': 1582,      # Actual force (feedback)
    'Action': 1600,        # Action control status
    'errCode': 1606,       # Error code
    'statusCode': 1612,    # System status code
    'temp': 1618,          # Cylinder temperature
    'actionSeq': 2320,     # Predefined motion sequence index
    'actionRun': 2322      # Trigger execution of the current sequence
}

def open_modbus(ip, port):
    """Initialize a Modbus TCP connection."""
    client = ModbusTcpClient(host=ip, port=port)
    client.connect()
    return client

def write_register(client, address, values):
    """Write a list of values to Modbus holding registers."""
    client.write_registers(address=address, values=values)

def read_register(client, address, count):
    """Read a number of consecutive Modbus holding registers."""
    response = client.read_holding_registers(address=address, count=count)
    return response.registers if not response.isError() else []

def write6(client, reg_name, val):
    """
    Write 6-channel data (for 6 fingers) to a Modbus register group.
    
    Parameters:
        reg_name: Register group name ('angleSet', 'forceSet', 'speedSet').
        val: A list of 6 integer values (0–1000). Use -1 as a placeholder to skip.
    """
    if reg_name in ['angleSet', 'forceSet', 'speedSet']:
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFFFF)  # Extract the lower 16 bits
        write_register(client, regdict[reg_name], val_reg)
    else:
        print("Function usage error: reg_name must be 'angleSet', 'forceSet', or 'speedSet'. "
              "val must be a list of 6 integers (0–1000), where -1 acts as a placeholder.")

def read6(client, reg_name):
    """
    Read 6-channel or 3-channel values from the specified Modbus register group.
    
    Supported 6-channel groups:
        ['angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct', 'Action']
    
    Supported 3-channel groups:
        ['errCode', 'statusCode', 'temp']
    """
    if reg_name in ['angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct', 'Action']:
        # Read 6 consecutive registers for the specified data type
        val = read_register(client, regdict[reg_name], 6)
        if len(val) < 6:
            print("No data received.")
            return
        print("Read values: ", end="")
        for v in val:
            print(v, end=" ")
        print()

    elif reg_name in ['errCode', 'statusCode', 'temp']:
        # Read 3 registers (each containing two bytes of information)
        val_act = read_register(client, regdict[reg_name], 3)
        if len(val_act) < 3:
            print("No data received.")
            return

        # Split each 16-bit register into high and low bytes
        results = []
        for i in range(len(val_act)):
            low_byte = val_act[i] & 0xFF        # Lower 8 bits
            high_byte = (val_act[i] >> 8) & 0xFF  # Upper 8 bits
            results.append(low_byte)
            results.append(high_byte)

        print("Read values: ", end="")
        for v in results:
            print(v, end=" ")
        print()

    else:
        print("Invalid function call. Supported names: "
              "'angleSet', 'forceSet', 'speedSet', 'angleAct', "
              "'forceAct', 'errCode', 'statusCode', 'temp'.")

if __name__ == '__main__':
    ip_address = '192.168.11.210'
    port = 6000
    print("Opening Modbus TCP connection...")
    client = open_modbus(ip_address, port)
    
    print("Setting dexterous hand motion speed. (-1 = not set)")
    write6(client, 'speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
    time.sleep(2)
    
    print("Setting dexterous hand grip force.")
    write6(client, 'forceSet', [500, 500, 500, 500, 500, 500])
    time.sleep(1)
    
    #open fingers
    print("Setting dexterous hand angle to 1000 (open). (-1 = skip this joint)")
    write6(client, 'angleSet', [0, 1000, 1000, 1000, 1000, 0])
    time.sleep(5)

    # # close fingers
    # print("Setting dexterous hand angle to 1000 (open). (-1 = skip this joint)")
    # write6(client, 'angleSet', [0, 0, 0, 0, 400, -1])
    # time.sleep(5)
    
    read6(client, 'angleAct')
    time.sleep(1)
    
    print("Reading fault information:")
    read6(client, 'errCode')
    time.sleep(1)
    
    print("Reading cylinder temperature:")
    read6(client, 'temp')
    time.sleep(1)
    
    # Close Modbus TCP connection
    client.close()
    print("Modbus T" \
    "CP connection closed.")

