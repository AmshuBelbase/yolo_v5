# import serial

# import time

# # Configure the serial connection

# port = '/dev/ttyACM0'  # Adjust the port to match your setup

# baudrate = 115200

# serial_connection = serial.Serial(port, baudrate)

# # Read and write data until the transfer is complete

# for i in range(1, 1001):

#     print(i)

#     serial_connection.write((str(i) + ',').encode())

#     time.sleep(0.01)

# time.sleep(10)

# serial_connection.close()

import serial 
import time

serial_port = '/dev/ttyACM0'
baud_rate = 115200 
ser = serial.Serial(serial_port, baud_rate, timeout=1)




class trying:
        def __init__(self):
                count = 1000
                while True:
                        if count %1010 == 0:
                                trying.call_back(ser, 0)
                                break
                        trying.call_back(ser, count)
                        count += 1 
                        time.sleep(1)

        @staticmethod
        def call_back(ser, count):
                # Open serial port
                # time.sleep(0.5) 
                at = time.time()
                print("Start",at)
                # Define floats to send
                float1 = int(count)
                float2 = int(count)
                float3 = int(count)
                float4 = int(count)

                # Convert to bytes
                data = (str(float1) + '|' + 
                        str(float2) + '|' +
                        str(float3) + '|' +
                        str(float4)) + "#"
                
                # Send data
                ser.write(data.encode())  
                print(f"Sent: {data}")
                print("End",time.time())
                print("Time taken", time.time() - at)


if __name__ == "__main__":
        obj = trying()
