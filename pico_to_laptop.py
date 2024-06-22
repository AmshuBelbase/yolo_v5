import serial
import time

serial_port = '/dev/ttyACM0'  # Adjust this to match your serial port
baudrate = 115200  # Match this to the baud rate set on your Pico

try:
    # Open the serial connection
    serial_connection = serial.Serial(serial_port, baudrate)
    print(f"Serial port {serial_port} opened successfully.")

    while True:
        if serial_connection.in_waiting > 0:
            # Read data from the serial port
            data = serial_connection.readline().strip().decode()
            print(f"{data}")
            
            # Check if received data is "6"
            if data == "6":
                print("Received 6")
        else:
            print("..")

        time.sleep(2)  # Adjust as needed to control the polling frequency

except serial.serialutil.SerialException as e:
    print(f"SerialException: {e}")

except KeyboardInterrupt:
    print("\nSerial communication interrupted by user.")

finally:
    if 'serial_connection' in locals() and serial_connection.isOpen():
        serial_connection.close()
        print("Serial port closed.")


# import serial

# # Configure the serial connection
# serial_port = '/dev/ttyACM0'
# baudrate = 115200
# serial_connection = serial.Serial(serial_port, baudrate)

# # Read and write data until the transfer is complete
# while True:
#     serial_connection.flush() 
#     data = serial_connection.readline().strip()
#     print(data.decode())
#     if data.decode() == "6":
#         print("Received 6")  