# python3 -m venv myenv
# source myenv/bin/activate
import serial
import time
import os
c = 0
flag = 0
serial_port = '/dev/ttyACM1'
baud_rate = 115200 
while True:
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        flag += 1
        c = 0
        ser.close()
    except:
        c+=1
        print ("Receive Error : ", c) 
        flag = 0
    print("Flag : ", flag)
    if(flag == 1):
        print("Received Command")
        # os.system('python3 detect_short_magneto.py')
        # os.system('/home/ashutosh/Downloads/R2_FILES/ball_detection/myenv/bin/python /home/ashutosh/Downloads/R2_FILES/ball_detection/detect_short_magneto.py')
        os.system('/home/ashutosh/Downloads/R2_FILES/ball_detection/myenv/bin/python /home/ashutosh/Downloads/R2_FILES/ball_detection/detect_short_magneto_blue.py')

    time.sleep(1)