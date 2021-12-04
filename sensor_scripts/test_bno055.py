import board
import adafruit_bno055
import time
from datetime import datetime

i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)
today=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
file = open("imuData-{}.txt".format(today),"w")

last_val = 0xFFFF
file.write("Current time - Acceleration x/y/z - Magnetic x/y/z - Gyro x/y/z - Euler x/y/z - Temperature \n")
def temperature():
    global last_val  # pylint: disable=global-statement
    result = sensor.temperature
    if abs(result - last_val) == 128:
        result = sensor.temperature
        if abs(result - last_val) == 128:
            return 0b00111111 & result
    last_val = result
    return result

i=0
while i<10:
    current_time = datetime.now().strftime("%H:%M:%S")

    file.write(str(datetime.now())+",")
    file.write(str(sensor.acceleration[0])+",")
    file.write(str(sensor.acceleration[1])+",")
    file.write(str(sensor.acceleration[2])+",")
    file.write(str(sensor.magnetic[0])+",")
    file.write(str(sensor.magnetic[1])+",")
    file.write(str(sensor.magnetic[2])+",")
    file.write(str(sensor.gyro[0])+",")
    file.write(str(sensor.gyro[1])+",")
    file.write(str(sensor.gyro[2])+",")
    file.write(str(sensor.euler[0])+",")
    file.write(str(sensor.euler[1])+",")
    file.write(str(sensor.euler[2])+",")
    file.write(str(sensor.temperature)+"\n")
    i+=1

file.close()
