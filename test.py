import campoint
campoint.show_debug['cap'] = True

while True:
    campoint.find_marker(2)
    campoint.cap()