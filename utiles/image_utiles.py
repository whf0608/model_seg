
def img_de_green(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([20, 20, 20])
    lower_green = np.array([35, 43,46])
    upper_green = np.array([77, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    ret, binary=cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)
    binary  = cv2.dilate(binary , kernel, 6) # 更改迭代次数为2
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # binary = cv2.erode(binary, kernel, iterations=2)
    # res = cv2.bitwise_and(image, image, mask=~binary)
    image[binary>0]=[0,0,0]
    return image