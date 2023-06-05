from mss import mss
import numpy as np
import cv2
import time

import pyautogui
pyautogui.PAUSE = 0


LEFT, TOP = 0, 64
# WIDTH, HEIGHT = 445, 940 # Full screen
WIDTH, HEIGHT = 445, 860

def show_screenshot(img):

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def capture_screenshot():

    with mss() as sct:
        monitor = {"top": TOP, "left": LEFT,
                   "width": WIDTH, "height": HEIGHT}
        screenshot = np.array(sct.grab(monitor))

        return cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        # return cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) # Gray

def find_vertical_line(lines, y, x):

    res = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            y3 = min(y1, y2)
            if x > x1 and y3 > y and x - x1 < 100 and y3 - y < 100:
                res.append(x1)

    return res

def find_objects(img):

    lower_red = np.array([0, 200, 200], dtype = "uint8")
    upper_red= np.array([10, 255, 255], dtype = "uint8")

    mask = cv2.inRange(img, lower_red, upper_red)

    lower_green = np.array([30, 100, 150], dtype = "uint8")
    upper_green= np.array([40, 155, 255], dtype = "uint8")

    mask_bird = cv2.inRange(img, lower_green, upper_green)

    bird_loc = np.average(np.where(mask_bird==255), axis=-1)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(mask, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    
    for line in lines:
        for x1, y1, x2, y2 in line:

            if abs(y1 - y2) < 10: # Horizontal line
                res = find_vertical_line(lines, y1, min(x1, x2))
                if len(res) > 0:
                    line_image = cv2.circle(line_image, (res[0], y1), radius=10, color=(0, 255, 0), thickness=-1)
                    line_image = cv2.circle(line_image, (res[0] + 150, y1), radius=10, color=(0, 255, 0), thickness=-1)

                    line_image = cv2.circle(line_image, (res[0], y1 - 300), radius=10, color=(255, 255, 0), thickness=-1)
                    line_image = cv2.circle(line_image, (res[0] + 150, y1 - 300), radius=10, color=(255, 255, 0), thickness=-1)

    
    line_image = cv2.circle(line_image, (int(bird_loc[1]), int(bird_loc[0])), radius=10, color=(255, 255, 255), thickness=-1)

    
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow('edges', lines_edges)
    # cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def click(x, y):
    pyautogui.moveTo((x+LEFT)/2, (y+TOP)/2)
    pyautogui.click()

def main():

    start = time.time()
    img = capture_screenshot()
    print(f'{time.time() - start}s')
    
    start = time.time()
    find_objects(img)
    print(f'{time.time() - start}s')


if __name__ == "__main__":

    main()
