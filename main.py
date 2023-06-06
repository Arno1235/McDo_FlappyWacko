from mss import mss
import numpy as np
import cv2
import time

import pyautogui
pyautogui.PAUSE = 0


LEFT, TOP = 0, 64
WIDTH, HEIGHT = 445, 860 # Full screen: 445, 940

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

        return cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV), cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

def find_objects(img, img_gray, template1, template2, template1_h):

    lower_green = np.array([30, 100, 150], dtype = "uint8")
    upper_green= np.array([40, 155, 255], dtype = "uint8")

    mask_bird = cv2.inRange(img, lower_green, upper_green)

    bird_loc = np.average(np.where(mask_bird==255), axis=-1)

    object_locations = {
        "bird" : (int(bird_loc[1]), int(bird_loc[0])),
        "pillars" : [],
    }

    method = cv2.TM_CCOEFF_NORMED
    threshold = 0.9

    res = cv2.matchTemplate(img_gray, template1, method)

    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):

        if len(object_locations["pillars"]) == 0:
            object_locations["pillars"].append({
                'x': pt[0],
                'y1': pt[1] + template1_h,
                'y2': -1,
            })
            continue

        for pillar in object_locations["pillars"]:
            if abs(pt[0] - pillar['x']) < 20:
                break

        else:
            object_locations["pillars"].append({
                'x': pt[0],
                'y1': pt[1] + template1_h,
                'y2': -1,
            })

    
    res = cv2.matchTemplate(img_gray, template2, method)

    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):

        for pillar in object_locations["pillars"]:
            if abs(pt[0] - pillar['x']) < 20 and pillar['y2'] != -1:
                break
        else:

            for i, pillar2 in enumerate(object_locations["pillars"]):
                if abs(pt[0] - pillar2['x'] < 20 and pillar2['y2'] == -1):
                    object_locations["pillars"][i]['y2'] = pt[1]
                    break

    return object_locations

def show_object_locations(img, object_locations):

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    img = cv2.circle(img, object_locations["bird"], radius=10, color=(255, 255, 0), thickness=-1)

    for pillar in object_locations["pillars"]:

        img = cv2.circle(img, (pillar['x'], pillar['y1']), radius=10, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, (pillar['x'], pillar['y2']), radius=10, color=(255, 0, 0), thickness=-1)

    cv2.imshow('object locations', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def prediction(object_locations):

    for pillar in object_locations["pillars"]: # Order?

        if pillar["x"] > object_locations["bird"][0]:

            if object_locations["bird"][1] > pillar["y2"] - 50:
                return True

            break

    return False

def main():

    TIMING = False

    pyautogui.moveTo((LEFT + WIDTH/2), (TOP + HEIGHT/2))

    input("Enter to start")

    for i in range(200):

        if TIMING:
            start = time.time()
        
        img, gray = capture_screenshot()

        if TIMING:
            print(f'{time.time() - start}s')

        template = cv2.imread('template1.png', cv2.IMREAD_GRAYSCALE)
        template2 = cv2.imread('template2.png', cv2.IMREAD_GRAYSCALE)
        
        if TIMING:
            start = time.time()

        object_locations = find_objects(img, gray, template, template2, template.shape[::-1][1])

        if TIMING:
            print(f'{time.time() - start}s')

        # show_object_locations(img, object_locations)

        print(object_locations)

        if prediction(object_locations):
            pyautogui.click()
            time.sleep(0.1)


if __name__ == "__main__":

    main()
