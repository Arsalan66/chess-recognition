# from PIL.Image import new
import os
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
import statistics
from tkinter import *
# from PIL import Image, ImageTk

# # img1 = cv.imread('board_images/5_1.jpg')
# img1 = cv.imread('board_images/1B2b3-Kp6-8-8-4k3-8-8-8.JPG')
def find_med_smallest_dist_between_pts(pts):
    all_smallest = []
    for pt in pts:
        smallest = 10000 
        for pt2 in pts:
            if pt != pt2:
                d = dist(pt, pt2)
                if d < smallest: smallest = d
        all_smallest.append(smallest)
        return statistics.median(all_smallest)

def dist(i,p): # finds distance between pts, kinda
    # return ((i[0] - p[0])**2 + (i[1] - p[1])**2)**0.5
    res = (abs(i[0]-p[0]) + abs(i[1]-p[1]))
    if (res == 0): return 100 # this is just to exclude duplicatepts
    return res

def remove_duplicates(list): # list = set(list)
    # return [p for p in list if all(dist(i,p) > 3 for i in list)] 
    out = []
    l = len(list)
    for i in range(l): 
        c = True
        if all(dist(list[i], p) > 6 for p in out): 
            out.append(list[i])
        return out
def find_intersections(lines, warped_img):
    new_lines_v = [] 
    new_lines_h = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x1 == x2): 
            m = 10000
        else: m = abs((y2-y1)/(x2-x1)) # print(x1, y1, x2, y2, m)
        if (m < .5): # Horizontal 
            new_lines_h.append((y1+y2)/2)
        elif (m>35): # Vertical 
            new_lines_v.append((x1+x2)/2)
        else:
            cv.line(warped_img, (x1,y1), (x2,y2), (255,100,50), 1)

    pts = []
    for hline in new_lines_h:
        for vline in new_lines_v: 
            pts.append([hline, vline])

    return pts, warped_img

def hough_pts(img1):

    width = int(img1.shape[1] * 0.2) # print(width)
    height = int(img1.shape[0] * 0.2) # print(height)
    img1 = cv.resize(img1, (width, height))


    image = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    d, img = cv.threshold(image, 140, 255, cv.THRESH_BINARY)

    num_labels, labels, stats, centroids= cv.connectedComponentsWithStats(img, 8, cv.CV_32S)
    m=1
    for i in range(2,num_labels-1):
        if stats[m][cv.CC_STAT_AREA] < stats[i][cv.CC_STAT_AREA]: 
            m = i

    contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    a=contours[0]
    for contour in contours:
        if cv.contourArea(contour) > cv.contourArea(a): 
            a = contour

    br = 0 #bottom right is max sum
    tl = 1000000 # top left is min sum
    tr = 0 # top right is smallest difference
    bl = 1000000 # bottom left is largest difference brp = [0,0]
    trp = [0,0]
    tlp = [0,0]
    trp = [0,0]
    for point in a:
        # print(point)
        sum = point[0][0] + point[0][1] 
        diff = point[0][0] - point[0][1] 
        if sum > br:
            br = sum
            brp = point[0]
        if sum < tl:
            tl = sum
            tlp = point[0]
        if diff > tr:
            tr = diff
            trp = point[0]
        if diff < bl:
            bl = diff
            blp = point[0]
	
    rect = np.array(((tlp[0]+5, tlp[1]+5), (trp[0]-5, trp[1]+5), (brp[0]-5, brp[1]-5), (blp[0]+5, blp[1]-5)), dtype="float32") 
    width = 200
    height = 200
    dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-
        1]],dtype="float32")
    M = cv.getPerspectiveTransform(rect,dst)
    warped_img = cv.warpPerspective(image, M, (width, height)) 
    edges = cv.Canny(warped_img, 100, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=40, maxLineGap=70)
    pts, warped_img = find_intersections(lines, warped_img) 
    pts = remove_duplicates(pts)
    for pt in pts: 
        # print(pt)
        cv.circle(warped_img, (int(pt[1]), int(pt[0])), 3, (0,255,0), 1)
    for line in lines:
        # x1, y1, x2, y2, l, p, n = line 
        x1, y1, x2, y2 = line[0]

        # print(line)
        cv.line(warped_img, (x1,y1), (x2,y2), (0,0,255), 1) 
    return warped_img

def load_images():
    images = [] 
    cnt = 1
    for filename in os.listdir('test_data/'):
        img = cv.imread(os.path.join('test_data/',filename)) 
        print("loading image: ", cnt, " file: ", filename) 
        cnt+=1
    if img is not None: 
        images.append(img)
    return images

def hough_images(images): 
    cnt = 1
    warped_imgs = []
    for image in images: 
        print("warping image ", cnt) 
        cnt+=1
        # try:
        warped = hough_pts(image) 
        warped_imgs.append(warped) # except (Exception):
        # print("error warping image ", cnt)
    return warped_imgs 

images = load_images()
warped_images = hough_images(images=images)

def plot_figs(images, hough_images):
    split_images = [images[i:i+4] for i in range(0, len(images), 4)] 
    split_hough = [hough_images[i:i+4] for i in range(0, len(hough_images),4)] 
    c = 0
    fig_num = 1
    for i in range(len(split_hough)): 
        fig = plt.figure() 
        plt.tight_layout()
        cnt = 1
        for j in range(len(split_hough[i])): 
            fig.add_subplot(len(split_images[i]), 2, cnt) 
            plt.imshow(split_images[i][j]) 
            plt.axis('off')
            plt.title("Original #" + str(fig_num))
            fig.add_subplot(len(split_hough[i]), 2, cnt+1) 
            plt.imshow(split_hough[i][j])
            plt.axis('off')
            plt.title("Hough_result #" + str(fig_num))
            cnt+=2 
            fig_num+=1
        c+=1

plot_figs(images, warped_images) 
plt.show()