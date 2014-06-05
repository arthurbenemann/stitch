import numpy
import cv2
 
import sys
from matplotlib import pyplot as plt
 
###############################################################################
# Image Matching
###############################################################################

def match_images(img1, img2):
    """Given two images, returns the matches"""
    detector = cv2.SURF(5000, 5, 5, upright = True)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
 
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))
 
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp_pairs
 
def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs


###############################################################################
# Match Diplaying
###############################################################################
 
def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((h1 + h2, max(w1, w2)), numpy.uint8)
    vis[:h1, :w1] = img1
    vis[h1:h1 + h2, :w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
 
    if H is not None:
        corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (0, h1))
        cv2.polylines(vis, [corners], True, (255, 255, 255), thickness=20)
 
    if status is None:
        status = numpy.ones(len(kp_pairs), numpy.bool_)
    p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (0, h1)
 
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 5, col, -1)
            cv2.circle(vis, (x2, y2), 5, col, -1)
        else:
            col = red
            r = 2
            thickness = 6
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green, thickness = 10)
 
    plt.imshow(vis), plt.show()
    
    
def merge_images(image1, image2, homography, size, offset):

  ## Combine the two images into one.
  panorama = cv2.warpPerspective(image2,homography,size)
  (h1, w1) = image1.shape[:2]

  for h in range(h1):
    for w in range(w1):
        if image1[h][w][0] != 0 or image1[h][w][3] != 0 or image1[h][w][4] != 0:
            panorama[h+offset[1]][w + offset[0]] = image1[h][w]

  return panorama
  
def calculate_size(img1, img2, homography):
    ## Calculate the size and offset of the stitched panorama.
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    offset = abs((homography*(w2-1,h2-1,1))[0:2,2]) 
    print offset
    size   = (h1 + int(offset[0]), w1 + int(offset[1]))
    if (homography*(0,0,1))[0][1] > 0:
        offset[0] = 0
    if (homography*(0,0,1))[1][2] > 0:
        offset[1] = 0

    ## Update the homography to shift by the offset
    homography[0:2,2] +=  offset
    return (size, offset)  
  
  
def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)
    
    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])
    
    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (numpy.sum(status), len(status))
    else:
        H, status = None, None
        print '%d matches found, not enough for homography estimation' % len(p1)
    
    if len(p1):
        explore_match(window_name, img1, img2, kp_pairs, status, H)
        
    if H is not None:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        w1 = w1*2
        base_img_warp = cv2.warpPerspective(img1, H, (w1,h1))
        print "Warped base image"
        for h in range(h2):
            for w in range(w2):
                if base_img_warp[h][w] <=0:
                    base_img_warp[h][w] = img2[h][w]
                    
        #size,offset = calculate_size(img1, img2, H)
        #merge_images(img1, img2, H, size, offset)
        plt.imshow(base_img_warp), plt.show()
    
###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    
    img1 = cv2.imread('/home/arthur/workspaces/imagesync/Sticher/src/IMG_5410.JPG', 0)  # queryImage
    img2 = cv2.imread('/home/arthur/workspaces/imagesync/Sticher/src/IMG_5412.JPG', 0)  # trainImage

    kp_pairs = match_images(img1, img2)
    
    if kp_pairs:
        draw_matches('find_obj', kp_pairs, img1, img2)
    else:
        print "No matches found"
    print 'finished'
