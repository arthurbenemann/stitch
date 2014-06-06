import numpy
import cv2
 
 
######################################################
# Util 
######################################################
def showImage(img, title = '', scale = 1 ):
    cv2.imshow(title,cv2.resize(img,(0,0),fx = scale, fy = scale))

###############################################################################
# Image Matching
###############################################################################

def match_images(img1, img2):
    """Given two images, returns the matches"""
    detector = cv2.SURF(5000, 5, 5, upright = True)  # @UndefinedVariable
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
    vis = numpy.zeros((h1 + h2, max(w1, w2),3), numpy.uint8)
    vis[:h1, :w1,:] = img1
    vis[h1:h1 + h2, :w2,:] = img2
 
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
 
    showImage(vis,'matches',0.1)
    
    
def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = numpy.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = numpy.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = numpy.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = numpy.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = numpy.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = numpy.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return   result;
  
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
        
        
    result = warpTwoImages(img2, img1, H)
    showImage(result,scale = 0.2)
    

###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    
    img1 = cv2.imread('IMG_5410.JPG')  # queryImage
    img2 = cv2.imread('IMG_5412.JPG')  # trainImage

    kp_pairs = match_images(img1, img2)
    
    if kp_pairs:
        draw_matches('find_obj', kp_pairs, img1, img2)
    else:
        print "No matches found"
    print 'finished'
    
    while True:
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
