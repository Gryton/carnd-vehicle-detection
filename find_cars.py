import numpy as np
import matplotlib.image as mpimg
import cv2
import glob
import os
import pickle
import project
import time
from scipy.ndimage.measurements import label

# line below seems to be needed if training on .png and searching in .jpg
# img = img.astype(np.float32)/255

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        color = [0, 0, 0]
        color[car_number % 3] = 255
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    return img


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              spatial_feat=True, hist_feat=True, hog_feat=True):
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    # ctrans_tosearch = project.convert_color(img_tosearch, conv='RGB2YCrCb')
    # ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = project.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = project.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = project.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    on_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = project.bin_spatial(subimg, size=spatial_size)
            hist_features = project.color_hist(subimg, nbins=hist_bins)
            features = []
            if spatial_feat == True:
                features.append(spatial_features)
            if hist_feat == True:
                features.append(hist_features)
            if hog_feat == True:
                features.append(hog_features)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack(features).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            # test_prediction = svc.predict(test_features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                on_windows.append(((xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return on_windows
    # return draw_img


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = project.bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = project.color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(project.get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = project.get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def find_cars_pipeline(img):
    ystart = 400
    ystop = 656
    hot_windows = find_cars(img, ystart, ystop, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    hot_windows += find_cars(img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    hot_windows += find_cars(img, ystart, ystop, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float32)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    out_img = draw_labeled_bboxes(np.copy(img), labels)
    return out_img


dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
spatial_feat = dist_pickle["spatial_feat"]

ystart = 400
ystop = 656
scale = 1.5


class Vehicle():
    def __init__(self):
        self.recent_bbox = None
        self.all_bboxes = []


class Frame():
    def __init__(self):
        self._current_heat = None
        self.n_heats = []
        self._current_labels = None
        self.n_labels = []
        self.n = 3

    @property
    def current_heat(self):
        return self._current_heat

    @current_heat.setter
    def current_heat(self, heat):
        self.n_heats.append(heat)
        if len(self.n_heats) > self.n:
            self.n_heats.pop(0)
        self._current_heat = heat

    @property
    def current_labels(self):
        return self._current_labels

    @current_labels.setter
    def current_labels(self, labels):
        self.n_labels.append(labels[0])
        if len(self.n_labels) > self.n:
            self.n_labels.pop(0)
        self._current_labels = labels

    @property
    def avg_heat(self):
        return np.mean(self.n_heats, axis=0, dtype=int)

    @property
    def avg_labels(self):
        return np.mean(self.n_labels, axis=0, dtype=int)


scales = [1, 1.5, 2, 2.5, 3]
# find_cars = project.find_cars
images = glob.glob('./test_images/*39*.jpg')
frame = Frame()
for image_name in images:
# img = mpimg.imread('test_images/test1.jpg')
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hot_windows = []
    start = time.time()
    for scale in scales:
        hot_windows += find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins, spatial_feat=spatial_feat)
    # cv2.imwrite('./output_images/windows.jpg'.format(os.path.basename(image_name)), cv2.cvtColor(draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6), cv2.COLOR_RGB2BGR))
    # hot_windows += find_cars(img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins, spatial_feat=spatial_feat)
    # hot_windows += find_cars(img, ystart, ystop, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins, spatial_feat=spatial_feat)
    print('find cars time:', time.time() - start)
    # out_img1 = find_cars(img, ystart, ystop, 3, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)
    # out_img2 = find_cars(img, ystart, ystop, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float32)
    heat = add_heat(heat, hot_windows)
    frame.current_heat = heat
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    cv2.imwrite('./output_images/avg_heat_{0}'.format(os.path.basename(image_name)), heat * 25)
    labels = label(heatmap)
    frame.current_labels = labels
    # out_img = draw_labeled_bboxes(np.copy(img), frame.avg_labels)
    out_img_avg_heat = draw_labeled_bboxes(np.copy(img), label(apply_threshold(frame.avg_heat, 2)))
    # cv2.imwrite('./output_images/avg_{0}'.format(os.path.basename(image_name)), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./output_images/avg_heat_labels_{0}'.format(os.path.basename(image_name)), cv2.cvtColor(out_img_avg_heat, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./output_images/scale1_{0}'.format(os.path.basename(image_name)), cv2.cvtColor(out_img1, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./output_images/scale05_{0}'.format(os.path.basename(image_name)), cv2.cvtColor(out_img2, cv2.COLOR_RGB2BGR))
# plt.imshow(out_img)
#     draw_image = np.copy(img)


#     start = time.time()
#     windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[ystart, ystop],
#                         xy_window=(64, 64), xy_overlap=(0.5, 0.5))
#     windows += slide_window(img, x_start_stop=[None, None], y_start_stop=[ystart, ystop],
#                         xy_window=(96, 96), xy_overlap=(0.5, 0.5))
#     windows += slide_window(img, x_start_stop=[None, None], y_start_stop=[ystart, ystop],
#                         xy_window=(128, 128), xy_overlap=(0.5, 0.5))
#
#     hot_windows = search_windows(img, windows, svc, X_scaler, color_space='YCrCb',
#                             spatial_size=spatial_size, hist_bins=hist_bins,
#                             orient=orient, pix_per_cell=pix_per_cell,
#                             cell_per_block=cell_per_block,
#                             hog_channel='ALL', spatial_feat=True,
#                             hist_feat=True, hog_feat=True)
#     heat = np.zeros_like(img[:,:,0]).astype(np.float32)
#     heat = add_heat(heat, hot_windows)
#     heat = apply_threshold(heat,0)
#     heatmap = np.clip(heat, 0, 255)
#     # cv2.imshow('heatmap', heatmap)
#     cv2.imwrite('./output_images/heatmap_{0}'.format(os.path.basename(image_name)), cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR))
#     labels = label(heatmap)
#     draw_image = draw_labeled_bboxes(np.copy(img), labels)
#     window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
#     print('seatch_windows time: ', time.time()-start)
#     cv2.imwrite('./output_images/find_single__{0}'.format(os.path.basename(image_name)), cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))