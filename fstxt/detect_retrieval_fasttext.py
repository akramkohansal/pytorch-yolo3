import sys
import time
import os
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from pyfasttext import FastText

def detect(cfgfile, weightfile, imgfolder):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    namesfile = 'data/recognition.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)
    image_list = os.listdir(imgfolder)

    # words, neigh = KNNclassifier()

    for imgfile in image_list:
        img_full_path = imgfolder+imgfile
        img = Image.open(img_full_path).convert('RGB')
        sized = img.resize((m.width, m.height))

        # Paper -> conf_t = 0.0025 and no NMS
        #conf_threshold = 0.0025
        #nms_threshold = 0.0013

        conf_threshold = 0.025
        nms_threshold = 0

        # MOST OF EXPERIMENTS:
        #conf_threshold = 0.025
        #nms_threshold = 0.05

        # ORIGINAL?
        # conf_threshold = 0.1
        # nms_threshold = 0.2
        for i in range(1):
            start = time.time()
            # boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
            boxes = do_detect_retrieval(m, sized, conf_threshold, nms_threshold, use_cuda)
            finish = time.time()

            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        #result_image_path = '/media/amafla/ssd/pytorch-yolo2-master/results/sports10k_results_32cluster/' + imgfile
        #result_image_path = '/media/amafla/ssd/pytorch-yolo2-master/results/svt1_results_32cluster/' + imgfile
        result_image_path = '/media/amafla/ssd/pytorch-yolo2-master/results/iiit_results_36cluster/' + imgfile

        # plot_boxes(img, boxes, words, neigh, result_image_path, class_names)
        write_retrieval(img, boxes, result_image_path, class_names)


if __name__ == '__main__':


    imgfolder = '/home/amafla/Documents/Datasets/IIIT_STR_V1.0/imgDatabase/'
    #imgfolder = '/home/amafla/Documents/Datasets/sports10K/imgDatabase/'
    #imgfolder = '/home/amafla/Documents/Datasets/svt1/img/'
    #imgfolder = '/home/amafla/Documents/Datasets/IC13/test/'

    #cfgfile = 'cfg/convtest.cfg'
    cfgfile = 'cfg/yolo-fasttext-13anchors.cfg'

    weightfile = 'backup/cluster36.weights'
    #weightfile = 'bin/yolo-phoc.weights'

    detect(cfgfile, weightfile, imgfolder)
    print ("OPERATION COMPLETE..!!")
    '''
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
    '''
