import os
import tqdm
import argparse
import cv2 as cv
import random
import numpy as np

last_img1 = ''
faces1 = None
feature1 = None

def detect_align_compare(img1, img2, detector, recognizer, thresh_mode='cosine'):
    # for saving redudant computation
    global last_img1
    global faces1
    global ret1
    global aligned_face1
    global feature1

    if last_img1  != img1 :
        print('last img: {}'.format(last_img1))
        print('cur img: {}'.format(img1))
        img1_data = cv.imread(img1)
        detector.setInputSize((img1_data.shape[1], img1_data.shape[0]))
        ret1, faces1 = detector.detect(img1_data)
        # in case the detector not detect any faces
        if faces1 is None :
            print('no faces found on face1.')
            last_img1 = img1
            return -1, -1
        aligned_face1 = recognizer.alignCrop(img1_data, faces1[0])
        feature1 = recognizer.feature(aligned_face1)
        last_img1 = img1

    img2_data = cv.imread(img2)

    detector.setInputSize((img2_data.shape[1], img2_data.shape[0]))
    ret2, faces2 = detector.detect(img2_data)
    # print('detected ret{}:{}'.format(ret1, ret2))

    if faces2 is None:
        print('no faces found on face1.')
        return -2, -2

    # print('detected num of faces {}:{}'.format(len(faces1), len(faces2)))
    aligned_face2 = recognizer.alignCrop(img2_data, faces2[0])
    # aligned_face2 = recognizer.alignCrop(img2_data, faces2[0])

    feature2 = recognizer.feature(aligned_face2)

    ret_cos = 0
    ret_l2 = 0

    cosine_thresh = 0.363
    cosine_score = recognizer.match(feature1, feature2, 0)
    if cosine_score >= cosine_thresh:
        ret_cos = 1

    l2_thresh = 1.128
    l2_score = recognizer.match(feature1, feature2, 1)
    if l2_score <= l2_thresh:
        ret_l2 = 1

    return (ret_cos, ret_l2)


def test_and_eval(lfw_list_file):
    #read lfw list
    f = open(lfw_list_file, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))

    all_data = []
    for line in lines:
        all_data.append(list(line.rstrip('\r\n').split(' ')))

    # print('all data: {}'.format(all_data))

    # LFW images are fixed image size
    faceDetector = cv.FaceDetectorYN.create("/workspace/src/github/libfacedetection.train/tasks/task1/onnx/yunet.onnx", "", [250, 250])
    #TODO
    faceRecognizer = cv.FaceRecognizerSF.create("./face_recognition_sface_2021sep.onnx", "")

    TP_COS = 0
    TN_COS = 0
    TP_L2 = 0
    TN_L2 = 0
    TOTAL = 0

    num_of_person = len(all_data)
    ibar = tqdm.tqdm(all_data)
    for idx, entries_for_1_person in enumerate(ibar):
    # for idx, entries_for_1_person in enumerate(all_data):
        ibar.set_description('Processing one person ')
        # print('person detail: {}'.format(entries_for_1_person))

        num_of_entries = len(entries_for_1_person)
        if num_of_entries > 1:
            name_dir = entries_for_1_person[0]

            for i in range(1, num_of_entries):
                img1 = os.path.join(name_dir, entries_for_1_person[i])

                # more than one image for this person
                # compare with positive sample first
                for j in range(i, num_of_entries):
                    img2 = os.path.join(name_dir, entries_for_1_person[j])

                    # print('compare between positive samples {} and {}'.format(img1, img2))

                    ret, ret_l2 = detect_align_compare(img1, img2, faceDetector, faceRecognizer)

                    if ret < 0 : # in case no face detected
                        continue

                    if ret == 1:
                        TP_COS = TP_COS + 1
                    if ret == 1:
                        TP_L2 = TP_L2 + 1

                    TOTAL = TOTAL + 1

                # compare with negative
                num_compare_with_negative = 0
                #TODO: Tune the steps
                while num_compare_with_negative < 3:
                    rand_idx = random.randint(0, num_of_person)
                    if rand_idx != idx :
                        entries_of_neg = all_data[rand_idx]
                        img2 = os.path.join(entries_of_neg[0], entries_of_neg[1])

                        # print('compare between negative samples {} and {}'.format(img1, img2))

                        ret, ret_l2 = detect_align_compare(img1, img2, faceDetector, faceRecognizer)

                        # if face not found on face1 img, then should skip this img and jump out of the while
                        # to avoid dead lock
                        if ret == -1 :
                            num_compare_with_negative = 100

                        if ret == -2 : # in case no face detected on img2
                            continue

                        if ret == 0:
                            TN_COS = TN_COS + 1
                        if ret_l2 == 0:
                            TN_L2 = TN_L2 + 1

                        TOTAL = TOTAL + 1

                        num_compare_with_negative = num_compare_with_negative + 1

        if (idx > 0) and (idx % 10) == 0 :
            print('eval result')
            print('TP_COS: {}'.format(TP_COS))
            print('TN_COS: {}'.format(TN_COS))
            print('TP_L2: {}'.format(TP_COS))
            print('TN_L2: {}'.format(TN_COS))
            print('TOTAL: {}'.format(TOTAL))
            print('ACC w COS: {}'.format(float(TP_COS + TN_COS) / TOTAL))
            print('ACC w L2: {}'.format(float(TP_L2 + TN_L2) / TOTAL))

    print('eval result')
    print('TP_COS: {}'.format(TP_COS))
    print('TN_COS: {}'.format(TN_COS))
    print('TP_L2: {}'.format(TP_COS))
    print('TN_L2: {}'.format(TN_COS))
    print('TOTAL: {}'.format(TOTAL))
    print('ACC w COS: {}'.format(float(TP_COS + TN_COS) / TOTAL))
    print('ACC w L2: {}'.format(float(TP_L2 + TN_L2) / TOTAL))


def generate_list(imgs_dir, list_txt):
    print('generate list...')

    names = os.listdir(imgs_dir)
    ibar = tqdm.tqdm(names)

    f = open(list_txt, 'w')
    count = 0
    for person in ibar:
        ibar.set_description('Processing image ')

        name_dir = os.path.join(imgs_dir, person)
        person_images = os.listdir(name_dir)

        # 1 line for 1 person
        # NAME LIST_OF_PICTURE_NAMES
        f.write('{}'.format(name_dir))
        for img in person_images:
            f.write(' {}'.format(img))
        f.write('\r\n')

        count = count + 1

    f.close()

def extract_feature_and_store(lfw_list_file):
    #read lfw list
    f = open(lfw_list_file, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))

    all_data = []
    for line in lines:
        all_data.append(list(line.rstrip('\r\n').split(' ')))

    # print('all data: {}'.format(all_data))

    # LFW images are fixed image size
    faceDetector = cv.FaceDetectorYN.create("/workspace/src/github/libfacedetection.train/tasks/task1/onnx/yunet.onnx", "", [250, 250])
    faceRecognizer = cv.FaceRecognizerSF.create("./face_recognition_sface_2021sep.onnx", "")

    num_of_person = len(all_data)
    ibar = tqdm.tqdm(all_data)

    all_features = []
    for idx, entries_for_1_person in enumerate(ibar):
        ibar.set_description('Extracting feature')

        num_of_entries = len(entries_for_1_person)
        if num_of_entries > 1:
            name_dir = entries_for_1_person[0]

            p_feature = []

            for i in range(1, num_of_entries):
                img = os.path.join(name_dir, entries_for_1_person[i])
                img_data = cv.imread(img)

                faceDetector.setInputSize((img_data.shape[1], img_data.shape[0]))
                ret, faces = faceDetector.detect(img_data)

                if faces is None:
                    print('no faces found on face')
                    continue

                aligned_face = faceRecognizer.alignCrop(img_data, faces[0])

                feature = faceRecognizer.feature(aligned_face)
                p_feature.append(feature)

            if len(p_feature) > 0 :
                all_features.append(p_feature)

    #
    num_of_person = len(all_features)

    ibar = tqdm.tqdm(all_features)

    COS_SCORES = []
    L2_SCORES = []
    SAME_MASK = []

    for idx, p_features in enumerate(ibar):
        count_features = len(p_features)

        ibar.set_description('Compare feature')

        for i in range(0, count_features) :
            feature1 = p_features[i]
            #compare between same person
            for j in range(i, count_features) :
                feature2 = p_features[j]

                cosine_score = faceRecognizer.match(feature1, feature2, 0)
                l2_score = faceRecognizer.match(feature1, feature2, 1)

                COS_SCORES.append(cosine_score)
                L2_SCORES.append(l2_score)
                SAME_MASK.append(1)

            #compare between different persons
            num_compare_with_negative = 0
            #TODO: Tune the steps
            while num_compare_with_negative < 3:
                rand_idx = random.randint(0, num_of_person - 1)
                if rand_idx != idx :
                    feature2 = all_features[rand_idx][0]

                    cosine_score = faceRecognizer.match(feature1, feature2, 0)
                    l2_score = faceRecognizer.match(feature1, feature2, 1)

                    COS_SCORES.append(cosine_score)
                    L2_SCORES.append(l2_score)
                    SAME_MASK.append(0)

                    num_compare_with_negative = num_compare_with_negative + 1

    cos_threshes = np.arange(0, 4, 0.01)
    l2_threshes = np.arange(0, 4, 0.01)

    ACCS = {}
    ACCL = {}
    TP = 0
    TN = 0
    for th in cos_threshes:
        TP = 0
        TN = 0
        for idx in range(0, len(SAME_MASK)) :
            if SAME_MASK[idx] == 1 :
                if COS_SCORES[idx] >= th :
                    TP = TP + 1
            else:
                if COS_SCORES[idx] < th :
                    TP = TP + 1

        ACCS['cos-' + str(th)] = '{:.2f}'.format(float(TP + TN) / len(SAME_MASK))

    for th in l2_threshes:
        TP = 0
        TN = 0
        for idx in range(0, len(SAME_MASK)) :
            if SAME_MASK[idx] == 1 :
                if L2_SCORES[idx] <= th :
                    TP = TP + 1

            else:
                if L2_SCORES[idx] > th :
                    TP = TP + 1

        ACCL['l2-' + str(th)] = '{:.2f}'.format(float(TP + TN) / len(SAME_MASK))

        max_accs = -1.0
        max_accs_th = ''
        for k, v in ACCS.items() :
            print('Accuracy of cosine distance:')
            print('{} : {}'.format(k, v))
            if v > max_accs :
                max_accs = v
                max_accs_th = k

        max_accl = -1.0
        max_accl_th = ''
        for k, v in ACCL.items() :
            print('Accuracy of l2 distanec:')
            print('{} : {}'.format(k, v))
            if v > max_accl :
                max_accl = v
                max_accl_th = k

        print('Max accuracy of cosine {}:{}'.format(max_accs_th, max_accs))
        print('Max accuracy of l2 {}:{}'.format(max_accl_th, max_accl))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs', default='/workspace/dataset/lfw/lfw', help='path to images dir to be tested')
    parser.add_argument('-g', '--generate_list', action='store_true', help='generate lfw file list only')
    parser.add_argument('-c', '--feature_compare', action='store_true', help='extract and compare feature')

    args = parser.parse_args()

    list_txt = './lfw_list.txt'

    if args.generate_list :
        generate_list(args.imgs, list_txt)
    else:
        if args.feature_compare:
            extract_feature_and_store(list_txt)
        else:
            test_and_eval(list_txt)
