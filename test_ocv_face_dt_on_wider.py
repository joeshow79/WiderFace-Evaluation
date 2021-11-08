import os
import tqdm
import argparse
import cv2 as cv

def do_test(imgs_dir):
    events = os.listdir(imgs_dir)
    ibar = tqdm.tqdm(events)

    for event in ibar:
        ibar.set_description('Processing image ')
        event_dir = os.path.join(imgs_dir, event)
        event_images = os.listdir(event_dir)
        for img in event_images:
            img_name = os.path.join(event_dir, img)
            # TODO: inference on this image
            print('img_name: {}'.format(img_name))
            img_data = cv.imread(img_name)
            faceDetector = cv.FaceDetectorYN.create("/workspace/src/github/libfacedetection.train/tasks/task1/onnx/yunet.onnx", "", img_data.shape[:2])
            faceDetector.setInputSize((img_data.shape[1], img_data.shape[0]))

            ret, faces = faceDetector.detect(img_data)

            #print('faces: {}'.format(faces))
            # The output is as below
            # faces: (1, array([[ 411.48217773,  361.2265625 ,  125.81674194,  129.13446045,
            # 460.88241577,  389.42178345,  490.09075928,  423.27612305,
            # 451.96624756,  423.83236694,  422.34909058,  429.95529175,
            # 447.07940674,  458.84817505,    0.99273545]], dtype=float32))
            # Write the inference result into txt
            img_txt = img_name.rstrip('.jpg')
            f = open(img_txt, 'w')
            f.write('{}\r\n'.format(img))
            f.write('bboxes:\r\n')

            if faces is not None:
                for idx, face in enumerate(faces):
                    f.write('{} {} {} {} {}\r\n'.format(face[0], face[1], face[2], face[3], face[14]))
                    
            f.close()

            # break

        # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs', default='./WIDER_t/WIDER_val/images/')

    args = parser.parse_args()
    do_test(args.imgs)
