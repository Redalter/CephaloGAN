import torch
import cv2


def encode_input(label_map):

        # create one-hot vector for label map
        label_map = torch.tensor(label_map)
        size1 = label_map.size()
        oneHot_size = (size1[0], 0, size1[1], size1[2])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
        return input_label





if __name__=='__main__':
    image_path = r"F:\pytorch-CycleGAN-and-pix2pix-master\pix2pixHD-master\pix2pixHD-master\datasets\cephalo\train_A\1.jpg"
    image = cv2.imread(image_path)
    img = image[:, :, ::-1].transpose(2, 0, 1)


    a = encode_input(img)
    print(a.size())