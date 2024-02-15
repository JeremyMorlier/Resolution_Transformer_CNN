import sys
sys.path.append("/users/local/j20morli/Resolution_Tranformer_CNN/")

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
import third_party.ADE20K_utils.utils.utils_ade20k as ade20k
from mobile_sam import SamPredictor, sam_model_registry
import glob
import torch

DATASET_PATH = '/nasbrain/datasets/ADE20k_full/ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

nfiles = len(index_ade20k['filename'])
root_path = '/nasbrain/datasets/ADE20k_full'

sam_checkpoint = "/users/local/j20morli/data/checkpoints/retrained_mobilesam.pth"
sam_visual_checkpoint = "/users/local/j20morli/data/checkpoints/results/vit_b_vit_t_sam10000_110000.pth"
sam_visual_checkpoint = "/users/local/j20morli/data/checkpoints/img_encoder.pth"
sam_visual_checkpoint = "/nasbrain/j20morli/sam_checkpoints/SAM_dataset_vit_h_vit_t_1_3.pth"
sam_visual_checkpoint = "/users/local/j20morli/data/checkpoints/SAM_dataset_vit_h_vit_t_1_10.pth"
tiny_sam_checkpoint = "/users/local/j20morli/data_foundation/third_party/MobileSAM/weights/mobile_sam.pt"

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = "vit_b"
student_model = "vit_t"
# sam = sam_model_registry[model](checkpoint=sam_checkpoint)
# sam.to(device=device)

# sam_tmp = sam_model_registry[model](checkpoint=sam_checkpoint)
# # sam_tmp.image_encoder.load_state_dict(torch.load(sam_visual_checkpoint))
# sam_tmp.to(device=device)

sam_tmp = sam_model_registry[student_model](checkpoint=sam_checkpoint)
sam_tmp.image_encoder.load_state_dict(torch.load(sam_visual_checkpoint))
sam_tmp.to(device=device)
#sam_tmp.image_encoder.target_img_size = 1024

#student_encoder = sam_tmp.image_encoder

#sam_tiny = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
#sam_tiny.to(device=device)
#student_encoder = sam_tiny.image_encoder

#print(student_encoder)
#student_encoder = torch.load("/users2/local/r17bensa/mobilesam_lowres_19.pth")
#print("##############################################")
#print(student_encoder)
#sam_tiny.image_encoder = student_encoder
#student_encoder.eval()
#sam_tiny.image_encoder.target_img_size = 1024



def find_center(img): 
    img_binary = (img != 0)*1
    if np.sum(img_binary) > 100 : 
        mass_x, mass_y = np.where(img_binary == 1)
        cent_x = int(np.average(mass_x))
        cent_y = int(np.average(mass_y))
        center_point = np.array([[cent_y, cent_x]]) 
        if img_binary[cent_x, cent_y] == 1 : 
            valid_mask = True
        else : 
            valid_mask = False
        return center_point, valid_mask
    else : 
        return np.array([0,0]), False 

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def calculate_iou(segmentation1, segmentation2):
    intersection = np.sum(segmentation1 * segmentation2)
    union = np.sum(segmentation1) + np.sum(segmentation2) - intersection
    iou = (intersection / union) * 100
    return iou


iou_liste = []
nb_crops = 0
for i in range(5000) : 
    print(i)
    full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])
    folder_name = '{}/{}'.format(root_path,full_file_name.replace(".jpg", ''))
    print(folder_name)
    folder_files = glob.glob(f"{folder_name}/*")

    file_name = index_ade20k['filename'][i]
    info = ade20k.loadAde20K('{}/{}'.format(root_path, full_file_name))
    image = cv2.imread(info['img_name'])[:,:,::-1]
    #plt.figure()
    #plt.imshow(image)
    #plt.savefig("images_test/image.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg = cv2.imread(info['segm_name'])[:,:,::-1]
    #plt.figure()
    #plt.imshow(seg)
    #plt.savefig("images_test/mask.png")
    predictor = SamPredictor(sam_tmp)
    predictor.set_image(image)
    input_label = np.array([1])
    for i, image_test in enumerate(folder_files) : 
        aux = cv2.imread(image_test)[:,:,0]
        label = (aux != 0)*1
        center_point, valid_mask = find_center(aux)
        #plt.figure()
        #show_points(center_point, np.array([1]), plt.gca())
        #plt.imshow(aux)
        #plt.savefig(f"images_test/mask_{i}.png")
        if valid_mask : 
            mask, _, _ = predictor.predict(
            point_coords=center_point,
            point_labels=input_label,
            multimask_output=False,
            )
            print(calculate_iou(mask*1,label*1))
            iou_liste.append(calculate_iou(mask*1,label*1))
            print(np.mean(iou_liste))
            nb_crops += 1
            print("###############################")
print("CROOOOOOOOOOOOOOOOOOOOPS")
print(nb_crops)



