
from segment_anything import SamPredictor, sam_model_registry
import segment_anything as sam
import MobileSAM as msam


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

def evaluate(ade20k_dataloader) :

    sam_model = 
    for i, (images, targets) in enumerate(ade20k_dataloader) :
        predictor = sam.SamPredictor(sam_model)
        predictor.set_torch_image(images, upscale= True)
        input_label = np.array([1])