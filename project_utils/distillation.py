
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision.transforms import v2

import PIL
import io

import wandb

def convert_to_rgb(image):
    return image.convert('RGB')

transform_pilTensor = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])    
    
class Adapter(nn.Module) :
    def __init__(self, in_features, out_features) :
        super(Adapter, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=None)
    
    def forward(self, input) :
        return self.linear(input)


def distillate_one(name, teacher, student, criterion, optimizer, scheduler, device, dataloader, epochs, data_name, batch_size, teacher_preprocess, student_preprocess, student_dim, teacher_dim, checkpoints=[]) :
    adaptor = None
    if student_dim != teacher_dim :
        adaptor = Adapter(student_dim, teacher_dim).to(device)
        adaptor.train()

    results = []

    student.train()
    teacher.eval()
    for epoch in range(1, epochs+1) :
        
        running_loss = 0.0

        optimizer.zero_grad()
        print("Epoch : ", epoch)
        for i, inputs in enumerate(dataloader) :

            # Transform inputs based on dataset class
            if data_name == "yfcc" :
                student_temp = []
                teacher_temp = []
                for image in inputs["img"] :
                    temp = PIL.Image.open(io.BytesIO(image))
                    student_temp.append(student_preprocess(temp))
                    teacher_temp.append(teacher_preprocess(temp))

                student_inputs = torch.stack((student_temp)).to(device)
                teacher_inputs = torch.stack((teacher_temp)).to(device)
            else :
                inputs = inputs.to(device)
                student_inputs = inputs
                teacher_inputs = inputs

            student_outputs = student(student_inputs)
            if adaptor != None :
                student_outputs = adaptor(student_outputs)
                
            with torch.no_grad() :
                teacher_outputs = teacher(teacher_inputs)

            loss = criterion(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()

            sys.stdout.write(f'\r {time.strftime("%H:%M:%S", time.gmtime())} {name} : {epoch}/{epochs} - {i}/{len(dataloader)} - loss {round(loss.item() / (batch_size), 3)} ' f' - running loss {round(running_loss / ((i + 1) * batch_size), 3)}')
            wandb.log({"loss":loss.item() / (batch_size), "running_loss":running_loss / ((i + 1) * batch_size)})
        scheduler.step()

        results.append(running_loss)
        if epoch in checkpoints :
            save_name = name + "_" + str(epoch) + ".pth"
            torch.save(student.state_dict() , save_name)
            print(" \r\nSave at " + str(epoch) + " epochs      File Name : " + save_name)
            print(running_loss)
            
    return results