from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from EMNIST_Nets import *
from EMNIST_labels import *

print("initialized alphabet_predict.py")

imagePath = input()
imageSize = int(input())
analizeType = input()
num = int(input())

parent = Path(__file__).resolve().parent
data = {
    28 : {
        "balanced" : "data/EMNIST_28_balanced_weights.pth",
        "byclass" : "data/EMNIST_28_byclass_weights.pth",
        "letters" : "data/EMNIST_28_letters_weights.pth"
        },
    56 : {
        "balanced" : "data/EMNIST_56_balanced_weights.pth",
        "byclass" : "data/EMNIST_56_byclass_weights.pth",
        "letters" : "data/EMNIST_56_letters_weights.pth"

        }
}

def picture_predict(img, img_size, pred_type, n, use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    model=Net[img_size][pred_type]().to(device)
    model.load_state_dict(torch.load(parent.joinpath(data[img_size][pred_type]), map_location=device))
    model.eval()

    img = Image.open(img)
    img = ImageOps.invert(img.convert("L"))
    img_box = img.getbbox()
    if img_box == None:
        return None, None, 0
    x1, y1, x2, y2 = img_box
    x_size, y_size = x2-x1, y2-y1
    size = max(x_size, y_size)
    x_padsize = int((size-x_size)/2)
    y_padsize = int((size-y_size)/2)
    img = img.crop(img_box)
    transform = transforms.Compose([
        transforms.Pad(padding=(x_padsize,y_padsize)),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomRotation(degrees = (90,90)),
        transforms.ToTensor()
    ])
    img = transform(img)

    img = torch.reshape(img, (1,1,img_size,img_size)).to(device)

    with torch.no_grad():
        Soft = nn.Softmax(dim=1)
        pred = Soft(model(img))
        if n == 0:
            n = torch.sum(pred[0]>0.001)
        value, pred = pred[0].topk(k=n)
    return value, pred, n

if __name__ == "__main__":
    pred_type = analizeType
    value, pred, n = picture_predict(imagePath, imageSize, pred_type, num)
    if n == 0:
        print("Noting")
    else:
        for i in range(n):
            print("%s:%f" % (label[pred_type][int(pred[i])], float(value[i])))

print("fin")