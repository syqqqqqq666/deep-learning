from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


img_path ="data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
writer =SummaryWriter("logs")

tensor_trains=transforms.ToTensor()
tensor_img = tensor_trains(img)
writer.add_image("img", tensor_img)
writer.close()