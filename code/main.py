import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from missc.inception_score import get_inception_score
from missc.config import cfg
from torchvision import utils
from itertools import chain
from PIL import Image
import os

#if channels are 3,please change writer and config

def save_model(G,D,epoch):
    torch.save(G, '../save_model/generator_{}.pth'.format(epoch))
    torch.save(D, '../save_model/discriminator_{}.pth'.format(epoch))




class MyData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = Image.open(img_path)  # 读取该图片
        #img = img.convert("L")
        if self.transform:
            img = self.transform(img)
            #img = img.transform.Resize((32,32))
        return img  # 返回该样本




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transforms = transforms.Compose(
        [
            transforms.Resize(cfg.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(cfg.CHANNELS_IMG)], [0.5 for _ in range(cfg.CHANNELS_IMG)])
        ])

    # 数据集
    dataset = MyData(root_dir=r"..\data\stmap", transform=transforms)                 # RGB

    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # initialize Generator and Discriminator
    G = Generator(cfg.Z_DIM, cfg.CHANNELS_IMG, cfg.FEATURES_G).to(device)
    D = Discriminator(cfg.CHANNELS_IMG, cfg.FEATURES_D).to(device)

    # initialize
    initialize_weights(G)
    initialize_weights(D)


    opt_G = optim.Adam(G.parameters(), lr=cfg.GENERATOR_LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg.DISCRIMINATOR_LR, betas=(0.5, 0.999))
  
    # loss function
    criterion = nn.BCELoss()
    # noise
    fixed_noise = torch.randn(32, cfg.Z_DIM, 1, 1).to(device)



    writer = SummaryWriter(f"../result/log1/loss")


    # 模型改为训练模式（启用BN层）
    G.train()
    D.train()
    step = 0

    # Start training
    for epoch in range(cfg.NUM_EPOCH):
        for batch_idx, (real) in enumerate(loader):                                         # iterator datasets
            real = real.to(device)
            noise = torch.randn(cfg.BATCH_SIZE, cfg.Z_DIM, 1, 1).to(device)
            fake = G(noise)

            # max log(D(x)) + log(1 - D(G(z)))
            D_real = D(real).reshape(-1)
           # print(D_real.size())
            loss_D_real = criterion(D_real, torch.ones_like(D_real))                  # ---> log(D(real))
            D_fake = D(fake.detach()).reshape(-1)
            loss_D_fake = criterion(D_fake, torch.zeros_like(D_fake))                 # ---> log(1-D(G(z)))
            loss_D = (loss_D_fake + loss_D_real)/2

            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            #  min log(1-D(G))------ max log(D(G(z)))
            output = D(fake).reshape(-1)
            loss_G = criterion(output, torch.ones_like(output))                              # ————> log(D(G(z))
            G.zero_grad()
            loss_G.backward()
            opt_G.step()


            if batch_idx % 10== 0:
                sample_list = []
                for i in range(10):
                    z = torch.randn(32, cfg.Z_DIM, 1, 1).to(device)
                    samples = G(z)
                    sample_list.append(samples.data.cpu().numpy())

                # Flattening list of lists into one list of numpy arrays
                new_sample_list = list(chain.from_iterable(sample_list))
                print("Calculating Inception Score over 8k Generated images")
                # Feeding list of numpy arrays
                inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                                                      resize=True, splits=10)
                writer.add_scalar("Inception score:", inception_score, step)
                print(inception_score)


                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')


                print(
                    f"Epoch [{epoch}/{cfg.NUM_EPOCH}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_D:.4f}, loss G: {loss_G:.4f} "
                )
                with torch.no_grad():
                    fake = G(fixed_noise)
                    # 输出32个示例
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    writer.add_image("Real", img_grid_real, global_step=step)
                    writer.add_image("Fake", img_grid_fake, global_step=step)
                    utils.save_image(img_grid_fake, 'training_result_images/img_Generator_iter_{}.png'.format(step))
                    writer.add_scalar("G loss:", loss_G, step)
                    writer.add_scalar("D loss:", loss_D, step)
                step += 1



        save_model(G, D, epoch)

    writer.close()
    # 显示结果 tensorboard --logdir="G:\DCGAN\result\log1"