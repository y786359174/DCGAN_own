import os
import time
import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
from network import Generator,  Discriminator



import cv2
import einops
import numpy as np

def train(gen:Generator, dis:Discriminator, ckpt_path, device = 'cuda'):
    
    n_epochs = 300
    batch_size = 512
    lr = 0.0002
    beta1 = 0.5
    criterion = nn.BCELoss().to(device)         # 二元交叉熵损失，因为此时只有0和1的概率，且和为100%，所以可以不算向量，只算一个值（反正另一个也能间接求出来。
                                                # Loss = -w * [p * log(q) + (1-p) * log(1-q)]
                                                # 这东西也能放在cuda中啊
    dataloader = get_dataloader(batch_size)
    optim_gen = torch.optim.Adam(gen.parameters(), lr, betas=(beta1, 0.999))    # betas 是一个优化参数，简单来说就是考虑近期的梯度信息的程度
    optim_dis = torch.optim.Adam(dis.parameters(), lr, betas=(beta1, 0.999))
    label_fake = torch.full((batch_size,), 0., device=device)       # 真实图是1，虚假是0,需要注意,这里用的时候是计算loss，是小数，要用1. 和0.
    label_real = torch.full((batch_size,), 1., device=device)       # 还有一件事，这里不能随意用batch_size，因为最后可能不满512，但是还是会继续算。得实时读取x的bn大小
                                                                    # 一般四个维度分别是bn c h w
                                                                    # 我把他从循环中挪出来并且检测x的bn不是batch_size就跳过

    for epoch_i in range(n_epochs):
        tic = time.time()
        # i_tmp  = 0
        for x,_ in dataloader:
            if(x.shape[0]!=batch_size):
                continue
            # print('i=',i_tmp)
            # i_tmp+=1
            img_real = x.to(device)

            
            z = torch.randn(batch_size, gen.nz, 1, 1, device=device)   # gen用的噪声向量
            # 训练Gen
            gen.zero_grad()                                            # 这个和optim_gen.zero_grad()有什么区别？现在没啥区别因为一个model就对应一个optim
            img_fake = gen(z)
            
            g_loss = criterion(dis(img_fake), label_fake)  
            g_loss.backward()
            optim_gen.step()
            
            # 训练Dis
            dis.zero_grad()

            d_loss_real = criterion(dis(img_real), label_real)
            d_loss_fake = criterion(dis(img_fake.detach()), label_fake) # detach是把gen部分从计算图中取出来不反向传播，千万别写括号外面啊
            d_loss = (d_loss_fake+d_loss_real)/2
            d_loss.backward()
            optim_dis.step()
        if(epoch_i%50==0):
            sample(gen, device=device)
        toc = time.time()
        print(f'epoch {epoch_i} g_loss: {g_loss.item()} d_loss: {d_loss.item()} time: {(toc - tic):.2f}s')
        
        gan_weights = {'gen': gen.state_dict(), 'dis': dis.state_dict()}
        torch.save(gan_weights, ckpt_path)


def sample(gen:Generator, device='cuda'):
    i_n = 5
    # for i in range(i_n*i_n):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        z = torch.randn(i_n * i_n, gen.nz, 1, 1, device=device)   # gen用的噪声向量
        x_new = gen(z)

        x_new = einops.rearrange(x_new, '(n1 n2) c h w -> (n2 h) (n1 w) c', n1 = i_n)
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite('dcgan_sample.jpg', x_new)


if __name__ == '__main__':

    ckpt_path = './model_dcgan.pth'
    device = 'cpu'
    image_shape = get_img_shape()
    nz = 100
    ngf = 64
    ndf = 64

    gen = Generator(nz = nz, ngf = ngf, nc = image_shape[0]).to(device)
    dis = Discriminator(ndf = ndf, nc = image_shape[0]).to(device)

    train(gen, dis, ckpt_path, device=device)
    gan_weights = torch.load('/path/to/combined_weights.pth')
    gen.load_state_dict(gan_weights['gen'])
    dis.load_state_dict(gan_weights['dis'])

    sample(gen, device=device)
