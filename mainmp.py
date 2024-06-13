import os
import time
import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
from network import Generator,  Discriminator , wgan_flag, weights_init_normal

import torch.distributed as dist ## DDP
# from torch.utils.data.distributed import DistributedSampler ## DDP
from torch.nn.parallel import DistributedDataParallel as DDP ## DDP

import cv2
import einops
import numpy as np
import random
# import tqdm as tqdm
Distributed_Flag = True # torchrun -m --nnodes=1 --nproc_per_node=2 --master_port 29500 mainmp

def train(gen:Generator, dis:Discriminator, ckpt_path, local_rank = 0, device = 'cuda'):
    print("using: ", device)    #你问我为什么写在这？因为main里总是有各种各样的地方会改device
    n_epochs = 10000
    batch_size = 256
    lr = 1e-5
    beta1 = 0.5
    k_max = 1
    gennz = gen.nz
    criterion = nn.BCELoss().to(device)         # 二元交叉熵损失，因为此时只有0和1的概率，且和为100%，所以可以不算向量，只算一个值（反正另一个也能间接求出来。
                                                # Loss = -w * [p * log(q) + (1-p) * log(1-q)]
                                                # 这东西也能放在cuda中啊
    data_dir = '../faces'
    if Distributed_Flag:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen).to(device)
        gen = DDP(gen, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis).to(device)
        dis = DDP(dis, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    dataloader = get_dataloader(batch_size, data_dir, num_workers=4, distributed=True)

    
    
    if not wgan_flag:
        optim_gen = torch.optim.Adam(gen.parameters(), lr, betas=(beta1, 0.999))    # betas 是一个优化参数，简单来说就是考虑近期的梯度信息的程度
        optim_dis = torch.optim.Adam(dis.parameters(), lr, betas=(beta1, 0.999))
    else:
        optim_gen = torch.optim.RMSprop(gen.parameters(),lr) #
        optim_dis = torch.optim.RMSprop(dis.parameters(),lr) #WGAN作者建议不要用Adam

    gen = gen.train()
    dis = dis.train()
    label_fake = torch.full((batch_size,), 0., dtype=torch.float, device=device)       # 真实图是1，虚假是0,需要注意,这里用的时候是计算loss，是小数，要用1. 和0.
    label_real = torch.full((batch_size,), 1., dtype=torch.float, device=device)       # 还有一件事，这里不能随意用batch_size，因为最后可能不满512，但是还是会继续算。得实时读取x的bn大小
                                                                    # 一般四个维度分别是bn c h w
                                                                    # 我把他从循环中挪出来并且检测x的bn不是batch_size就跳过
    k = k_max
    for epoch_i in range(n_epochs):
        if Distributed_Flag and local_rank >= 0:
            torch.distributed.barrier()
            dataloader.sampler.set_epoch(epoch_i)
        tic = time.time()
        score_r_avg = 0
        score_f_avg = 0
        if(epoch_i%100==0 and k < k_max):          # 让k缓慢的增大到k_max
            k+=1
        for x,_ in dataloader:
            img_real = x.to(device)

            # 训练Dis
            z = torch.randn(batch_size, gennz, 1, 1, device=device)   # gen用的噪声向量
            img_fake = gen(z)
            dis.zero_grad()
            score_r = dis(img_real)
            score_f = dis(img_fake.detach())
            if not wgan_flag:
                d_loss_real = criterion(score_r, label_real)
                d_loss_fake = criterion(score_f, label_fake) # detach是把gen部分从计算图中取出来不反向传播，千万别写括号外面啊
                d_loss = (d_loss_fake+d_loss_real)/2
            else:
                d_loss = -torch.mean(score_r-score_f) 
            d_loss.backward()
            optim_dis.step()
            if not wgan_flag:
                pass
            else:
                for p in dis.parameters():
                    p.data.clamp_(-0.01, 0.01)
            if(epoch_i%k==0):                           # 因为最开始dis梯度大，快收敛时gen梯度大，所以在这调节一下平衡
                # 训练Gen
                z = torch.randn(batch_size, gennz, 1, 1, device=device)   # gen用的噪声向量
                img_fake = gen(z)
                gen.zero_grad()                                            # 这个和optim_gen.zero_grad()有什么区别？现在没啥区别因为一个model就对应一个optim
                if not wgan_flag:
                    g_loss = criterion(dis(img_fake), label_real)               # woc xdm训练gen时千万记得别再傻呵呵用fake label当target了
                else:
                    g_loss = -torch.mean(dis(img_fake)) 
                g_loss.backward()
                optim_gen.step()
                score_r_avg = score_r.mean().item()
                score_f_avg = score_f.mean().item()
        if Distributed_Flag and local_rank >= 0:
            torch.distributed.barrier()
            # # 测试：每个GPU进程输出的模型第一层参数是相同的
            # for param in gen.parameters():
            #     print("    GPU{} Model param layer1=>".format(local_rank), param[0][0])
            #     break
            
        toc = time.time()  
        if Distributed_Flag==False or local_rank<=0:
            if(epoch_i%20==0):
                gan_weights = {'gen': gen.module.state_dict(), 'dis': dis.module.state_dict()}
                torch.save(gan_weights, ckpt_path)
                sample(gen, device=device)
        print(f'epoch {epoch_i} score_r_avg {score_r_avg:.4e} score_f_avg {score_f_avg:.4e} g_loss: {g_loss.item():.4e} d_loss: {d_loss.item():.4e} time: {(toc - tic):.2f}s device: {device}')
                
        
    if Distributed_Flag:    
        dist.destroy_process_group()       



sample_time = 0
def sample(gen:Generator, device='cuda'):
    global sample_time
    sample_time += 1
    i_n = 5
    # for i in range(i_n*i_n):
    gen = gen.to(device)
    gen = gen.eval()
    with torch.no_grad():
        z = torch.randn(i_n * i_n, 
                        gen.module.nz if Distributed_Flag else gen.nz,
                        1, 1, device=device)   # gen用的噪声向量
        x_new = gen.module(z) if Distributed_Flag else gen(z)         # 第一个epoch卡住的原因在这，我感觉是因为梯度被关闭了再gen，dpp就不会跑了

        x_new = einops.rearrange(x_new, '(n1 n2) c h w -> (n2 h) (n1 w) c', n1 = i_n)
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir,'./wgan_sample_%d.jpg' % (sample_time)), x_new)
    gen = gen.train()

save_dir = './data/wgan_faces64'


if __name__ == '__main__':

    if Distributed_Flag:
        # setup
        local_rank = int(os.environ["LOCAL_RANK"]) ## DDP   
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        # 
        local_rank = dist.get_rank()
        total_rank = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = 'cuda'
    # #
    # cv2.setNumThreads(0)
    # cv2.ocl.setUseOpenCL(False)
    
    # random reset
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


    ckpt_path = os.path.join(save_dir,'model_wgan.pth') 
    
    image_shape = get_img_shape()
    # nz = 100
    # ngf = 64
    # ndf = 64
    nz = 4096
    ngf = int(1024/8)
    ndf = int(1024/8)
    gen = Generator(nz = nz, ngf = ngf, nc = image_shape[0]).to(device)
    dis = Discriminator(ndf = ndf, nc = image_shape[0]).to(device)

    gen.apply(weights_init_normal)
    dis.apply(weights_init_normal)
    # gan_weights = torch.load(ckpt_path)
    # gen.load_state_dict(gan_weights['gen'])
    # dis.load_state_dict(gan_weights['dis'])

    train( gen, dis, ckpt_path, local_rank, device=device)

    # gan_weights = torch.load(ckpt_path)
    # gen.load_state_dict(gan_weights['gen'])
    # dis.load_state_dict(gan_weights['dis'])
    # sample(gen, device=device)
