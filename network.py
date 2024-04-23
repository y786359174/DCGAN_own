import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):         # 这里Module是大写啊
    def __init__(self, ndf=64, nc=3):
        super(Discriminator,self).__init__()
        self.ndf = ndf      # 判别器d将图片转换成分数，首先经过的是channal维扩维，并且在每次下采样都会继续扩维，ndf是第一次扩维大小
        self.nc = nc        # 图片channal数
        self.net = nn.Sequential(                           # 这个sequential是把所有网络层叠在一起的工具，一般就在这里面按顺序写各种层
                    # 输入图片大小 (nc) x 32 x 32，输出 (ndf) x 16 x 16
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),            # 4*4卷积核输出本应该是61*61(偶数卷积核不常见，突然有点不会算了)，bias不使用，可以暂时不管他
                                                                        # pad是1，两边各补一个，那变成63*63了
                                                                        # 步长是2，正好此时是奇数，变成32*32了（把池化层干掉了）         
                    nn.LeakyReLU(0.2, inplace=True),                    # DCGAN相比于GAN的改进

                    # 输入(ndf) x 16 x 16，输出 (ndf*2) x 8 x 8
                    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),       #没啥说的，一样的原理，都除二，下面也一样
                    nn.BatchNorm2d(ndf * 2),                            # DCGAN相比于GAN的改进，括号里是channal维个数，一般normalization都有这个参数
                    nn.LeakyReLU(0.2, inplace=True),

                    # 输入(ndf*2) x 8 x 8，输出 (ndf*4) x 4 x 4
                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    # 输入(ndf*4) x 4 x 4，输出1 x 1 x 1
                    nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),         # 最后得到的分数，因为是1个所以就步长1来算了。
                    nn.Sigmoid()                                        # sigmoid既有非线性作用，值域还是0-1，很适合当分数

        )
    def forward(self, input):                               # 网络运行时就可以直接调用网络名做forward，backward一般都是它自己算
        return self.net(input).reshape(-1)                  # 一般确实是不加东西的，但是这里输出是bn*1*1*1,没法和label的一维向量对应上
    

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngf = ngf      # 生成器g的扩维值
        self.nz = nz        # 生成器输入一般是一个1维的噪声，一维的个数就是nz，这里把噪声看作是nz*1*1大小的张量
        self.nc = nc        # channal维个数
        self.net = nn.Sequential(
                    # 输入噪声向量Z，(ngf*4) x 4 x 4特征图
                    nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),       # 因为生成器生成的图片是上采样，宽高一点点变大，所以channal维就要从大到小（经验吧），就先从ngd*4开始了
                                                                                # 使用的是上采样中的转置卷积，相比于插值，是一种可学习参数的上采样方法
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # 输入(ngf*4) x 4 x 4，输出(ngf*2) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 用这个举例子看看是怎么算的，正常的转置卷积并不是卷积，但是这个实现方法是将转置卷积看作一个新的卷积，重新计算特征图并卷积。
                                                                                # 转置卷积，先看输入图变化，stride2，输入图变成7*7
                                                                                # 新的padding变为卷积核size-pad-1=2，输入图变为11*11
                                                                                # 卷积核仍为4*4，卷积后少了4-1=3，那卷积之后输出为8*8
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # 输入(ngf*2) x 8 x 8，输出(ngf) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # 输入(ngf) x 16 x 16，输出(nc) x 32 x 32
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh()                                                       # 据说生成任务用Tanh结尾效果好。你问我为什么判别器不用？别这么较真，判别器虽然在生成任务中但是不是用来生成的是用来算分的。
        )

    def forward(self, input):
        return self.net(input)