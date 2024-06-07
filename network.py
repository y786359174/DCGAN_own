import torch
import torch.nn as nn
import torch.nn.functional as F

img96_flag = False
img64_flag = True
wgan_flag = True

class Discriminator(nn.Module):         # 这里Module是大写啊
    def __init__(self, ndf:int=64, nc:int=3):
        super(Discriminator,self).__init__()
        self.ndf = ndf      # 判别器d将图片转换成分数，首先经过的是channal维扩维，并且在每次下采样都会继续扩维，ndf是第一次扩维大小
        self.nc = nc        # 图片channal数
        self.net1 = nn.Sequential(                           # 这个sequential是把所有网络层叠在一起的工具，一般就在这里面按顺序写各种层
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
                    nn.LeakyReLU(0.2, inplace=True)

                    
        )
        # if(img96_flag):
            # self.net2 = nn.Sequential(
            #         # 适配96*96图片
            #         nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #         nn.BatchNorm2d(ndf * 8),
            #         nn.LeakyReLU(0.2, inplace=True),

            #         nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False),       
        # )
        if(img64_flag):
            self.net2 = nn.Sequential(
                    # 适配96*96图片
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),       
        )
        else:
            self.net2 = nn.Sequential(
                    # 输入(ndf*4) x 4 x 4，输出1 x 1 x 1
                    nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),         # 最后得到的分数，因为是1个所以就步长1来算了。
                    
            )
        if not wgan_flag:
            self.net3 = nn.Sequential(
                    nn.Sigmoid()                                        # sigmoid既有非线性作用，值域还是0-1，很适合当分数
            )
        else:
            self.net3 = nn.Sequential(
                    nn.Identity()
            )
        self.net = self.net1+self.net2+self.net3

    def forward(self, input):                               # 网络运行时就可以直接调用网络名做forward，backward一般都是它自己算
        return self.net(input).reshape(-1)                  # 一般确实是不加东西的，但是这里输出是bn*1*1*1,没法和label的一维向量对应上
    

class Generator(nn.Module):
    def __init__(self, nz:int=100, ngf:int=64, nc:int=3):
        super(Generator, self).__init__()
        self.ngf = ngf      # 生成器g的扩维值
        self.nz = nz        # 生成器输入一般是一个1维的噪声，一维的个数就是nz，这里把噪声看作是nz*1*1大小的张量
        self.nc = nc        # channal维个数

        # if(img96_flag):
        #     self.net1 = nn.Sequential(
        #             # 适配96*96图片
        #             nn.ConvTranspose2d(nz, ngf * 8, 6, 1, 0, bias=False),                                                    
        #             nn.BatchNorm2d(ngf * 8),
        #             nn.ReLU(True),
        #             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),                                             
        #             nn.BatchNorm2d(ngf * 4),
        #             nn.ReLU(True),
        # )
        if(img64_flag):
            self.net1 = nn.Sequential(
                    # 适配64*64图片
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),                                                    
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),                                             
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
        )
        else:
            self.net1 = nn.Sequential(
                    # 输入噪声向量Z，(ngf*4) x 4 x 4特征图
                    nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),       # 因为生成器生成的图片是上采样，宽高一点点变大，所以channal维就要从大到小（经验吧），就先从ngd*4开始了
                                                                                # 使用的是上采样中的转置卷积，相比于插值，是一种可学习参数的上采样方法
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
            )

        self.net2 = nn.Sequential(
                    # 输入(ngf*4) x 4 x 4，输出(ngf*2) x 8 x 8                      h+h-1+2*(size-pad-1)
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

        self.net = self.net1+self.net2
    def forward(self, input):
        return self.net(input)
    
## 定义参数初始化函数
def weights_init_normal(m):                                    
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字. 
    if classname.find("Conv") != -1:                        ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("Linear") != -1:  # 添加线性层的初始化
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)