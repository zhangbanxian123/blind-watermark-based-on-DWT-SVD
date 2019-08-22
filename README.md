# blind-watermark-based-on-DWT-SVD
Blind watermarking algorithm based on dwt-svd
这是一个基于DWT-SVD的盲水印添加与提取算法，运行在jupyter上
步骤：
嵌入过程：
1.首先将lena图像进行一级小波分解得到低频段LL
2.将LL均分为4×4的网格
3.对内个网格做SVD分解得到奇异值s
4.对s[0]进行量化嵌入水印操作
5.svd分解逆过程，dwt逆过程，得到嵌入图像

提取过程：
1.首先将lena图像进行一级小波分解得到低频段LL
2.将LL均分为4×4的网格
3.对内个网格做SVD分解得到奇异值s
4.根据量化方法判断二值水印图像的0,1值
5.得到水印

