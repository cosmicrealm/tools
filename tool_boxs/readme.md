## 简明的图像评价指标统计工具

计算 psnr, ssim, niqe, fid, lpips

用法见 test_evaluation.py 中示例，注意根据需要对待测试的文件夹进行 reindex，得的要计算的 hr 和 sr 数据路径能够一一对应

未考虑数据维度不一致时的处理，应该进行一些 resize 或者裁切操作

- psnr,ssim,niqe 参考  [basicsr](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/metrics)

- fid 的计算通过 [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

- lpips 的计算通过 [lpips](https://github.com/richzhang/PerceptualSimilarity)