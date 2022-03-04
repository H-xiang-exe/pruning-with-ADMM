# 论文复现：A Systematic DNN Weight Pruning Framework Using ADMM

**论文地址**：[A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers (thecvf.com)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.pdf)



### 参考

[kaiqzhan/admm-pruning: Prune DNN using Alternating Direction Method of Multipliers (ADMM) (github.com)](https://github.com/kaiqzhan/admm-pruning)

[bzantium/pytorch-admm-pruning: Prune DNN using Alternating Direction Method of Multipliers (ADMM) (github.com)](https://github.com/bzantium/pytorch-admm-pruning)



### Pre Train

**Epochs:** 50

**Training Loss:** 0.10071072731167079

**Testing Loss:** Average loss: 1.2088

**Testing Accuracy:** 76%

### Train

**Epochs:** 30

**Training Loss:** 0.027573506912837425

**Testing Loss:** Average loss: 1.3904, Accuracy: 79%

### after pruning

**参数情况：**(7526, 3768, 36532, 15237, 15237, 34283, 15237, 15237, 34283, 15237, 30474, 137134, 60948, 60948, 137134, 60948, 60948, 137134, 60948, 60948, 137134, 60948, 121897, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 487587, 2194145, 975175, 975175, 2194145, 975175, 975175, 2194145, 975175, 19046)

**Testing Loss:** 22%

### Re Train

**Epochs:** 20

**训练集Loss:** 0.050077743986427475

**测试集Loss:** 1.5145, Accuracy: 74%

**总参数量：**20706496

**各层为0参数量：**(7526, 3768, 36532, 15237, 15237, 34283, 15237, 15237, 34283, 15237, 30474, 137134, 60948, 60948, 137134, 60948, 60948, 137134, 60948, 60948, 137134, 60948, 121897, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 243793, 548536, 243793, 487587, 2194145, 975175, 975175, 2194145, 975175, 975175, 2194145, 975175, 19046)=19258002

**剪枝率：**93%



**Loss变化图：**![loss](D:\LearningWorks\PythonWorks\admm-pruning\pytorch-mnist-model\loss.jpg)