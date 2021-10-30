# MLP-classification
# MLP的简单实现

> train.py       是训练函数
>
> test.py         选用一个不在验证集的图片，通过保存的模型来验证结果
>
> main.py       训练主程序
>
> MLP.py        结果输出参考

选用海贼王的数据集作为示例，**data**路径为训练集的路径，**val**为验证集路径。

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(400*400 * 3, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

模型可由自己定义。

