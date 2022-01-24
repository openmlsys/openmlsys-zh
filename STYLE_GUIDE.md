# 样式规范
从LaTeX迁移到Markdown。工具列表：
* [Pandoc:](https://pandoc.org/getting-started.html) LaTeX转换为Markdown。
* [Inkscape:](https://inkscape.org/release/inkscape-1.1.1/) pdf图片文件转换成svg图片文件

书写内容：
* 页面划分：每章建一个目录，以一级子章节的维度建一个md页面，[参考地址](https://github.com/openmlsys/openmlsys-zh/tree/master/chapter_programming_interface) 。
* 图存放位置： openmlsys-zh/img/自己章节，如openmlsys-zh/img/ch01，把自己章节的图片全部放置在对应目录。

## 文本

* LaTeX文件转换Markdown文件
    * Linux下安装Pandoc命令：agt-get install pandoc
    * 使用Pandoc命令：pandoc -s example.tex -o example.md
* 使用Pandoc转换后需要注意修改的地方
    * 图片需要手动改
    * 公式部分可能会有不正确，需要注意
    * 代码部分需要手动改，样式如下：
 ```python
    ```python
    import os
    import argparse
    ```
```
  
## 图片

* 软件
    * 使用PPT制图，以pdf导出，再使用Inkscape转成svg。
    * Inkscape软件使用：
        * 用Inkscape打开pdf，渐变曲线精细度选择为精细。
        ![Inkscape打开PDF](./img/guide/step1.png)
        * 选中图片，我们可以看到图和其白底
        ![1](./img/guide/step2.png)
        * 随意找一块能选择的图，此时会出现周边有虚框
        ![2](./img/guide/step3.png)
        * 按住ctrl拉一下白色框，此时能将图片和白框分离出来，按Delete删除白框。
        ![3](./img/guide/step4.png)
        * 选择文件-文档属性-缩放页面内容-缩放页面到绘图或选区。
        ![4](./img/guide/step5.png)
        * 最后保存图片，用Pycharm看图片效果如下：无白色底，大小刚好框住整图。
        ![5](./img/guide/step6.png)
        
* 样式
    * 格式：
        * svg：自己绘制的图片需要用svg，注意去掉白底
        * png：一些常规举例图片不需去除白底的可以使用png
    * md里插入图片，大小选择800个像素，label自动生成，例如：
    ```python
        ![机器学习系统工作流](../img/ch02/workflow.svg)
        :width:`800px`
        :label:`img_workflow`
    ```
  
* 版权
    * 不使用网络图片
* 位置
    * 两张图不可以较邻近
        * 两张图拼一下
* 引用
    * 生成html后可以看到自动label的引用，手动引用（例如，图7.1）

