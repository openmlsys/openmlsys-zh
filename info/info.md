## 环境安装
机器学习系统书籍部署在Github是依赖于d2lbook工具实现的。因此我们首先要安装d2lbook。
```bash
git clone git@github.com:d2l-ai/d2l-book.git
cd d2l-book
python setup.py install
```
使用d2lbook构建HTML需要安装`pandoc`, 可以使用`apt-get install pandoc`（如果是MacOS可以用Homebrew）和。
构建PDF时如果有SVG图片需要安装LibRsvg来转换SVG图片，安装`librsvg`可以通过`apt-get install librsvg`（如果是MacOS可以用Homebrew）。
当然构建PDF必须要有LaTeX，如安装[Tex Live](https://www.tug.org/texlive/).

## 编译HTML版本
在编译前先下载[openmlsys-zh](https://github.com/openmlsys/openmlsys-zh) 所有的编译命令都在改文件目录内执行。
```bash
 git clone git@github.com:openmlsys/openmlsys-zh.git
 cd openmlsys-zh
```
使用d2lbook工具编译HTML。
```
d2lbook build html
```

生成的html会在`_build/html`。

此时我们将编译好的html整个文件夹下的内容拷贝至openmlsys.github.io的docs发布。

需要注意的是docs目录下的.nojekyll不要删除了，不然网页会没有渲染。

## 编译PDF版本

编译pdf版本需要xelatex、librsvg2-bin（svg图片转pdf）和思源字体。在Ubuntu可以这样安装。

```
sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
```

```
wget https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SourceHanSansSC.zip
wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/SourceHanSerifSC_SB-H.zip
wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/SourceHanSerifSC_EL-M.zip

unzip SourceHanSansSC.zip
unzip SourceHanSerifSC_EL-M.zip
unzip SourceHanSerifSC_SB-H.zip

sudo mv SourceHanSansSC SourceHanSerifSC_EL-M SourceHanSerifSC_SB-H /usr/share/fonts/opentype/
sudo fc-cache -f -v
```


这时候可以通过 `fc-list :lang=zh` 来查看安装的中文字体。

同样的去下载和安装英文字体

```
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

sudo fc-cache -f -v
```

然后就可以编译了。

```
d2lbook build pdf
```

## 样式规范

贡献请遵照本教程的[样式规范](style.md)。

## 中英文术语对照

翻译请参照[中英文术语对照](terminology.md)。
