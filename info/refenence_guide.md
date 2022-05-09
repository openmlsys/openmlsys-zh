# 参考文献引用方式
  下面为参考文献的引用，需要注意引用时前面需要有一个空格：
  1. 单篇参考文献
  这篇文章参考了论文 :cite:`cnn2015`
  2. 多篇参考文献可以用逗号分开
  这篇文章参考了论文 :cite:`cnn2015,rnn2015`
  3. 此时在对应bib中应该有如下参考文献
  @inproceedings{cnn2015,
	title = {CNN},
	author = {xxx},
	year = {2015},
	keywords = {xxx}
  }
  @inproceedings{rnn2015,
	title = {RNN},
	author = {xxx},
	year = {2015},
	keywords = {xxx}
  }

# 参考文献置于章节末尾方式
1.将章节所引用的全部参考文献生成一个chapter.pip，放置于references文件夹下。
如机器人系统章节将该章节参考文献全部放在rlsys.bib，并将其放在reference文件夹下。

```
参考文献目录

/references/rlsys.bib`
```
2.将对应章节参考文献引用添加至文章末尾处，如机器人系统章节在summary最后加上
   ```
	## 参考文献
	
	:bibliography:`../references/rlsys.bib`
   ```
