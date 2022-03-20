# Issue的label
目前我们的issue主要有如下label:

- great suggestion: 表示该issue是用户为本书的内容提供的写作建议，并且该建议是一个很好的建立
- discussion: 表示该issue是用户针对文章内容进行特定讨论，或用户对内容进行了建议并且该建议还处在商讨中
- to be confirmed: 表示该issue被assign给了章节作者，但是目前章节作者并没有回复处理这个issue
- confirmed: 表示该issue被章节作者已经确认
- fixed: 表示该issue相关的pr被approve/merge

常规而言，一个针对书籍内容校正的issue的状态变换应该为:

    to be confirmed ----> confirmed ----> fixed
