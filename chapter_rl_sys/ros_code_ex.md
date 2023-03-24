## 案例分析：使用机器人操作系统

在这一章节中，我们将带领大家安装ROS2并配置好使用环境，然后再通过一些简单的代码示例来让大家更深入的了解如何使用ROS2和上一章节所介绍的概念。

在本章节以及本章后续的案例章节中，我们将使用ROS2 Foxy Fitzroy（笔者撰写时的最新的ROS2 LTS版本），Ubuntu Focal（20.04）和Ubuntu Focal系统所带的Python 3.8（笔者的Ubuntu Focal所带的是3.8.10）。
其中ROS2 Foxy Fitzroy和Ubuntu Focal是官方的搭配，而如果你采用debian安装的方式（官方推荐方式）来安装ROS2的话，则Python必须使用Ubuntu所带的Python3版本。
这是因为debian安装方式会将很多ROS2的Python依赖库以`apt install`（而非`pip install`）的方式安装到Ubuntu自带的Python3路径中去。
这也就是说，当你选定ROS2版本后，你所需的Ubuntu版本和Python版本也就随之确定了。

如果想要使用Python虚拟环境（virtual env）的话，也必须指定使用Ubuntu系统所带的Python解释器（interpreter），并在创建时加上`site-packages`选项。添加这个选项是因为我们需要那些安装在系统Python3路径中的ROS2的依赖库。

举例来说，对于`pipenv`用户，可以通过下面这条命令来创建一个使用系统Python3并添加了`site-packages`的虚拟环境。

```shell
pipenv --python $(/usr/bin/python3 -V | cut -d" " -f2) --site-packages
```

因为要使用系统Python3的原因，用`conda`创建的虚拟环境可能会出现各种不兼容的问题。

对于其它版本的ROS2，安装过程和使用方式基本相同。

在本章节以及本章后续的案例章节中，我们在合适的场合将用ROS2，Ubuntu和Python来分别指代ROS2 Foxy Fitzroy，Ubuntu Focal和Ubuntu Focal所带的Python 3.8。

本章节中的案例有参考ROS2的官方教程。这个官方教程讲解的非常详细，非常适合初学者入门ROS2。

### 安装ROS2 Foxy Fitzroy

在Ubuntu上安装ROS2相对简单，绝大多数情况跟随官方教程安装即可。

#### 系统区域（locale）需要支持UTF-8

在开始安装之前，我们需要先确保我们Ubuntu系统的区域（locale）已经设置成了支持UTF-8的值。
我们可以通过`locale`命令来查看目前的区域（locale）设置。
如果`LANG`的值是以`.UTF-8`结尾的话，则代表系统已经是支持UTF-8的区域（locale）设置了。
否则，可以使用下面的命令来将系统的区域（locale）设置为支持UTF-8的美式英语。
想设置成其它语言只需更改相应的语言代码即可。

```shell
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

#### 设置软件源

我们还需要将ROS2的软件源加入到系统中。我们可以通过下面这些命令完成这点。

```shell
sudo apt update && sudo apt install curl gnupg2 lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

#### 安装ROS2

现在我们可以开始安装ROS2了。我们可以先更新软件源缓存，然后再安装ROS2 Desktop版。这个版本包含了ROS2框架和大部分ROS2开发常用的软件库，如RViz等，因此是首选的版本。

```shell
sudo apt update
sudo apt install ros-foxy-desktop
```

另外，让我们再来安装两个额外的软件，`colcon`和`rosdep`。前者是ROS2的编译工具，后者可以帮助我们迅速安装一个ROS2工程所需的依赖库。

```shell
sudo apt-get install python3-colcon-common-extensions python3-rosdep
```

到此，我们已经安装好了ROS2。但是，如果想要使用它，我们还需要一个额外的环境设置步骤。

#### 环境设置

对于任意安装好的ROS2（和ROS）版本，我们需要source对应的setup脚本来为对应的版本设置好所需环境，然后才能开始使用其版本。

例如，对于刚安装好的ROS2 Foxy Fitzroy，我们可以在终端中执行下面的命令来设置好ROS2所需的环境。

```shell
source /opt/ros/foxy/setup.bash
```

如果你用的是bash以外的shell，你可以尝试将setup的文件扩展名改为对应shell的名字。例如zsh的用户可以尝试使用`source /opt/ros/foxy/setup.zsh`命令。

如果你不想每次使用ROS2之前都要输入上述命令，可以尝试将这条命令加入到你的`.bashrc`文件中去（或者是`.zshrc`或其它对应的shell文件）。这样，你以后的每个新命令行终端都会自动设置到ROS2所需的环境。

这种环境设置方式的好处在于你可以放心的安装多个不同版本的ROS2（和ROS），然后只需在需要时`source`对应版本的`setup.bash`文件，从而使用这个版本的ROS2并不受其它版本的干扰。

如果你是一个Python的重度用户，上面这种将`setup.bash`加入到`.bashrc`的方式可能会对你造成一些困扰。因为你的所有virtual env从此都会自动引入ROS2的环境设置，并且ROS2所包含的python libraries也会加入到你的virtual env的路径里面去。
我相信，你可能对于virtual env会检测到ROS2的库这种情况不会感到特别开心，即使这些库并不会被用到或破坏你virtual env中程序的运行。

解决这个问题的方法也很简单。当你准备主要用Python来开发一个ROS2项目时，你可以为这个项目新建一个virtual env，然后将`source /opt/ros/foxy/setup.bash`这条命令加入到这个virtual env的`activate`脚本中去。

注意！你可能需要将这条`source`命令添加到脚本结尾前一些的位置或脚本最开头，要不然当你进入（activate）virtual env时你有可能会遇到下面这个错误（例如，对于`pipenv`的用户就需要添加到脚本结尾处的`hash -r 2>/dev/null`这条命令之前而不是最末尾）。

```shell
Shell for UNKNOWN_VIRTUAL_ENVIRONMENT already activated.
No action taken to avoid nested environments.
```

#### 测试安装成功

当我们执行了上述的`source`命令之后，我们可以测试ROS2的安装以及环境设置时成功的。

我们只需在执行了`source`命令的命令行中执行`printenv | grep -i ^ROS`。输出的结果应该包含以下三个环境变量。

```shell
ROS_VERSION=2
ROS_PYTHON_VERSION=3
ROS_DISTRO=foxy
```

此外，我们可以新开两个执行了`source`命令的终端窗口，然后分别执行以下两条命令。

终端1:
```shell
ros2 run demo_nodes_cpp talker
```

终端2：
```shell
ros2 run demo_nodes_py listener
```

如果成功安装并执行了`source`命令的话，我们将会看到`talker`显示它正在发布消息，同时`listener`显示它听到了这些消息。

恭喜！您已经成功安装好了ROS2并配置到了环境。下面我们将会通过几个简单的案例来展示上章节中介绍过的ROS2的核心概念。

### ROS2节点和Hello World

在这一小节中，我们将会创建一个ROS2项目，并使用Python来编写一个Hello World案例，以便展示ROS2 Node的基本结构。

#### 新建一个ROS2项目

首先，在一个合适的位置新建一个文件夹。这个文件夹将是我们ROS2项目的根目录，同时也是上一章节中介绍过的“工作区”（Workspace）。这个工作区是我们自己创建的，所以它是一个Overlay Workspace。相对的，我们之前执行的`source`命令会帮我们准备好这个Overlay所基于的核心工作区（Underlay Workspace）。

假设我们创建了名为`openmlsys-ros2`的工作区。

```shell
mkdir openmlsys-ros2
cd openmlsys-ros2
```

然后让我们为这个工作区创建一个Python的虚拟环境（virtual env）并依照上面*环境设置*小节中所介绍的那样将`source`命令添加到虚拟环境对应的`activate`脚本中去。

**我们默认之后所有案例章节的命令都是在这个新建的虚拟环境中执行的。**

不同的虚拟环境管理工具会有不同的指令，因此这一步笔者没有提供可执行命令的示例，而是留给读者自行处理。

接下来，我们要在这个工作区文件夹内新建一个名为`src`的子文件夹。在这个子文件夹内，我们将会创建不同的ROS2的程序库（package）。这些程序库相互独立，但又会互相调用其他库的功能来达成整个ROS2项目想要达成的各种目的。

在创建好`src`文件夹后，我们可以尝试调用`colcon build`命令。`colcon`是ROS2项目常用的一个编译工具（build tool）。这个命令会尝试编译整个ROS2项目（即目前工作区内的所有的程序库）。在成功运行完命令后，我们可以发现工作区内多出了三个新文件夹：`build`，`install`和`log`。其中`build`内是编译过程的中间产物，`install`内是编译的最终产物（即编译好的库），而`log`内是编译过程的日志。

到此，我们已经新建好了一个ROS2项目的框架，可以开始编写具体的代码了。

#### 新建一个ROS2框架下的Python库

下面，让我们在`src`文件夹内新建一个ROS2的程序库。我们将在这个程序库内编写我们的Hello World案例。

```shell
cd src
ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs --node-name hello_world_node my_hello_world
```

`ros2`命令的`pkg create`子项可以帮助我们快速的创建一个ROS2程序库的框架。`build-type`参数指明了这是一个纯Python库，`dependencies`参数指明了这个库将会使用`rclpy`和`std_msgs`这两个依赖库，`node-name`参数指明了我们创建的程序库中会有一个名为hello_world_node的ROS2节点，而最后的`my_hello_world`则是新建程序库的名字。

进入新建好的程序库文件夹`my_hello_world`，我们可以看到刚运行的命令已经帮我们建好一个Python库文件夹`my_hello_world`。其与程序库同名，且内含`__init__.py`文件和`hello_world_node.py`文件。后者的存在是由于我们使用了`node_name`参数的原因。我们将在这个Python库文件夹内编写我们的Python代码。

除此之外，还有`resource`和`test`这两个文件夹。前者帮助ROS2来定位Python程序库，因此我们不需要管它。后者用来包含所有的测试代码，并且我们可以看到里面已经有了三个测试文件。

除了这三个文件夹外，还有三个文件，`package.xml`，`setup.cfg`和`setup.py`。

`package.xml`是ROS2程序库的标准配置文件。打开后我们可以发现很多内容已经预生成好了，但是我们还需填写或更新`version`，`description`，`maintainer`和`license`这几项的内容。在此笔者推荐大家每次新建一个ROS2库的时候都第一时间将这些信息补全。除了这些项，我们还能看到`rclpy`和`std_msgs`已经被列为依赖库了，这是因为我们使用了`dependencies`参数的原因。如果我们要添加或修改依赖库，可以直接在`package.xml`内的`depend`列表处修改。除了最常用的`depend`（同时针对build，export和execution），我们还有`build_depend`，`build_export_depend`，`exec_depend`，`test_depend`，`buildtool_depend`和`dec_depend`。关于`package.xml`的具体介绍可以参考此英文[Wiki Page](http://wiki.ros.org/catkin/package.xml)。

`setup.cfg`和`setup.py`都是Python库的相关文件，但是ROS2也会通过这两个文件来了解怎么安装这个Python库至`install`文件夹以及有哪些需要注册的entry points，即可以直接用ROS2命令行命令来直接调用的程序。我们可以看到在`setup.py`中的`entry_points`项的`console_scripts`子项中已经将`hello_world_node`这个名字设置为`my_hello_world/hello_world_node.py`这个Python文件中`main()`函数的别名。我们后续就可以使用ROS2命令行命令和这个名字来直接调用这个函数。具体方式如下:

```shell
# ros2 run <package_name> <entry_point>
ros2 run my_hello_world hello_world_node
```

后续如果需要添加新的entry point的话可以直接在此位置添加。

除了entry point需要关注之外，我们也需要及时将`setup.py`中的`version`，`maintainer`，`maintainer_email`，`description`和`license`项都更新好。

#### 第一个ROS2节点

让我们打开`my_hello_world/hello_world_node.py`这个Python文件，清空里面全部内容，以便于编写我们需要的代码。

首先，让我们引入必要的库：

```python
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
```

`rclpy`（ROS Client Library for Python）让我们能够通过Python来使用ROS2框架内的各种功能。而`Node`类则是所有ROS2节点的基类（Base Class），我们的节点类也需要继承这个基类。`std_msgs`则包含了ROS2预定义的一些用于框架内通信的标准信息格式，我们需要使用`String`这种消息格式来传递字符串信息。

接下来让我们定义我们自己的ROS2节点：

```python
class HelloWorldNode(Node):

    def __init__(self):
        super().__init__('my_hello_world_node')
        self.msg_publisher = self.create_publisher(String, 'hello_world_topic', 10)
        timer_period = 1.
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0
    
    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.count}'
        self.msg_publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1
```

如上所述，我们的节点类`HelloWorldNode`继承于`Node`基类。

在`__init__()`方法中，我们先调用基类的初始化方法，并通过这个调用将我们的节点命名为`my_hello_world_node`。接着我们创建一个信息发布者，它可以将字符串类型的信息发布到`hello_world_topic`这个主题上，并且会维持一个大小为10的缓冲区。再接着我们创建一个计时器，它会每秒钟调用一次`timer_callback()`方法。最后，我们初始化一个计数器，来统计总共有多少条信息被发布了。

在`timer_callback()`方法中，我们简单的创建一条带计数器的Hello World信息，并通过信息发布者发送出去。然后我们在日志中记录这次操作并将计数器加一。

定义好我们的HelloWorldNode类后，我们可以开始定义`main()`函数。这个函数就是我们之前在`setup.py`中看到的那个entry point。

```python
def main(args=None):
    rclpy.init(args=args)
    hello_world_node = HelloWorldNode()
    rclpy.spin(hello_world_node)
    hello_world_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

这个`main()`也比较简单。我们先通过`rclpy.init()`方法来启动ROS2框架。然后我们创建一个`HelloWorldNode`的实例。接着我们通过`rclpy.spin()`方法将这个实例加入到运行的ROS2框架中去，让其参与ROS2的事件循环并正确运行。`rclpy.spin()`是一个阻碍方法，它会一直运行一直到被阻止（例如ROS2框架停止运行）。这时候我们就会摧毁我们的节点，并且确保关闭ROS2框架。如果我们忘记了摧毁不再使用的节点，不用慌，garbage collector也会帮忙摧毁这个节点。

到此，我们创建了第一个ROS2节点！

#### 第一次编译和运行

让我们尝试编译新编写的这个库。这里，我们并不是真的要编译一个Python项目，而是将我们写的Python库安装到一个ROS2能找到的地方。

```shell
# cd <workspace>
cd openmlsys-ros2
colcon build --symlink-install
```

通过在运行这个编译命令，我们会编译工作区内`src`文件夹下所有的Python和C++库，并将编译好的C++库和Python库安装到`install`文件夹下。
通过指定`--symlink-install`这个选项，我们要求`colcon`对于Python库用生成symlink的方式来代替复制安装。这样一来，我们在`src`中做的后续改动都会直接反应到`install`中去，而不用一直反复执行编译命令。

在编译成功之后，编译好的库还不能直接使用。例如你现在执行`ros2 run my_hello_world hello_world_node`的话很有可能会得到`Package 'my_hello_world' not found`这样一个结果。

为了使用编译好的库，我们需要让ROS2知道`install`文件夹。具体来说，我们需要`source`在`install`文件夹下的`local_setup.bash`文件。即：

```shell
source install/local_setup.bash
```

有些机敏的读者可能会想到我们可以像之前添加那个`setup.bash`一样将这个`install/local_setup.bash`也加入到虚拟环境的`activate`脚本中去，这样我们就不用每次都单独`source`这个文件了。很可惜，这样会带来一些问题。

具体来说，一方面我们需要将这两个文件都`source`了（不管是通过`activate`脚本还是手动输入）才能顺利运行编译好的ROS2程序，但另一方面我们必须只`source`第一个`setup.bash`而不`source`第二个`local_setup.bash`才能顺利编译带有C++依赖项的纯Python的ROS2库。
在稍后面一点的案例中我们会看到，对于一个使用了自定义消息接口库（自己编写的C++库）的纯Python的ROS2程序库来说，必须只`source`第一个`setup.bash`而不`source`第二个`local_setup.bash`才能顺利编译。

在成功`source`了`install/local_setup.bash`之后，我们就可以尝试调用写好的节点了。

从现在开始，除非特殊说明，**新开一个终端窗口**都是指*新开一个确保`setup.bash`和`install/local_setup.bash`都已经被`source`了的终端窗口*，而在**工作区执行`colcon build`命令**则都是*在一个只`source`了`setup.bash`而忽略了`install/local_setup.bash`的终端窗口中执行此编译命令*。

```shell
ros2 run my_hello_world hello_world_node
```

我们应该会看到类似下面这样的信息：

```shell
[INFO] [1653270247.805815900] [my_hello_world_node]: Publishing: "Hello World: 0"
[INFO] [1653270248.798165800] [my_hello_world_node]: Publishing: "Hello World: 1"
```

我们还可以再新开一个终端窗口，然后执行`ros2 topic echo /hello_world_topic`。我们应该能看到类似下面的信息：

```shell
data: 'Hello World: 23'
---
data: 'Hello World: 24'
---
```

这代表着我们的信息确实被发布到了目标主题上。因为`ros2 topic echo <topic_name>`这条命令输出的就是给定名字的主题所接收到的信息。

恭喜！您已成功运行了您的第一个ROS2节点！

#### 一个消息订阅者节点

只是发布消息并不能组成一个完整的流程，我们还需要一个消息订阅者来消费我们发布的信息。

让我们在`hello_world_node.py`所在的文件夹内新建一个名为`message_subscriber.py`的文件，并添加以下内容：

```python
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MessageSubscriber(Node):

    def __init__(self):
        super().__init__('my_hello_world_subscriber')
        self.msg_subscriber = self.create_subscription(
            String, 'hello_world_topic', self.subscriber_callback, 10
        )
    
    def subscriber_callback(self, msg):
        self.get_logger().info(f'Received "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    message_subscriber = MessageSubscriber()
    rclpy.spin(message_subscriber)
    message_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

这个新添加的文件以及其中的消息订阅者节点类和上面的HelloWorldNode类十分相似，甚至更为简单些。我们只需要在初始化时通过基类初始化方法赋予节点`my_hello_world_subscriber`这个名字，然后创建一个消息订阅者来订阅`hello_world_topic`主题下的消息，并指定`subscriber_callback()`方法来处理接收到的消息。而在`subscriber_callback()`中，我们将接收到的消息记录进日志。`main()`方法则和HelloWorld节点类的基本一样。

在能正式使用这个新节点之前，我们需要将其添加成为一个entry point。为此，我们只需在`setup.py`的对应位置添加下面这行：

```python
'message_subscriber = my_hello_world.message_subscriber:main'
```

但是，添加完成之后在终端窗口运行`ros2 run my_hello_world message_subscriber`还是会得到`No executable found`这样的错误反馈。这是因为我们新增了一个entry point，必须重新编译整个ROS2项目才能让ROS2知道这个新增点。

让我们再次在工作区目录执行`colcon build --symlink-install`。在成功编译后，让我们新建两个终端窗口，都分别确保`source`好了两个`setup`文件。然后分别用`ros2`命令调用它们：

```shell
# in terminal 1
ros2 run my_hello_world hello_world_node
```

```shell
# in terminal 2
ros2 run my_hello_world message_subscriber
```

我们应该可以看到终端窗口1中会不断显示发布了第N号Hello World消息，而终端窗口2中则不断显示收到了第N号Hello World消息。

恭喜！你完成了一对ROS2节点，一个负责发送信息，一个负责订阅接受信息。

### ROS2参数

顺利完成上面的消息发布者和消息订阅者是个很好的开始，但是实际项目的节点不会这么简单。
至少，实际项目的节点会是参数化的。下面，就让我们一起看看怎样让一个节点读取一个参数。

让我们在`hello_world_node.py`所在的文件夹内新建一个名为`parametrised_hello_world_node.py`的文件，并添加以下内容：

```python
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class ParametrisedHelloWorldNode(Node):

    def __init__(self):
        super().__init__('parametrised_hello_world_node')
        self.msg_publisher = self.create_publisher(String, 'hello_world_topic', 10)
        timer_period = 1.
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0
        self.declare_parameter('name', 'world')
    
    def timer_callback(self):
        name = self.get_parameter('name').get_parameter_value().string_value
        msg = String()
        msg.data = f'Hello {name}: {self.count}'
        self.msg_publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    hello_world_node = ParametrisedHelloWorldNode()
    rclpy.spin(hello_world_node)
    hello_world_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

我们可以看到，这个新的参数化HelloWorld节点类和之前的HelloWorld节点类基本相同。
唯二的区别在于：1）这个新类在初始化方法中额外通过`self.declare_parameter()`方法来向ROS2框架声明新的节点实例会有一个名为`name`的参数，并且这个参数的初始值为`world`；2）这个新类在`timer_callback()`回调函数中尝试获取这个`name`参数的实际值，并以这个实际值来组成要发送的信息的内容。

让我们先将这个新文件的`main()`方法注册为一个新的entry point。
同样的，在`setup.py`中的相应位置加入下面这行即可。
然后别忘了在工作区根目录下执行`colcon build --symlink-install`来重新编译项目。

```python
'parametrised_hello_world_node = my_hello_world.parametrised_hello_world_node:main'
```

在编译完成之后，如果我们在终端中执行`ros2 run my_hello_world parametrised_hello_world_node`，我们将看到这个参数化HelloWorld节点将正常运行，并持续发布"*Hello World: N*"这样的信息。此时节点使用的是`world`这个初始值。

让我们在一个新的终端中执行`ros2 param list`，我们将看到下面的信息：

```shell
/parametrised_hello_world_node:
  name
  use_sim_time
```

这个信息表示`parametrised_hello_world_node`这个节点的确申明并使用一个`name`参数。
另外一个名为`use_sim_time`的参数是ROS2默认给与的一个参数，用来表示这个节点是否使用ROS2框架内部的模拟时间，而不是电脑的系统时间。

我们可以继续在这个终端中输入下面这个命令来将值`ROS2`赋予给`name`这个参数。

```shell
ros2 param set /parametrised_hello_world_node name "ROS2"
```

如果赋值成功的话，这个命令会返回`Set parameter successful`，并且我们可以在持续运行参数化HelloWorld节点的那个终端窗口内看到其发布的信息变为了"*Hello ROS2: N*"。

恭喜！你现在掌握了如何让ROS2节点（和其它类型的ROS2程序）使用参数的方法。

### 服务端-客户端服务模式

在上一章节中我们知道了ROS2框架除了发布者-订阅者这种通信模式，还有服务端-客户端这种模式。
在这一小节中，我们将通过一个简单的串联两个字符串的服务来演示如何使用这种模式。

#### 自定义的服务接口

在正式开始编写服务端和客户端的代码之前，我们需要先定义好它们之间进行沟通的信息接口。

ROS2框架内有三种类型的信息接口：

- 发布者-订阅者模式下的节点所用的**消息**类型接口（message/msg）：这种接口只负责单向的消息传递，也只用定义单向传递的信息的格式。
- 服务端-客户端模式下的服务节点所用的**服务**类型接口（service/srv）：这种接口需要负责双向的消息传递，即需要定义客户端发给服务端的请求的格式和服务端发给客户端的响应的格式。
- 动作模式下的动作节点所用的**动作**类型接口（action）：这种接口需要负责双向的消息传递以及中间的进展反馈，即需要定义动作发起节点发给动作节点的请求的格式，动作节点发给发起节点的结果的格式，以及动作节点发给发起节点的中间进展反馈的格式。

对于前面定义的那些HelloWorld节点，我们使用的是已经预定义好的`std_msgs`库内的`std_msgs.msg.String`类型的消息类型接口。
实际上，因为消息类型接口只负责定义单向的信息格式，我们很容易找到现成的符合我们需求的类型。
但是对于服务（service）和动作（action）来说，因为涉及到定义双向沟通的格式，很多时候我们需要自己定义一个接口类型。接下来，就让我们自行定义我们的字符串串联服务将要使用的服务类型接口。

首先，让我们在工作区的`src`文件夹内新建一个库来专门维护自定义的消息，服务和动作类型接口。

```shell
cd openmlsys-ros2/src
ros2 pkg create --build-type ament_cmake my_interfaces
```

这个新建的库是一个C++库，而不是Python库。这是因为ROS2的自定义接口类型只能以C++库的方式存在。新建好库之后，记得更新`package.xml`中的相关项。

下面，让我们在新建的`src/my_interfaces`文件夹内新建三个子文件夹：`msg`，`srv`和`action`。这是因为一般会将自定义的接口放到相对应的子文件夹中去，以方便维护。

```shell
cd my_interfaces
mkdir msg srv action
```

接着，让我们在`srv`子目录下创建我们想要定义的服务类型接口。

```shell
cd srv
touch ConcatTwoStr.srv
```

然后，让我们将以下内容添加到`ConcatTwoStr.srv`中去：

```
string str1
string str2
---
string ret
```

其中，`---`之上的是客户端发给服务端的请求的格式，而之下的是服务端发给客户端的响应的格式。

定义好了接口后，我们还需要更改`CMakeLists.txt`以便让编译器知道有自定义接口需要编译并能找到它们。让我们打开`my_interfaces/CMakeLists.txt`并在`if(BUILD_TESTING)`这行之前添加下面的内容。

```make
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ConcatTwoStr.srv"
)
```

上面这两段代码的主要作用是告诉编译器需要`rosidl_default_generators`这个库并生成我们指明的自定义接口。

在更新好`CMakeLists.txt`之后，我们还需要把`rosidl_default_generators`添加到`package.xml`中作为自定义接口库的依赖项。打开`package.xml`，在`<test_depend>ament_lint_auto</test_depend>`这行前添加下面内容。

```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

更新好`package.xml`后，我们就可以编译这个自定义接口库了。

```shell
cd openmlsys-ros2
colcon build --packages-select my_interfaces
```

上述命令中我们通过`--packages-select`选项指定了只编译`my_interfaces`这一个库从而节省时间，因为`my_hello_world`这个库目前并没有任何更改。另外，我们没有使用`--symlink-install`选项是因为这个自定义接口库是一个C++库，每次更改后必须重新编译。

在运行这次的编译命令时，读者有可能会遇到`ModuleNotFoundError: No module named 'XXX'`这类的错误（`XXX`可以是`em`，`catkin_pkg`，`lark`，`numpy`或其它Python库）。
遇到这类错误多半是因为所使用的Python虚拟环境并不是指向Ubuntu系统Python3或`site-packages`并没有被包含在虚拟环境中。
读者可能需要删除当前的虚拟环境并按照本章节开头所讲解的那样重新创建一个符合要求的虚拟环境。

我们可以通过在新的终端窗口运行`ros2 interface show my_interfaces/srv/ConcatTwoStr`来验证是否已经编译成功了。成功的话终端会显示自定义服务接口`ConcatTwoStr`的具体定义。

现在，我们定义好了需要使用的服务接口，下面可以开始编写我们的服务端和客户端了。

#### ROS2服务端

让我们在`hello_world_node.py`所在的文件夹内新建一个名为`concat_two_str_service.py`的文件，并添加以下内容：

```python
from my_interfaces.srv import ConcatTwoStr

import rclpy
from rclpy.node import Node


class ConcatTwoStrService(Node):

    def __init__(self):
        super().__init__('concat_two_str_service')
        self.srv = self.create_service(ConcatTwoStr, 'concat_two_str', self.concat_two_str_callback)

    def concat_two_str_callback(self, request, response):
        response.ret = request.str1 + request.str2
        self.get_logger().info(f'Incoming request\nstr1: {request.str1}\nstr2: {request.str2}')

        return response


def main(args=None):
    rclpy.init(args=args)
    concat_two_str_service = ConcatTwoStrService()
    rclpy.spin(concat_two_str_service)
    concat_two_str_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

我们可以发现，编写一个服务（Service）和编写一个一般的节点（Node）很相似，甚至它们都是继承自同一个基类`rclpy.node.Node`。在这个文件中，我们先从编译好的`my_interfaces`库中引入自定义的服务接口`ConcatTwoStr`。然后在服务端节点的初始化方法中通过`self.create_service()`创建一个服务器对象，并指明服务接口类型是`ConcatTwoStr`，服务名字是`concat_two_str`，处理服务请求的回调函数是`self.concat_two_str_callback`。而在回调函数`self.concat_two_str_callback()`中，我们通过`request`对象取得请求的`str1`和`str2`，计算出结果并赋值到`response`对象的`ret`上，并进行日志记录。我们可以看到，`request`和`response`对象的结构符合我们在`ConcatTwoStr.srv`中的定义。

另外别忘记了将此文件的`main()`方法作为一个entry point添加到`setup.py`中去。

```python
'concat_two_str_service = my_hello_world.concat_two_str_service:main'
```

#### ROS2客户端

让我们在`hello_world_node.py`所在的文件夹内新建一个名为`concat_two_str_client_async.py`的文件，并添加以下内容：

```python
import sys

from my_interfaces.srv import ConcatTwoStr
import rclpy
from rclpy.node import Node


class ConcatTwoStrClientAsync(Node):

    def __init__(self):
        super().__init__('concat_two_str_client_async')
        self.cli = self.create_client(ConcatTwoStr, 'concat_two_str')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = ConcatTwoStr.Request()

    def send_request(self):
        self.req.str1 = sys.argv[1]
        self.req.str2 = sys.argv[2]
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    concat_two_str_client_async = ConcatTwoStrClientAsync()
    concat_two_str_client_async.send_request()

    while rclpy.ok():
        rclpy.spin_once(concat_two_str_client_async)
        if concat_two_str_client_async.future.done():
            try:
                response = concat_two_str_client_async.future.result()
            except Exception as e:
                concat_two_str_client_async.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                concat_two_str_client_async.get_logger().info(
                    'Result of concat_two_str: (%s, %s) -> %s' %
                    (concat_two_str_client_async.req.str1, concat_two_str_client_async.req.str2, response.ret))
            break

    concat_two_str_client_async.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

```

相比于服务端，这个客户端较为复杂一点。在客户端节点的初始化方法中，我们先创建一个客户端对象，并指明服务接口类型是`ConcatTwoStr`，服务名字为`concat_two_str`。然后通过一个`while`循环，这个客户端将一直等待知道对应服务上线才会进行下一步。这个循环等待的技巧是很多客户端都会使用的。当服务端上线以后，初始化方法将创建一个服务请求对象的模板并暂存于客户端节点的`req`属性上。除了初始化方法，客户端节点还定义了另一个方法`send_request()`来读取程序启动时命令行的前两个参数，然后存入服务请求对象并异步发送给服务端。

而在`main()`方法中，我们先创建一个客户端并发送服务请求，然后通过一个`while`循环来等待服务返回结果并记录进日志。其中，`rclpy.ok()`是用来检测ROS2是否还在正常运行，以保证当ROS2在服务结束前就停止运行了的话，客户端这边不会陷入死循环。而`rclpy.spin_once()`和`rclpy.spin()`略有不同，后者会不断执行事件循环直到ROS2停止，而前者则只会执行一次事件循环。这也是为什么前者更适合用在这里，因为我们已经有了一个while循环了。另外我们可以看到，`concat_two_str_client.future`对象提供了很多方法来帮助我们确定目前服务请求的状态。

同样的，别忘记了将此文件的`main()`方法作为一个entry point添加到`setup.py`中去。

```python
'concat_two_str_client_async = my_hello_world.concat_two_str_client_async:main'
```

我们现在编写好了我们的服务端和客户端，让我们在工作区根目录下重新编译一边`my_hello_world`库。

```shell
cd openmlsys-ros2
colcon build --packages-select my_hello_world --symlink-install
```

然后让我们在两个新的终端窗口中分别运行以下命令。

```shell
# in terminal 1
ros2 run my_hello_world concat_two_str_client_async Hello World
```

```shell
# in terminal 2
ros2 run my_hello_world concat_two_str_service
```

如果一切正常的话，我们应该看到类似以下的信息。

```shell
# in terminal 1
[INFO] [1653525569.843701600] [concat_two_str_client_async]: Result of concat_two_str: (Hello, World) -> HelloWorld
```

```shell
# in terminal 2
[INFO] [1653516701.306543500] [concat_two_str_service]: Incoming request
str1: Hello
str2: World
```

恭喜！您现在已经了解如何在ROS2框架中新建自定义的接口类型和创建服务端节点和客户端节点了！

### 动作模式

在上一章节中我们了解了ROS2框架内的服务端-客户端模式。这样一来，我们只剩下动作（action）这一种模式了。
在这一小节中，我们将通过一个简单的逐个累加一个数列的每项元素来求和的动作来演示如何使用这种模式。

#### 自定义的动作接口

在正式开始编写动作相关的节点代码之前，我们需要先定义好动作的信息接口。

我们可以继续使用之前建好的`my_interfaces`库。
让我们在`my_interfaces/action`中新建一个`MySum.action`文件，并添加以下内容。

```
# Request
int32[] list
---
# Result
int32 sum
---
# Feedback
int32 sum_so_far
```

可以看到，整个信息接口十分简单。动作的请求信息只有一项类型为整数数列的项`list`，动作的最终结果信息只有一项类型为整数的项`sum``，而中间反馈信息则只有一项类型同为整数的项`sum_so_far`，用以计算到目前位置累加的和。

接下来，让我们在`CMakeLists.txt`中添加这个新的信息接口。具体来说只用将`"action/MySum.action"`添加到`rosidl_generate_interfaces()`方法内的`"srv/ConcatTwoStr.srv"`之后即可。

最后别忘了编译所做的更改：在工作区根目录中运行`colcon build --packages-select my_interface`。

#### ROS2动作服务器

让我们在`hello_world_node.py`所在的文件夹内新建一个名为`my_sum_action_server.py`的文件，并添加以下内容：

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from my_interfaces.action import MySum


class MySumActionServer(Node):

    def __init__(self):
        super().__init__('my_sum_action_server')
        self._action_server = ActionServer(
            self, MySum, 'my_sum', self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = MySum.Feedback()
        feedback_msg.sum_so_far = 0
        for elm in goal_handle.request.list:
            feedback_msg.sum_so_far += elm
            self.get_logger().info(f'Feedback: {feedback_msg.sum_so_far}')
            goal_handle.publish_feedback(feedback_msg)
        goal_handle.succeed()
        result = MySum.Result()
        result.sum = feedback_msg.sum_so_far
        return result


def main(args=None):
    rclpy.init(args=args)

    my_sum_action_server = MySumActionServer()

    rclpy.spin(my_sum_action_server)


if __name__ == '__main__':
    main()
```

对于这个动作服务器节点类，类似的，我们还是在其初始化方法中新建一个动作服务器对象，并指定了之前定义的`MySum`作为信息接口类型，`my_sum`是动作名字，`self.execute_callback`方法则作为动作执行的回调函数。

紧接着，我们在`self.execute_callback()`方法中定义了当接收到了一个新目标是应做什么处理。在这里，我们可以把一个目标当作之前定义的`MySum`信息接口里的`request`部分来处理，因为这里的目标就是包含了动作请求的目的的相关信息的结构体，即`request`部分所定义的部分。

当我们接收到一个目标后，我们先从`MySum`创建一个反馈消息对象`feedback_msg`，并将其`sum_so_far`项用作一个累加器。然后我们遍历目标请求中的`list`项里面的数据，并这些数据逐项进行累加。每当我们累加一项后，我们都会通过`goal_handle.publish_feedback()`方法发送一次反馈消息。最后，当全部计算完成后，我们通过`goal_handle.succeed()`来标记此次动作已经成功完成，并且通过`MySum`新建一个结果对象，填充结果值并返回。

在`main()`函数中，我们只需要新建一个动作服务器节点类的新实例，并调用`rclpy.spin()`将其加入事件循环即可。

最后别忘了将`main()`也添加成为一个entry point。我们只需在`setup.py`中适当位置添加下面行即可。

```python
'my_sum_action_server = my_hello_world.my_sum_action_server:main'
```

#### ROS2动作客户端

让我们在`hello_world_node.py`所在的文件夹内新建一个名为`my_sum_action_client.py`的文件，并添加以下内容：

```python
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from my_interfaces.action import MySum


class MySumActionClient(Node):

    def __init__(self):
        super().__init__('my_sum_action_client')
        self._action_client = ActionClient(self, MySum, 'my_sum')

    def send_goal(self, list):
        goal_msg = MySum.Goal()
        goal_msg.list = list

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected...')
            return
        
        self.get_logger().info('Goal accepted.')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sum}')
        rclpy.shutdown()
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sum_so_far}')


def main(args=None):
    rclpy.init(args=args)

    action_client = MySumActionClient()

    action_client.send_goal([int(elm) for elm in sys.argv[1:]])

    rclpy.spin(action_client)


if __name__ == '__main__':
    main()
```

我们可以看到，这个动作客户端节点类比上面的服务器节点类要稍许复杂些，这是因为我们要适当的处理发送请求，接受反馈和处理结果这三件事。

首先，还是类似的，我们在这个动作客户端节点类的初始化方法中新建一个动作客户端对象，并指定`MySum`作为消息接口类型和`my_sum`作为动作名称。

然后，我们申明`self.send_goal()`方法来负责生成并发送一个目标/请求。具体来说，我们先从`MySum`新建一个目标对象并将接收到的`list`参数赋值到目标对象的`list`属性上去。紧接着，让我们等待动作服务器准备就绪。当动作服务器准备就绪后，让我们异步发送目标并指定`self.feedback_callback`作为反馈信息回调函数。最后，我们设定`self.goal_response_callback`作为发送目标信息这个异步操作的回调函数。

在`self.goal_response_callback()`这个异步发送目标信息的回调函数中，我们先检查目标请求是否被接受了，并日志记录相关结果。如果目标请求被接受了的话，我们就通过`goal_handle.get_result_async()`来得到处理结果这个异步操作的`future`对象，并通过这个`future`对象将`self.get_result_callback`设定为最终结果的回调函数。

在`self.get_result_callback()`这个最终结果的回调函数中，我们就简单的获取累加结果并记录进日志。最后我们调用`rclpy.shutdown()`来结束当前节点。

相对的，在`self.feedback_callback()`这个反馈消息的回调函数中。我们仅仅简单的获取反馈信息的内容并记录进日志。值得注意的是，反馈消息的回调函数可能被执行多次，所以最好不要在其中写入太多的处理逻辑，而是尽量让其轻量化。

最后，在`main()`方法中，我们创建一个动作客户端节点类的实例，将命令行的参数转化为需要被求和的目标数列，最后调用动作客户端节点类实例的`send_goal()`方法并传入目标求和数列来发起求和请求。

同样的，别忘了将`main()`也添加成为一个entry point。我们只需在`setup.py`中适当位置添加下面行即可。

```python
'my_sum_action_client = my_hello_world.my_sum_action_client:main'
```

我们现在编写好了我们的动作服务器和动作客户端，让我们在工作区根目录下重新编译一遍`my_hello_world`库。

```shell
cd openmlsys-ros2
colcon build --packages-select my_hello_world --symlink-install
```

然后让我们在两个新的终端窗口中分别运行以下命令。

```shell
# in terminal 1
ros2 run my_hello_world my_sum_action_client 1 2 3
```

```shell
# in terminal 2
ros2 run my_hello_world my_sum_action_server
```

如果一切正常的话，我们应该看到类似以下的信息。

```shell
# in terminal 1
[INFO] [1653561740.000499500] [my_sum_action_client]: Goal accepted.
[INFO] [1653561740.001171900] [my_sum_action_client]: Received feedback: 1
[INFO] [1653561740.001644000] [my_sum_action_client]: Received feedback: 3
[INFO] [1653561740.002327500] [my_sum_action_client]: Received feedback: 6
[INFO] [1653561740.002761600] [my_sum_action_client]: Result: 6
```

```shell
# in terminal 2
[INFO] [1653561739.988907200] [my_sum_action_server]: Executing goal...
[INFO] [1653561739.989213900] [my_sum_action_server]: Feedback: 1
[INFO] [1653561739.989549000] [my_sum_action_server]: Feedback: 3
[INFO] [1653561739.989855400] [my_sum_action_server]: Feedback: 6
```

恭喜！您现在已经了解如何在ROS2框架中新建自定义的接口类型和创建动作服务端节点和动作客户端节点了！
