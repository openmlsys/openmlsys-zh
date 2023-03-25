## 规划系统案例

在上一章节中，我们初步了解了机器人的规划系统。
这一章节中，我们将通过一个简单的案例来演示怎样结合ROS2和机器学习框架scikit-learn来完成一个我们设想的规划系统中的一个基本功能。
我们将使用和[感知系统案例](./perception_code_ex.md)这一章节类似的方法和结构来讲解本章节。

### 案例背景

假设我们想要帮某花园设计一款打理鸢尾花的园丁机器人。
很“碰巧”的是，这个小花园里面正好只有经典的[鸢尾花数据集](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)中的那三种鸢尾花，而且已经有人帮我们完成了一个“魔术般的”ROS2感知组件来自动的检测目标鸢尾花的花萼长和宽以及花瓣长和宽（Sepal Length, Sepal Width, Petal Length and Petal Width：鸢尾花数据集所需要的4个输入维度）。
同时因为机器人的性能限制，我们不能使用比较复杂的模型（例如神经网络）。
这种情况下，我们可以尝试使用经典的机器学习模型，例如决策树，来接受感知组件的结果并识别鸢尾花的类别，然后用一个映射表（mapping table）来查找出我们应该为机器人规划怎样的行为去执行。
当季节或情况改变时，花园的技术团队可以更新映射表来更改机器人的规划系统逻辑。

当然，上面的案例背景和解决方案都是为了生成一个简单的案例而设计的“非现实”的例子。
大家在现实项目中遇到的案例应该会复杂的多。
不过，我们任然希望这样一个简单的案例可以为大家带来些许价值。

让我们回到我们刚刚介绍的解决方案中。
在之前的感知系统的案例中，我们选择使用ROS2节点类来处理感知任务。
这是因为机器人会不断的接收到传感器的信号，而我们希望尽可能多的处理收到的信号。
而对于我们这一章节的案例来说，因为我们不一定需要不间断的进行新的规划，同时每一次规划我们都期待有一个结果，所以使用ROS2服务可能会是一个更好的选择。

和之前的案例类似，本章节的案例所使用的代码可以在本书相关的[ROS2案例代码库](https://github.com/openmlsys/openmlsys-ros2)中的`src/action_decider`文件夹内找到。

### 项目搭建

让我们继续沿用之前已经搭建好的ROS2项目框架。
和感知系统案例类似，我们只需在其中增加一个ROS2的Python库来实现我们想要的功能即可。
因此，让我们回到`src`目录下并创建此Python库。

```shell
cd openmlsys-ros2/src
ros2 pkg create --build-type ament_python --node-name action_decider_node action_decider --dependencies rclpy std_msgs scikit-learn my_interfaces
```

我们将`my_interfaces`添加为依赖项是因为我们需要为新的ROS2服务创建对应的消息类型接口。

在创建好Python库后，别忘了将`package.xml`和`setup.py`中的`version`，`maintainer`，`maintainer_email`，`description`和`license`项都更新好。

接下来，让我们在ROS2项目的Python虚拟环境中安装`scikit-learn`。
例如使用`pipenv`的用户可能会执行`pipenv install scikit-learn`这条命令。

### 添加消息类型接口

我们将要编写的新ROS2服务需要有它自己的服务消息接口。
让我们借用已有的`my_interfaces`库来放置这个新接口。

首先，让我们在`openmlsys-ros2/src/my_interfaces/srv`中新建一个名为`IrisData.srv`的文件并用下面的内容填充它。

```text
float32 sepal_length
float32 sepal_width
float32 petal_length
float32 petal_width
---
string action
```

我们可以看到，新的ROS2服务将会接受4个浮点值作为输入。
这4个浮点值分别为鸢尾花的花萼的长和宽还有花瓣的长和宽。
当规划完成后，服务会返回一个字符串。
这个字符串将会是机器人需要执行的动作的名称。

我们还需要在`my_interfaces`库的`CMakeLists.txt`文件中的相应位置（`rosidl_generate_interfaces`函数的参数部分）添加一行新的内容：

```cmake
"srv/IrisData.srv"
```

最后，别忘了在ROS2项目的根目录下执行`colcon build --packages-select my_interfaces`来重新编译`my_interfaces`这个库。

### 添加代码

之前创建Python库的命令应该已经帮我们创建好了`src/action_decider/action_decider/action_decider_node.py`这个文件。现在让我们用以下内容来替换掉此文件中已有的内容。

```Python
import os
import pickle

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from my_interfaces.srv import IrisData

from sklearn.datasets import load_iris
from sklearn import tree


def main(args=None):
    rclpy.init(args=args)
    action_decider_service = ActionDeciderService()
    rclpy.spin(action_decider_service)
    action_decider_service.destroy_node()
    rclpy.shutdown()


class ActionDeciderService(Node):

    IRIS_CLASSES = ['setosa', 'versicolor', 'virginica']

    IRIS_ACTION_MAP = {
        'setosa': 'fertilise',
        'versicolor': 'idle',
        'virginica': 'prune',
    }

    DEFAULT_MODEL_PATH = f'{os.path.dirname(__file__)}/../../../data/iris_model.pickle'

    def get_iris_classifier(self, model_path):
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as model_file:
                return pickle.load(model_file)
        self.get_logger().info(f"Cannot find trained model at '{model_path}', will train a new model.")
        iris = load_iris()
        X, y = iris.data, iris.target
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)
        with open(model_path, 'wb') as model_file:
            pickle.dump(clf, model_file)
        return clf

    def __init__(self):
        super().__init__('iris_action_decider_service')
        self.srv = self.create_service(IrisData, 'iris_action_decider', self.decide_iris_action_callback)
        self.iris_classifier = self.get_iris_classifier(self.DEFAULT_MODEL_PATH)
        self.get_logger().info('Iris action decider service is ready.')

    def decide_iris_action_callback(self, request, response):
        iris_data = [request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]
        iris_class_idx = self.iris_classifier.predict([iris_data])[0]
        iris_class = self.IRIS_CLASSES[iris_class_idx]
        response.action = self.IRIS_ACTION_MAP[iris_class]
        self.get_logger().info(
            f'Incoming request\nsepal_length: {request.sepal_length}\nsepal_width: {request.sepal_width}'
            f'\npetal_length: {request.petal_length}\npetal_width: {request.petal_width}'
            f'\niris class: {iris_class}'
            f'\ndecided action: {response.action}'
        )

        return response


if __name__ == '__main__':
    main()
```

细心的读者可能已经发现了，这段代码和我们之前创建的使用ROS2服务的服务端节点类的代码非常像。
实际上这段代码就是使用了同样的服务端节点类框架和一个新的服务来完成我们想要的功能。

这个服务端节点类的实例将被赋予名字`iris_action_decider_service`，它将提供一个名为`iris_action_decider`的服务并且这个服务期待`IrisData`格式的服务请求（即我们之前定义的消息类型接口的请求部分）。
当服务计算完成后，它将把结果返回给请求发起方。
这个结果是规划好的行为的名字并被封装到`IrisData`格式的服务结果中去（即我们之前定义的消息类型接口的结果部分）。

下面，让我们关注这个新节点类中的一些新细节。

首先，我们在新的服务端节点类`ActionDeciderService`中声明了三个类成员变量`IRIS_CLASSES`，`IRIS_ACTION_MAP`和`DEFAULT_MODEL_PATH`。
它们分别表示鸢尾花的类别标签，鸢尾花类别至机器人行动名称的映射表，和默认存放训练好的决策树模型的路径。

当我们的服务端节点类初始化时，它将调用`get_iris_classifier()`来读取训练好的决策树模型。
如果模型文件缺失，则会重新训练一个模型并保存。
这里我们把训练模型的代码放到了同一个节点内。
实际上，对于大型项目或大型模型，我们可以把模型训练和模型使用分开到不同的组件中去，并且它们可能在不同的时机运行。

当服务的回调函数`decide_iris_action_callback()`被调用时，服务将会使用训练好的模型和接收到的鸢尾花信息来预测鸢尾花的类别，然后通过查找映射表来决定机器人需要执行的动作。最后服务返回结果并进行日志记录。

至此，一个使用scikit-learn和决策树的简易“玩具级”规划组件就完成了。

### 运行及检测

下面，让我们尝试运行新写好的服务端节点类并检测它是否能正常运行。

首先，让我们编译这个新写的Python库。

```shell
cd openmlsys-ros2
colcon build --symlink-install
```

在成功编译之后，我们可以新开一个终端窗口并执行下面的命令来运行一个节点类实例。
记住，你可能需要先运行`source install/local_setup.zsh`来引入我们自己的ROS2项目。

```shell
ros2 run action_decider action_decider_node
```

如果你使用了Python虚拟环境，则可以尝试下面这条命令，而不是上面那条。背后具体的原因已在之前的案例章节叙述过。

```shell
PYTHONPATH="$(dirname $(which python))/../lib/python3.8/site-packages:$PYTHONPATH" ros2 run action_decider action_decider_node
```

当这个ROS2命令成功运行时，你应该能看到这行信息：`[INFO] [1655253519.693893500] [iris_action_decider_service]: Iris action decider service is ready.`。

在我们成功运行新的服务端节点后，让我们在一个新终端窗口中运行下面这行命令来测试新的服务是否能正常运行。同样的，你可能需要先运行`source install/local_setup.zsh`来引入我们自己的ROS2项目。

```shell
ros2 service call /iris_action_decider my_interfaces/srv/IrisData "{sepal_length: 1.0, sepal_width: 2.0, petal_length: 3.0, petal_width: 4.0}"
```

这里，我们用的`ros2 service call`命令是专门用来通过命令行调用一个ROS2服务的命令。其中服务请求的数据应该是字符串化的YAML格式数据。这个命令更多的信息可以通过`ros2 service call -h`来查阅。

一切顺利的话，执行完命令后不久，你应该就能在新窗口中很快看到类似这样的信息了：`response: my_interfaces.srv.IrisData_Response(action='prune')`。

### 小结

恭喜，你已经成功了解如何在ROS2项目中使用scikit-learn这样库并训练一个模型了！