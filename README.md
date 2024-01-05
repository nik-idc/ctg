# General
This code implements an AI model that recognises images based on their cover.
There are 6 models that can be used:
- Newly trained model ('new')
- MobileNet ('mobile_net')
- MobileNetV2 ('mobile_net_v2')
- DenseNet121 ('dense_net_121')
- NASNetMobile ('nas_net_mobile')
- Resnet50 ('resnet50')

# Usage
Dependencies list incoming 😖😖😖😖

There are 2 files you can use:
## main.py
To use this you need to specify models using -n (python main.py -n new -n resnet50) or -a to use all models (python main.py -a). You can also specify the number of epochs using -e (default is 5)
## plots.py
The same as 'main.py' but without epochs for obvious reasons

# Credit
All credit for the data goes to https://github.com/uchidalab/book-dataset
