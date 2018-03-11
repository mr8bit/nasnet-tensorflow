# NASNet-tensorflow

NASNet в tensorflow на основе tensorflow [slim](https://github.com/tensorflow/models/tree/master/research/slim) библиотеке.


## О NASNet и этом репозиторее 

NASNet до сих пор является современной архитектурой классификации изображений на наборе данных ImageNet (дата выпуска ArXiv - 21 июля 2017 года), точная точность обработки для крупной модели NASNet составляет 82,7. Подробнее о NASNet см. в документе  [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) by Barret Zoph etc.

С помощью этого репозитория вы сможете:

-  Обучить  NASNet с помощью настраиваемого набора данных для задачи классификации изображений с нуля. (If you want)

- Точная настройка NASNet (nasnet-a-large, nasnet-a-mobile) из модели Pre-train ImageNet для задачи классификации изображений.

-  Проверьте и оцените модель, которую вы обучили.

-  Разверните модель для вашего приложения или перенесите экстрактор функций на другие задачи, такие как обнаружение объектов (By yourself)

Подходить для тех, у кого есть солидный опыт работы с CNN, Tenserflow. У кого мало опыта работы или вообще нет [прочитайте "tensorflow slim walk through tutorial"](https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb).


## Зависимости
tensorflow >= 1.4.0

tf.contrib.slim

numpy


## Использование
### Скопировать репозиторий и войти в workspace.
```shell
git clone https://github.com/yeephycho/nasnet-tensorflow.git
cd nasnet-tensorflow
mkdir train pre-trained
```

### Загрузите и конвертируйте в формат TFRecord (эта часть совпадает с учебником tf.slim)
Много людей были заинтересованы в обучении Nasnet своими собственными данными.Я не уверен, что это хорошая идея для продвижения моего репозитория. Я используя набор данных, который предоставляется в учебнике Google. Если вы потратите некоторое время на код, вы сможете узнать, что может быть нелегко изменить сценарий генерации tfrecord самостоятельно, но вам очень легко изменить код шаблона, а цветовой набор данных - очень очень хороший шаблон для вас, чтобы изменить.

```shell
train_image_classifier.py
download_and_convert_data.py
datasets/dataset_factory.py
datasets/download_and_convert_flowers.py
datasets/flowers.py
```

Просто изменив несколько строк, вы сможете конвертировать свой набор данных в tfrecords.

Следующая инструкция приведет вас к созданию обучающих tfrecords.

Для каждого набора данных нам нужно загрузить необработанные данные и преобразовать их в TensorFlow's native
[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
формат. Каждый TFRecord содержит
[TF-Example](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/example/example.proto)
буфер протокола. Ниже мы продемонстрируем, как это сделать для набора данных Flowers.

```shell
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

Когда скрипт завершится, вы найдете несколько файлов TFRecord:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
...
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
...
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

Они представляют данные обучения и валидации, составленные по 5 файлов каждый.
Вы также найдете файл `$DATA_DIR/labels.txt`, который содержит сопоставление целых меток с именами классов.
Здесь я предоставляю вам удобную версию решения для генерации tfrecord.
Все, что вам нужно изменить, находится по адресу

```shell
datasets/customized.py
```
Строка 36, количество установленных наборов тренировок и проверки (теста).

Строка 39, количество полных классов.

```shell
datasets/convert_customized.py
```
Строка 61, номер проверки (теста).

```shell
# Создайте каталоги, имена которых называются labelN(метки, классы объектов), затем помещайте изображения в соответсвующие каталоги.
# ls /path/to/your/dataset/
# label0, label1, label2, ...
# ls /path/to/your/dataset/label0
# label0_image0.jpg, label0_image1.jpg, ...
#
# Название файла изображения не имеет большого значения.
DATASET_DIR=/path/to/your/own/dataset/

# Преобразование настроенных данных в tfrecords. Следует отметить, что  dataset_name должно быть  "customized"!
python convert_customized_data.py \
    --dataset_name=customized \
    --dataset_dir="${DATASET_DIR}"
```

### Поезд с нуля
```shell
DATASET_DIR=/tmp/data/flowers # /path/to/your/own/dataset/
TRAIN_DIR=./train

# For Nasnet-a-mobile
# --dataset_name=customized
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=nasnet_mobile

# For Nasnet-a-large
# --dataset_name=customized
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=nasnet_large
```

### Точная настройка с предварительно установленной контрольной точки ImageNet
```shell
# Этот скрипт загрузит предварительно подготовленную модель из Google, переместите файл вы предварительно подготовленную папку и распакует файл.
sh download_pretrained_model.sh

DATASET_DIR=/tmp/data/flowers # /path/to/your/own/dataset/
TRAIN_DIR=./train

# Для Nasnet-a-mobile
# --dataset_name=customized
CHECKPOINT_PATH=./pre-trained/nasnet-a_mobile_04_10_2017/model.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=nasnet_mobile \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=final_layer,aux_7 \
    --trainable_scopes=final_layer,aux_7

# Для Nasnet-a-large
# --dataset_name=customized
CHECKPOINT_PATH=./pre-trained/nasnet-a_large_04_10_2017/model.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=nasnet_large \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=final_layer,aux_11 \
    --trainable_scopes=final_layer,aux_11
```

### Ооценка
Можно загрузить загрузочную модель для набора цветов для nasnet [здесь](https://drive.google.com/open?id=1l_hhQoE6T4rc69OpRMJ8geQXzgTnXqUC) из google drive.

```shell
# Please specify the model.ckpt-xxxx file by yourself, for example
CHECKPOINT_FILE=./train/model.ckpt-29735

# For Nasnet-a-mobile
# --dataset_name=customized
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=/tmp/data/flowers \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=nasnet_mobile

# For Nasnet-a-large
# --dataset_name=customized
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=/tmp/data/flowers \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=nasnet_large
```

### Visualize the training progress
```shell
tensorboard --logdir=./train
```

## Reference
[Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

[tf.contrib.slim](https://github.com/tensorflow/models/tree/master/research/slim)

