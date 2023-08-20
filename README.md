# Turing-modifier-for-medical-AI-system

The Turing modifier presents an innovative framework that we have devised to enhance the fairness of medical AI systems. This approach facilitates the transformation of original X-ray images into attribute-neutral X-ray images. In comparison to unaltered X-ray images, training medical AI systems on attribute-neutral X-ray images can yield enhanced fairness.

In practice, the Turing modifier achieves attribute neutrality in X-ray images by modifying the image's attributes. The parameter α within the Turing modifier governs the extent of attribute alteration in an X-ray image, ranging from 0 to 1. When α equals 0, the Turing modifier refrains from altering the attribute. In contrast, an α value of 1 results in the attribute being edited to its opposite counterpart in the original image, such as changing from female to male or from young to old. Attribute-neutral X-ray images are created at α=0.5.

The subsequent video provides an introduction to the Turing modifier's performance in altering single or multiple attributes of X-ray images.


https://github.com/A-Big-Brain/Turing-modifier-for-medical-AI-system/assets/142569940/c814baa5-eced-40f3-94d2-8826d4c87908


This project encompasses three core components: the Turing modifier, the AI judge for the Turing test, and the disease diagnosis model. The Turing modifier's role is to produce attribute-neutral X-ray images. The AI judge, on the other hand, is tasked with discerning the original attributes of the modified X-ray images. Concurrently, the disease diagnosis model is trained using attribute-neutral X-ray images and serves to identify the findings within the X-ray images. Subsequently, we will provide detailed introductions to each of these three components.

## Chest X-ray image datasets
 
There are three chest X-ray image datasets involved in our project: ChestX-ray14, MIMIC-CXR, and CheXpert. They can be accessed by the following links:
|Dataset|link|
|--------------|------------|
|ChestX-ray14|https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345|
|MIMIC-CXR|https://physionet.org/content/mimic-cxr-jpg/2.0.0/|
|CheXpert|https://stanfordmlgroup.github.io/competitions/chexpert/|

Initially, you should download the datasets. Following that, you are required to preprocess each dataset into five **numpy** array files, which should be placed within the designated **test_data** folder. For the purpose of code testing, we have included smaller-scale files. To fully unleash the capabilities of the Turing modifier, the complete dataset needs to be downloaded. The five essential array files are:
|File name|Shape|Note|
|--------------|------------|--------------|
|*_img.npy|N×256×256|X-ray image data|
|*_info|N×M|the metadata for each X-ray image. the attribute is included in the file|
|*_lab|N×K|the label of each X-ray image|
|*_lab_na|K|the name of each label|
|*_div|N|the training/validation/test indexes of all X-ray images|

note: **N** is the number of X-ray images, **M** is the number of metadata variables, and **K** is the number of findings in each dataset.

Some X-ray images in jpg format are put in the folder **dataset_images**.

## Main python packages
**numpy**

**torch**

**torchvision**

**argparse**

**pickle**


## Turing modifier

All code of the Turing modifier is in the folder **Turing_modifier/py_script/**, and the result of the Turing modifier after each run is stored in the folder **Turing_modifier/save_results/**. All hyperparameters can be configured in the file **Turing_modifier/py_script/support_args.py**. Train a Turing modifier:
````python
python train.py
````
After the training is finished, A folder will be created in the folder **Turing_modifier/save_results/**. The running result, model parameters, and output images are stored in the new folder. The name of the new folder is in the following format:
````
(dataset name)_(modified attribute 1)_(modified attribute 2)_(batch size)_(epochs)_(lambda_1)_(lambda_2)_(update_lambda_rate)_(four random characters)
````
such as: **CheXpert_gender_age_3_50_100.0_10.0_0.0_AITu**, **MIMIC_gender_age_race_3_2_100.0_10.0_0.0_qVR8**.

Next, we should use the trained Turing modifier to generate attribute-neutral X-ray images. In the file **Turing_modifier/py_script/support_args.py**, configure the hyperparameter **edite_degree** to **0.5**. Then:
````python
python generate_with_degree.py
````
After the running is finished, a new folder will be created under the folder **dataset_images**. the name of the new folder is the same as that in the folder **Turing_modifier/save_results/**.

The generated X-ray images are stored as a **numpy** array file in the new folder. The name of the array file is in the following formate:
````
(dataset name)_(edite_degree×100).npy
````
You can change the hyperparameter **edite_degree** to another value, and the generated X-ray images are also stored in the folder.

## AI judge

All code of the AI judge is in the folder **AI_judge/py_script/**, and the result of the AI judge after each run is stored in the folder **AI_judge/save_results/**. All hyperparameters can be configured in the file **AI_judge/py_script/support_args.py**. Train an AI_judge:
````python
python main.py
````
After the training is finished, A folder will be created in the folder **AI_judge/save_results/**. The name of the new folder is in the following format:
````
(dataset name)_(identified attribute)_(model ID)_(batch size)_(epochs)_(training iteration)_(test iteration)_(four random characters)
````
such as: **ChestX-ray14_gender_2_25_5_20_20_xEnS**, **MIMIC_gender_2_25_5_20_20_aONW**.

The running result and model parameters are stored in the new folder.

## Disease diagnosis model

All code of the disease diagnosis model is in the folder **disease_diagnosis_model/py_script/**, and the result of the disease diagnosis model after each run is stored in the folder **disease_diagnosis_model/save_results/**. All hyperparameters can be configured in the file **disease_diagnosis_model/py_script/support_args.py**. Train a disease diagnosis model:
````python
python main.py
````
After the training is finished, A folder will be created in the folder **disease_diagnosis_model/save_results/**. The name of the new folder is the same as that of the folder in the Turing modifier.





