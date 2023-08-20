# Turing-modifier-for-medical-AI-system

The Turing modifier presents an innovative framework that we have devised to enhance the fairness of medical AI systems. This approach facilitates the transformation of original X-ray images into attribute-neutral X-ray images. In comparison to unaltered X-ray images, training medical AI systems on attribute-neutral X-ray images can yield enhanced fairness.

In practice, the Turing modifier achieves attribute neutrality in X-ray images by modifying the image's attributes. The parameter α within the Turing modifier governs the extent of attribute alteration in an X-ray image, ranging from 0 to 1. When α equals 0, the Turing modifier refrains from altering the attribute. In contrast, an α value of 1 results in the attribute being edited to its opposite counterpart in the original image, such as changing from female to male or from young to old. Attribute-neutral X-ray images are created at α=0.5.

The subsequent video provides an introduction to the Turing modifier's performance in altering single or multiple attributes of X-ray images.

https://github.com/A-Big-Brain/Turing-modifier-for-medical-AI-system/assets/142569940/c7b31f04-f5dc-4603-9083-377112d65876

This project encompasses three core components: the Turing modifier, the AI judge for the Turing test, and the disease diagnosis model. The Turing modifier's role is to produce attribute-neutral X-ray images. The AI judge, on the other hand, is tasked with discerning the original attributes of the modified X-ray images. Concurrently, the disease diagnosis model is trained using attribute-neutral X-ray images and serves to identify the findings within the X-ray images. Subsequently, we will provide detailed introductions to each of these three components.

## Turing modifier


先整体介绍，然后再介绍文件结构
需要的python包
把变化视频加上
