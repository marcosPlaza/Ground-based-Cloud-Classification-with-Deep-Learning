# Ground-based-Cloud-Classification-with-Deep-Learning

### About
---
Clouds are an indicator of climate and weather on earth. The ensemble of radiating effects can be affected by the presence of different cloud types. Therefore, identifying the type of cloud is important when characterising the climate of a region. Furthermore, the classification of the different types or genres of clouds has the clear objective of predicting the weather of a region slightly in advance. So, what we want to solve on this occasion is a cloud classification problem. Traditional cloud classification or identification relies heavily on the experience of observers and is very time-consuming.

![Cumulonimbus at the Observatory Fabra](https://github.com/marcosPlaza/Ground-based-Cloud-Classification-with-Deep-Learning/blob/main/cloudsfabra_Cb.JPG)

We propose to develop a neural network for accurate cloud classification on the ground. To this end, we will explore convolutional neural network architectures, a vision transformer, as well as other models that have greater flexibility from little data.

### About cloud classification
---
Cloud classification can be made according to various criteria:
* physical constitution,
* development,
* height and the relationship between vertical dimension and horizontal extension.

We will focus on classifying by it's **physical constitution**.

Clouds are continuously evolving, often ephemeral, and show an infinite variety of shapes. Moreover, it can be argued that no two clouds are identical. There are, however, a limited number of characteristic shapes, observed worldwide, which allow clouds to be grouped or classified. Specifically, the international cloud classification considers ten basic forms or genera, which form the backbone of the classification. Any cloud observed can be assigned to one and only one of these genera. Species and varieties are then distinguished, as well as other characteristics. This classification system is similar to the taxonomic classification of plants and animals, and similarly uses Latin names, although there are also equivalent names in spoken languages.
The intermediate forms between two genera or in transition between them are often observed. So this fact adds more difficulty in order to classify one cloud in one of the ten genera. In the following table are some useful descriptions in order to know the main features about them.

| Abbreviation | Name          | Description                                               |
|--------------|---------------|-----------------------------------------------------------|
| Ci           | Cirrus        | Fibrous, white feathery clouds of ice crystals            |
| Cs           | Cirrostratus  | Milky, translucent cloud veil of ice crystals             |
| Cc           | Cirrocumulus  | Fleecy cloud, cloud banks of small, white flakes          |
| Ac           | Altocumulus   | Grey cloud bundles, compound like rough fleecy cloud      |
| As           | Altostratus   | Dense, gray layer cloud, often even and opaque            |
| Cu           | Cumulus       | Heap clouds with flat bases in the middle or lower level  |
| Cb           | Cumulonimbus  | Middle or lower cloud level thundercloud                  |
| Ns           | Nimbostratus  | Rain cloud; grey, dark layer cloud, indistinct outlines   |
| Sc           | Stratocumulus | Rollers or banks of compound dark gray layer cloud        |
| St           | Stratus       | Low layer cloud, causes fog or fine precipitation         |
| Ct           | Contrails     | Line-shaped clouds produced by aircraft engine exhausts   |

### About data
---
* [Cirrus Cumulus Stratus Nimbus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD)
* [Swimcat](http://vintage.winklerbros.net/swimcat.html)
* We have built our own datasets with pictures taken from the ground at the Observatori Fabra in Barcelona. The first one with the ten cloud genera, called **Observatori Fabra Clouds dataset (FabraClouds)**, and then the second (**Observatori Fabra Swimcat dataset (FabraSwimcat)**) preserves the following 5 categories; (a) Clear Sky, (b) Patterned Clouds, (c) Thick Dark Clouds, (d) Thick White Clouds, (e) Veil Clouds.

### Implemented techniques for classification
---
* [Convolutional Neural Network](https://github.com/marcosPlaza/Ground-based-Cloud-Classification-with-Deep-Learning/blob/main/TrainingPipelineCNN.py).
* [Fine-tunning a classificator with MobileNetV2](https://github.com/marcosPlaza/Ground-based-Cloud-Classification-with-Deep-Learning/blob/main/TrainingPipelineCNN.py).
* Data Augmentation.
* [Vision Transformer](https://github.com/marcosPlaza/Ground-based-Cloud-Classification-with-Deep-Learning/blob/main/ViT_Clouds.ipynb).
* [Triplet-Loss function and siamese neural networks](https://github.com/marcosPlaza/Ground-based-Cloud-Classification-with-Deep-Learning/blob/main/CloudTriplet_WithClassifiers.ipynb).

### Contributions
---
Contributions are welcome! For bug reports or requests please [submit an issue](https://github.com/marcosPlaza/Ground-based-Cloud-Classification-with-Deep-Learning/issues).

### Contact
---
Feel free to contact me to discuss any issues, questions or comments.
* [Github](https://github.com/marcosPlaza)

### Acknowledgments
---
In first place, I want to express my gratitude to all of my supervisors for their outstanding assistance, dedication, and expertise in assisting me with this project.

I would especially like to thank **Jordi Vitrià** for patiently guiding me at all times and giving me the tools and the knowledge to overcome the obstacles that arose week after week. Also to **Gerard Gómez** for his support and for being the link of contact with **Alfons Puertas – Oservatori Fabra, RACAB**.
To Alfons Puertas for providing the magnificent photographs of the clouds, which although they have served to feed our models, have both a great scientific and artistic value.

I would want to express my gratitude to all of the professors who have con- tributed to the Fundamental Principles of Data Science Master’s Program at the University of Barcelona. I wouldn’t have developed the knowledge or inspiration for this project without this teaching.

### License
---
The content developed by Marcos Plaza is distributed under the following license:

    MIT License

    Copyright (c) 2022 Marcos Plaza González

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
