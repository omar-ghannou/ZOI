# ZOI

## Project Description: Automated Discovery and Annotation of Zones of Interest (ZOIs)

This Git repository contains the code implemented for the system described in submission #150 to SIGSPTIAL 2024.

The project focuses on the automated discovery and annotation of Zones of Interest (ZOIs) through a three-step process:

1. **ZOI Detection and Initial Shape Construction:** The first phase involves detecting potential ZOIs and constructing their initial shapes based on spatio-temporal data.
   
2. **Context Retrieval and Shape Refinement:** In the second phase, we retrieve contextual information from OpenStreetMap (OSM) using the initial shape of the detected ZOIs. This information is then used to provide a description of the ZOI as well as refine and accurately delineate the ZOI shapes.
   
3. **ZOI Annotation through Context Classification:** The final phase involves classifying the retrieved contextual information using NLP to provide meaningful annotations for each ZOI.

The following images demonstrate an example of a final ZOI shape along with its context.

![](https://github.com/omar-ghannou/ZOI/blob/main/images/ZOIs.png)
![](https://github.com/omar-ghannou/ZOI/blob/main/images/ZOIs2.png)

Please refer to the corresponding folder for demonstration videos of each part of the system.

