# Germany_Beer_Map Project

This Project is a Python-based project focused on efficiently associating beer bottle caps with their corresponding locations on a wooden map of Germany designed for displaying beer caps which was given to me as a present. Using computer vision techniques, it accurately identifies the positions of the holes in the wooden map from a photograph.

The core functionality involves creating a correlation between the beer bottle caps and their physical locations on the map. This is achieved by matching the list of beer producers with the detected holes. The optimisation process aims to minimize the cumulativethe spatial distance between each hole on the map (its respective geographic reference) and bottle cap placement.

Key Features:

-   Computer vision for precise detection of beer bottle cap locations on a wooden map.
-   A thin-spline transformation carried out on the wooden map onto a reference outline of Germany.
-   Geospatial correlation to establish links between hole cutouts for the bottle caps and their real-world geographic references.
-   Optimisation algorithm (Hungarian algorithm) to minimise the spatial distance between bottle cap placement and their brewery locations.
