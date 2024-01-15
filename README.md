# Germany_Beer_Map Project

This project is a Python-based project focused on efficiently associating beer bottle caps with their corresponding locations on a wooden map of Germany designed for displaying beer caps which was given to me as a present. Using computer vision techniques, it accurately identifies the positions of the holes in the wooden map from a photograph.

The core functionality involves creating a correlation between the beer bottle caps and the physical locations of their corresponding breweries on the map. This is achieved by matching the list of beer producers with the detected holes. The optimisation process aims to minimize the cumulative spatial distance between each hole on the map (its respective geographic reference) and bottle cap placement.

## Key Features:

-   Computer vision for precise detection of beer bottle cap locations on a wooden map.
-   Rotation, aligning and scaling of the wooden map contour to a reference contour of Germany
-   A thin-plate spline transformation carried out on the wooden map onto a reference outline of Germany.
-   A Geospatial interpolation of the (image) coordinates of distinctive points on the wooden map contour to the corresponding longitude and latitude coordinates to generate a convex hull encompassing the centres of the hole cutouts.
-   Geospatial correlation to establish links between hole cutouts for the bottle caps and their real-world geographic references.
-   Optimisation algorithm (Hungarian algorithm) to find optimal pairings of bottle cap-brewery locations and hole cutouts to minimise aggregate spatial discrepancy.

## How to Run

To run the application, first clone the repo with git, then navigate to the project root directory and run the `main.py` script:

```bash
git clone https://github.com/daniel-jakob/Bottle_Cap_Map
cd Bottle_Cap_Map
python germany_beer_map/src/main.py
```

### Dependencies

The project's dependencies are listed in the `requirements.txt` file. To install the dependencies, use the following command:

```python
pip install -r requirements.txt
```

### Running the Project with Conda

1. First, make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. Create a new Conda environment using the `conda.yaml` file:

    ```sh
    conda env create -f conda.yaml
    ```

3. Activate the new environment:

    ```sh
    conda activate germany_beer_map
    ```

4. Navigate to the project root directory and run the `main.py` script

    ```sh
    python germany_beer_map/src/main.py
    ```

5. When you're done, you can deactivate the environment with:

    ```sh
    conda deactivate
    ```

### If looking to adapt for your own bottlecap map...

You must take a photo of your map, ideally as similar as possible to the one I used (found in `germany_beer_map/data/images/map.jpg`). I suggest a high contrast background. Replace the aforementined file with your picture. Find a reference outline image of your country online and replace the `map_ref.jpg` file. Change the `None` `circles = detect_circles(...)` line in `main.py`
`<xml attrib="someVal"></xml>`{:.language-xml}

## Testing

Unit tests are located in the `tests` directory. To run the tests, use the following command from the project root directory:

```bash
python & pytest germany_beer_map/tests
```

## License

This project is licensed under the terms of the [`LICENSE`](LICENSE) file.
