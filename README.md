# SeedSize-Shape_PlantCV
Wrokflow designed using the Python package PlantCV to process rice seed images and measure grain size and shape.

PlantCV version (3.6.2+8.g640a968.dirty)  

Seeds were scanned using a desktop scanner and a simple phenotyping set-up consisting of a black tray, two color standards, and a ruler as a size standard.  

Seed images were saved using prefixes that include phenotyping date (format MM-DD-YYYY), the scanner model, and a three-digit number indicating sample phenotyping order (e.g., the first phenotyped sample was saved as 03-26-2020_EpsonV600_001.jpg). This prefix structure allows the PlantCV pipeline to read and process all images.  

Briefly, for each RGB image, the pipeline first standardizes image exposure using the white standard color. Then, it separates the seeds from the background by applying threshold values in the LAB (L * a *b; where L = lightness, a = green/magenta, b = blue/yellow) color-space by so creating a binary image (mask) with white pixels as objects. Afterward, small holes in the objects are filled, and morphological filters are applied (erosion and dilation) to remove isolated noise pixels (e.g., awns if present), define object boundaries and edges, and, therefore, separate seeds in close proximity or touching each other. This filtered binary image is then used to mask the normalized RGB image, and objects within a defined region of interest are detected and analyzed for shape characteristics.  

The pipeline was then parallelized over all sample images as described at https://plantcv.readthedocs.io/en/stable/pipeline_parallel/. All trait estimates per seed and per sample are saved in JSON text files, which are then merged and converted to a final CSV table file using the accessory tool “plantcv-utils.py” implemented in PlantCV.  

More details on the PlantCV functions used in our pipeline can be found in the online user manual of PlantCV (https://plantcv.readthedocs.io/en/latest/).



