## Name
ShapeDescriptionManagement - calculate eight descriptive statistics quantifying a class in terms of intra-class diversity, inter-class
interference, class imbalance, and data sparsity

## Set Up
+ (recommended) Install Miniconda.
+ Navigate into the root directory of the project.
+ Create a virtual environment:
   ```sh
   $ conda create -n ShapeDescriptionManagement python=3.8
   ```
+ Activate the virtual environment:
   ```sh
   $ conda activate ShapeDescriptionManagement
   ```
+ Install all dependencies:
   ```sh
   $ python -m pip install -r requirements.txt
   ```

## Synopsis
Enter a command following the scheme below in order to create a file `complexity_report.tsv`:
   ```sh
   $ python scripts/create_report.py [-h] [--source FILE] [--output DIR]
   ```

## Arguments
+ FILE
    + path to a directory that includes several data sets as nested directories of the form:

    DATASET_NAME/  
    ....|  
    ....|_ README.md  
    ....|  
    ....|_ eval/  
    ....|........|  
    ....|........|_ DATASET_NAME__TEST.csv  
    ....|........|  
    ....|........|_ DATASET_NAME__DEV.csv (optional)  
    ....|  
    ....|_ training/  
    ............|  
    ............|_ DATASET_NAME__FULL.csv  
  
+ DIR
    + directory that the complexity report will be saved to
    + if a file of the name `complexity_report.tsv` already exists, it will be overriden
    + defaults to current working directory
