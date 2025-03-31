This guide provides step-by-step instructions for setting up the necessary environments for this project using two methods: **pip** and **conda**. You can choose either method depending on your preference or system setup.

## **Using `pip` (with `requirements.txt`)**

This method uses `pip` to install dependencies listed in a `requirements.txt` file.

### Step 1: Install Dependencies
You can install the required dependencies from the `requirements.txt` file using the following command:

```
pip install -r requirements.txt
```

This will install all the libraries specified in the requirements.txt.

### Step 2: Verify the Installation
To verify the installation, check if the libraries are installed correctly by running:

```
pip list
```

This will display the installed packages.


## **Using `conda` (with `environment.yml`)**

This method uses conda to create an environment and install dependencies listed in an environment.yml file. If you prefer to use conda, follow these steps:

### Step 1: Create a Conda Environment
Run the following command to create a new conda environment from the `environment.yml` file:

```
conda env create -f environment.yml
```

This will create a new environment with the name specified in the `environment.yml` file and install all the required dependencies. The first line of the `yml` file sets the new environment's name. 

### Step 2: Activate the Conda Environment
Activate the newly created environment:

```
conda activate <environment_name>
```

Replace <environment_name> with the name of the environment, which can be found in the `environment.yml` file.

### Step 3: Verify the Installation
To verify that the environment is set up correctly, list the installed packages:

```
conda list
```

This will display all installed packages within the environment.

## Troubleshooting

### Error: Missing Dependencies
If you receive an error indicating that a package cannot be found or installed, ensure that you have the latest version of `pip` or `conda`. You can update pip using:

```
pip install --upgrade pip
```

Or update conda using:

```
conda update conda
```

### Conflict Between pip and conda
If you're using both pip and conda in the same environment, make sure conda installs the dependencies first, followed by pip to avoid conflicts. If necessary, manually install some dependencies using pip after creating the environment.