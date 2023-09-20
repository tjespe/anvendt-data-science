# anvendt-data-science

## Setting up the project for the first time

1. Create a virtual environment
   ```bash
   python3 -m virtualenv env
   ```
1. Activate the virtual environment
   ```bash
   source env/bin/activate
   ```
1. Install requirements
   ```bash
   pip install -r requirements.txt
   ```

After that, you should be able to run any file in the project.
**NB**: You only have to do the above steps 1 time. Follow the guide below for how to run the project later.

## Running the project after the initial setup

1. Activate the virtual environment
   ```bash
   source env/bin/activate
   ```

You should then be able to run the project.
If someone has added any new packages to the project since the last time you did anything, you might have to do

```bash
pip install -r requirements.txt
```

to make everything work.

## Adding new packages

1. Add the name of the package and its version to `requirements.txt`
1. Run
   ```bash
   pip install -r requirements.txt
   ```

Please **DO NOT** run `pip freeze > requirements.txt` to update the requirements file, because that fills it with lots of irrelevant stuff.
