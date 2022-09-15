# 

A BNS suite forward model and MNRE for Unbiased Inference of the Hubble Constant

## Dependencies
* **python** version 3.5 or more recent
* **numpy**
* **scipy**
* **matplotlib**
* **astropy**
* **gwtoolbox**
* **mosfit**
* **pytorch**
* **swyft**
* **healpy**
* **dustmaps**

## Files
* **config.py** - Contains the settings on variables relevant to the rest of the code, including BNS parameters and null return value of S.

* **distance.py** - Defines key functions used in evaluating $P(D_L, v)$ likelihood functions for each BNS merger.

* **sky.py** - References GW source localization file and queries EM follow-up from GWTreasureMap.

* **sforward.py** - Defines the kilonova forward model and evaluates whether simulated BNS mergers are observed in GW and EM.

* **sampler.py** - Generates S from a suite of BNS mergers defined in `events.py`.

* **events.py** - Defines various suites of BNS mergers. Booleans in the header are used to select the desired suite

* **datamerge.py** - Used in train.py to compute validation loss, and may be called on its own to calculate the validation loss on a pre-trained network.

* **findmax.py** - Applies wedge effects to data stored in the folder.

* **plot.py** - Removes files of a specified extension from database.

## Generating New Transformed Files
**fourier.py** offers three types of transformations which may be applied to existing 21cm boxes. These are **sweep**, **gaussian**, and **bar**. **sweep** is representative of "The Wedge", **gaussian** multiplies the map's Fourier profile by a Gaussian distribution, and **bar** removes a bar of a specified width in Fourier space.

To generate a transformed file from an existing data file, edit the main method of **fourier.py** to perform the transformation you desire. To perform a certain transformation, then set that argument to **True** in the main method. Each method has a corresponding variable which needs to be set before running, so set that as well. Then, navigate to /data in terminal and run the following line.
```
python fourier.py
```
