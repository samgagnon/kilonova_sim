### DeSCI

Debiased Standard siren Cosmological Inference

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

* **sampler.py** - Generates S from a suite of BNS mergers defined in **events.py**.

* **events.py** - Defines various suites of BNS mergers. Booleans in the header are used to select the desired suite.

* **datamerge.py** - Used to combine the samples produced by **sampler.py**.

* **findmax.py** - Finds the null S return value for the BNS merger suite specified in **events.py**.

* **plot.py** - Performs posterior inference on two datasets specified in command line.

## Using the Code

First, specify a BNS suite in **events.py**. Then, run

```
python sampler.py tag
```

where `tag` specifies the tag you wish to affix to the output dataset file. Multiple samplers may be run in parallel. The code is currently written with the assumption that five samplers will be run in parallel, with the tags `1`, `2`, `3`, `4`, and `5`. These may be combined into a single dataset by placing them into a directory `b` and running

```
python datamerge.py b
```

This produces a combined dataset in a directory named `data archive`. For any given bias correction scheme, there should be three datasets. One fully-biased, one which accounts for GW anisotropy but not EM anisotropy, and one which accounts for both biases. Their corresponding posteriors may be produced by running

```
python plot.py d1 d2
```

where `d1` is one dataset and `d2` is the other. Typically, they should be paired as (fully-biased, GW-correction-only) or (GW-correction-only, fully-corrected).
