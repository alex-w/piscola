.. _sn_file:

Supernova File
========================

Below is an example of a file with data of a supernova in a format that PISCOLA understand. It is recomended that the file has the same name as the supernova with ``.dat`` extension (e.g. ``03D1au.dat``).

The file should contain the following information: 
	
	* Supernova name (``name``)
	* Redshift (``z``)
	* Right ascension in degrees (``ra``)
	* Declination in degrees(``dec``)
	* Times (``time``) 
	* Fluxes (``flux``)
	* Flux errors (``flux_err``)
	* Zero-points (``zp``)
	* Bands (``band``)
	* Magnitude systems (``mag_sys``). 
	
The units of ``time``, ``flux`` and ``flux_err`` do not matter as long as they are consistent (e.g. using the appropriate ``zp``). The name of the band needs to match the name of one of the files containing the transmission functions under the ``piscola/filters`` `directory <https://github.com/temuller/piscola/tree/main/src/piscola/filters>`_. The magnitude system should be one between ``AB``, ``BD17`` or ``Vega``.

The file shown below can be found in the Github repository `here <https://github.com/temuller/piscola/tree/master/data>`_:

.. parsed-literal::

	name z ra dec
	03D1au 0.50349 36.043209 -4.037469
	time flux flux_err zp band mag_sys
	52880.58 -0.405 0.858 27.5 Megacam_g AB
	52900.49 37.056 2.151 27.5 Megacam_g AB
	52904.6 41.256 1.243 27.5 Megacam_g AB
	52908.53 40.504 1.171 27.5 Megacam_g AB
	52930.39 10.207 1.048 27.5 Megacam_g AB
	52934.53 7.409 0.947 27.5 Megacam_g AB
	52937.55 6.692 1.607 27.5 Megacam_g AB
	52944.39 3.177 1.825 27.5 Megacam_g AB
	52961.45 3.462 1.72 27.5 Megacam_g AB
	52964.37 1.745 1.136 27.5 Megacam_g AB
	52992.33 1.935 1.352 27.5 Megacam_g AB
	52999.32 0.056 1.201 27.5 Megacam_g AB
	52881.5 2.305 1.313 27.5 Megacam_i AB
	52886.6 13.816 1.561 27.5 Megacam_i AB
	52900.53 77.632 1.581 27.5 Megacam_i AB
	52904.53 86.383 1.486 27.5 Megacam_i AB
	52908.6 92.071 1.967 27.5 Megacam_i AB
	52912.49 87.108 1.404 27.5 Megacam_i AB
	52915.62 87.47 2.148 27.5 Megacam_i AB
	52929.44 54.422 2.273 27.5 Megacam_i AB
	52933.47 51.478 1.225 27.5 Megacam_i AB
	52937.48 44.467 1.336 27.5 Megacam_i AB
	52942.47 35.972 1.675 27.5 Megacam_i AB
	52944.35 36.155 1.421 27.5 Megacam_i AB
	52961.39 18.974 1.81 27.5 Megacam_i AB
	52964.32 18.176 1.463 27.5 Megacam_i AB
	52972.27 13.399 2.305 27.5 Megacam_i AB
	52990.26 9.793 1.532 27.5 Megacam_i AB
	52992.24 5.762 1.586 27.5 Megacam_i AB
	52995.35 5.896 1.651 27.5 Megacam_i AB
	52999.24 6.814 1.575 27.5 Megacam_i AB
	53018.26 6.45 1.399 27.5 Megacam_i AB
	53022.28 8.736 1.781 27.5 Megacam_i AB
	53026.27 5.2 1.641 27.5 Megacam_i AB
	53210.6 -0.355 1.398 27.5 Megacam_i AB
	52881.54 2.79 0.872 27.5 Megacam_r AB
	52900.56 82.501 1.643 27.5 Megacam_r AB
	52904.57 95.883 1.515 27.5 Megacam_r AB
	52908.56 96.742 1.332 27.5 Megacam_r AB
	52914.57 89.938 2.131 27.5 Megacam_r AB
	52916.46 84.95 1.313 27.5 Megacam_r AB
	52929.52 50.409 4.391 27.5 Megacam_r AB
	52930.37 43.33 1.833 27.5 Megacam_r AB
	52933.5 30.415 2.311 27.5 Megacam_r AB
	52934.5 30.672 1.175 27.5 Megacam_r AB
	52937.57 23.791 1.97 27.5 Megacam_r AB
	52944.43 15.433 1.459 27.5 Megacam_r AB
	52961.48 5.974 1.837 27.5 Megacam_r AB
	52964.4 6.847 1.766 27.5 Megacam_r AB
	52968.36 8.371 2.161 27.5 Megacam_r AB
	52990.34 3.73 1.741 27.5 Megacam_r AB
	52992.29 4.732 1.367 27.5 Megacam_r AB
	52995.39 3.861 1.228 27.5 Megacam_r AB
	52999.28 2.483 1.228 27.5 Megacam_r AB
	53018.32 4.345 1.497 27.5 Megacam_r AB
	53022.32 1.554 3.804 27.5 Megacam_r AB
	53023.23 3.594 0.988 27.5 Megacam_r AB
	53026.34 3.191 1.154 27.5 Megacam_r AB
	53207.6 0.069 1.317 27.5 Megacam_r AB
	52881.56 0.774 4.835 27.5 Megacam_z AB
	52900.59 73.999 6.751 27.5 Megacam_z AB
	52909.61 98.789 6.468 27.5 Megacam_z AB
	52930.41 57.318 6.97 27.5 Megacam_z AB
	52938.51 45.324 7.563 27.5 Megacam_z AB
	52944.46 42.703 5.649 27.5 Megacam_z AB
	52962.31 28.561 4.598 27.5 Megacam_z AB
	52964.44 0.037 11.353 27.5 Megacam_z AB
	52999.36 3.495 9.014 27.5 Megacam_z AB
	53000.24 9.228 4.57 27.5 Megacam_z AB
