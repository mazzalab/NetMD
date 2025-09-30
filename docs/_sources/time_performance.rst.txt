Time Performances
==================

.. raw:: html

   <style>
       table {
           font-size: 15px; /* Adjust the font size as needed */
       }
       table td {
           font-size: 12px; /* Adjust the font size as needed */
       }
   </style>

   <hr class="text-linebreak">


To give an idea of NetMD's time performance, we present the results from several runs on a machine equipped with 24 GB of RAM and a 13th Gen Intel(R) Core(TM) i7-13700F (2.10 GHz) CPU.

The experiments were run with different numbers of replicas (3, 5, and 8), different numbers of frames (100, 250, 350, 500, 750, 1000, and 1500), and two types of scale-free graphs (with a density of 0.01 and 0.03), each composed of 447 nodes. We used three Weisfeiler-Lehman iterations for the Graph2Vec embedding. The value reported in each cell is the mean time in seconds ± the standard deviation over five runs.

The table show the time performance in seconds for graphs with density of 0.01 (890 edges on average) on different number of replicas and frames.



.. list-table:: Time Performance in seconds for graph of density 0.01
    :header-rows: 1
    :widths: 5 10 10 10 10 10 10 10

    * - Replicas
      - 100
      - 250
      - 350
      - 500
      - 750
      - 1000
      - 1500
    * - 3
      - 8.42 ± 0.05
      - 33.9 ± 0.24
      - 56.49 ± 0.75
      - 91.31 ± 2.73
      - 181.66 ± 1.41
      - 263.69 ± 4.21
      - 399.63 ± 12.52
    * - 5
      - 14.8 ± 0.04
      - 48.05 ± 0.34
      - 83.3 ± 0.83
      - 148.05 ± 1.87
      - 314.0 ± 3.31
      - 464.22 ± 12.25
      - 572.09 ± 3.68
    * - 8
      - 23.87 ± 0.13
      - 77.69 ± 1.66
      - 125.48 ± 0.99
      - 233.66 ± 2.37
      - 472.09 ± 3.68
      - 785.42 ± 12.19
      - 1683.59 ± 26.17

.. image:: _static/img/timeplot1.png
   :alt: logscale line plot time performance NetMD Image
   :width: 100%
   :align: center

The table show the time performance in seconds for graphs with density of 0.03 (3080 edges on average) on different number of replicas and frames.

.. list-table:: Time Performance in seconds for graph of density 0.03
    :header-rows: 1
    :widths: 5 10 10 10 10 10 10 10

    * - Replicas
      - 100
      - 250
      - 350
      - 500
      - 750
      - 1000
      - 1500
    * - 3
      - 21.02 ± 0.3
      - 78.28 ± 0.83
      - 113.5 ± 0.92
      - 165.16 ± 2.96
      - 267.74 ± 11.21
      - 364.58 ± 16.12
      - 605.97 ± 14.14
    * - 5
      - 33.62 ± 0.15
      - 123.72 ± 4.87
      - 193.07 ± 2.22
      - 292.03 ± 11.32
      - 519.08 ± 17.92
      - 707.92 ± 19.73
      - 1003.39 ± 22.49
    * - 8
      - 53.39 ± 0.21
      - 206.6 ± 10.12
      - 310.33 ± 16.97
      - 469.0 ± 23.43
      - 806.09 ± 16.34
      - 1104.83 ± 29.63
      - 2158.54 ± 21.27

.. image:: _static/img/timeplot2.png
   :alt: logscale line plot time performance NetMD Image
   :width: 100%
   :align: center