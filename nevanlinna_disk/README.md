# Nevanlinna C++ code
The main class declarations and methods for this project are found in `prec.hpp` (for extended precision numbers) `nevanlinna.hpp` (for specific functions to the Schur reconstruction). The `nevanlinna.cpp` file uses these header files to run the reconstruction. After configuring the makefile appropriately, one can run the executable by specifying the input Green's function data and the path to the output file. For example, to read in Green's function data at `${dir}/data.h5`, and to write the corresponding output to `${dir}/recon.h5`, you may use the following code snippet:
```
    root=PATH_TO_inverse_problems
    dir=PATH_TO_DATA_DIRECTORY
    cd ${root}/nevanlinna_cpp
    make clean; make
    ./Nevanlinna "${dir}/data.h5" "${dir}/recon.h5"
```
There are also additional tests run in the `${root}/tests` directory, which may be useful for playing around with the different functions in `nevanlinna.hpp` and `prec.hpp`. 

## Pipeline
There are three steps in the simulation pipeline:
1. Generate Green's function data in Python (the script `inverse_problems/gen_data.py`)
2. Read Green's function data into C++ and reconstruct the spectral function (the directory `nevanlinna_cpp/`).
3. Plot reconstructed spectral function in Python (the script `plot_spectral.py`).
The data input / output consists of HDF5 files, which have fields that are specified below. **To run the entire pipeline, just use the `scripts/run_nev_simulation.sh` script with the appropriate directories specified.**
```
    cd ${root}/scripts
    bash run_nev_simulation.sh
```
The script at `scripts/run_nev_sim_cpp.sh` just runs the C++ part of the code, assuming the input data file is already generated. This is mostly here because it was useful for testing the C++ code. 

#### Simulation file format
- Input and output stored in same folder `$dir` in h5 files. 
  - Input file is `${dir}/data.h5`, output file is `${dir}/recon.h5`
- For extended precision data, store in h5 files as strings with separate datasets for real and imaginary parts. 
- Input file datasets:
  - 'beta': Temporal extent of data.
  - 'freqs/imag': Imaginary part of Matsubara frequencies (should be purely imaginary)
  - 'ng/real': Real part of Green's function.
  - 'ng/imag': Imaginary part of Green's function.
- Output file datasets: all input file datasets, and in addition:
  - 'start' : Start of the real domain data (int)
  - 'stop' : End of the real domain data (int)
  - 'npts' : Number of points for real domain data (int)
  - 'beta' : Temporal size of lattice (int)
  - 'eta' : eta value (string, i.e. extended precision number)
  - 'recon_real' : Reconstructed (real part of) spectral function.
  - 'recon_imag' : Reconstructed (imaginary part of) spectral function.