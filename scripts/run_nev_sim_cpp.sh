#!bin/bash
dir="/Users/theoares/Dropbox (MIT)/research/spectral/testing/inverse_problems/test-1673370143"
echo "Running c++ code with directory: ${dir}"
cd ../nevanlinna_cpp
make clean; make
./Nevanlinna "${dir}/data.h5" "${dir}/recon.h5"
