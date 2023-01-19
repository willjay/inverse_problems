#!bin/bash
data_root="/Users/theoares/Dropbox (MIT)/research/spectral/testing/inverse_problems"
name="two-peak-"           # either blank or string to identify the data.

id=$(date +%s)
dir="${data_root}/${name}${id}"
mkdir "${dir}"
echo "Writing to directory: ${dir}"

cd ../inverse_problems
python3 gen_data.py "${dir}/data.h5"

cd ../nevanlinna_cpp
make clean; make
./Nevanlinna "${dir}/data.h5" "${dir}/recon.h5"

cd ../inverse_problems
mkdir "${dir}/plots"
python3 plot_spectral.py "${dir}/recon.h5" "${dir}/plots/"

echo "Done running simulation."
