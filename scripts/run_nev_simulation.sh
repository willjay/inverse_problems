#!bin/bash
data_root="/Users/theoares/Dropbox (MIT)/research/spectral/testing/inverse_problems"

# id=$(date +%s)
# fn_name="two_peak"
fn_name="three_peak_b64_p128_half_points"
parent_dir="${data_root}/${fn_name}"
mkdir "${parent_dir}"
echo "Parent directory: ${parent_dir}"

# Generate data
cd ../inverse_problems
python3 gen_data.py "${parent_dir}/data.h5" "${parent_dir}/"

# Reconstruct at different eta
for eta in "1e-2" "1e-4" "1e-6" "1e-8" "1e-10"
do
    recon_name="${fn_name}/eta_${eta}"           # either blank or string to identify the data.
    dir="${data_root}/${recon_name}"
    mkdir "${dir}"
    echo "Writing to directory: ${dir}"

    cd ../nevanlinna_cpp
    make clean; make
    ./Nevanlinna "${parent_dir}/data.h5" "${dir}/recon.h5" "${eta}"

    cd ../inverse_problems
    python3 plot_spectral.py "${dir}/recon.h5" "${dir}/"
done

echo "Done running simulation."
