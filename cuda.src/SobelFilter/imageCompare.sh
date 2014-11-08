#!/bin/bash

# remove the binary and re-line
rm -rf SobelFilter
ln -s ../../bin/linux/release/SobelFilter

# run the image
./SobelFilter

# convert the output image
convert lena_shared.pgm lena_shared.jpg

# compare the images
compare -metric RMSE lena_shared.jpg lena_shared_orig.jpg out