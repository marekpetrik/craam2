#!/bin/sh
cp -ru ../craam inst/include/
# make sure to remove the redundant config file
rm -f inst/include/craam/craam_config.hpp
cp -ru ../include/eigen3 inst/include
cp -ru ../include/rm inst/include
cp -u ../include/csv.h inst/include
