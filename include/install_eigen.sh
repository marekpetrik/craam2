#!/bin/bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar.bz2
tar xf eigen-3.3.8.tar.bz2 
rm -rf eigen3
mv eigen-3.3.8 eigen3
rm eigen-3.3.8.tar.bz2 
