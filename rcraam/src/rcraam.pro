TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += ../../
INCLUDEPATH += /home/marek/R/x86_64-pc-linux-gnu-library/3.6/Rcpp/include
INCLUDEPATH += /usr/include/R
INCLUDEPATH += ../../include

SOURCES += \
    robust_algorithms.cpp \
    simulation.cpp
